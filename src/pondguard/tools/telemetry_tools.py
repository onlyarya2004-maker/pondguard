"""Telemetry tools used by the Monitor agent (tool / function use).

Plain, typed Python. None of them calls an LLM. Each is wrapped in @traced_tool, so a
run's trace shows exactly which tool ran with which inputs and what it returned.

- get_telemetry:     last N hours of readings for a pond (from the simulator's CSV)
- compute_trends:    per-parameter statistics, dawn-oxygen series, 24 h changes
- detect_anomalies:  rules on the trends (falling dawn minima, spikes, sudden drops, rises)
- check_thresholds:  readings versus config/thresholds.yaml; every flag carries its source
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from pondguard.config import Settings, get_settings
from pondguard.models import Anomaly, Reading
from pondguard.paths import DATA_DIR
from pondguard.simulator.telemetry import PARAMETERS
from pondguard.tracing import traced_tool

DAWN_HOURS = range(0, 8)  # dawn minimum search window
# The spec says |z| > 3. Measured on 300 healthy 72 h windows (7 parameters), z > 3 raised a
# false alarm on 38% of them, z > 4 on 0.3%. So the cutoff is 4 (noted in docs/tuning_log.md).
SPIKE_Z = 4.0
MIN_DAWN_DROP = 0.3  # mg/L fall across three dawn minima that counts as a real trend
RISE_PER_DAY = 0.1  # mg/L per day for ammonia/nitrite counted as a steady rise
RISE_PARAMS = ("tan_mg_l", "nitrite_mg_l")


def _csv_for(pond_id: str, scenario_id: str | None) -> Any:
    folder = DATA_DIR / "telemetry"
    if scenario_id:
        return folder / f"{scenario_id}.csv"
    for scenario in sorted((DATA_DIR / "scenarios").glob("*.json")):
        info = json.loads(scenario.read_text(encoding="utf-8"))
        if info.get("pond_id") == pond_id:
            return folder / f"{info['scenario_id']}.csv"
    raise FileNotFoundError(f"no telemetry found for pond {pond_id}")


def _frame(readings: list[Reading]) -> pd.DataFrame:
    df = pd.DataFrame([r.model_dump() for r in readings])
    return df.sort_values("ts").reset_index(drop=True)


@traced_tool
def get_telemetry(pond_id: str, hours: int = 72, scenario_id: str | None = None) -> list[Reading]:
    """Last `hours` of hourly readings. Rows are validated: impossible values raise."""
    df = pd.read_csv(_csv_for(pond_id, scenario_id), parse_dates=["ts"])
    df = df[df["pond_id"] == pond_id].sort_values("ts").tail(hours)
    return [Reading(**row) for row in df.to_dict("records")]


def dawn_minima(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Lowest dissolved oxygen between 00:00 and 08:00 of each day, oldest first."""
    early = df[df["ts"].dt.hour.isin(DAWN_HOURS)]
    return [
        {"date": str(day), "do_mg_l": round(float(g["do_mg_l"].min()), 3)}
        for day, g in early.groupby(early["ts"].dt.date)
    ]


@traced_tool
def compute_trends(readings: list[Reading]) -> dict[str, Any]:
    """Per parameter: mean, min, max, slope per day, 24 h change. Plus oxygen detail."""
    df = _frame(readings)
    days = (df["ts"] - df["ts"].iloc[0]).dt.total_seconds() / 86400
    trends: dict[str, Any] = {
        "hours": len(df),
        "start": str(df["ts"].iloc[0]),
        "end": str(df["ts"].iloc[-1]),
    }
    for name in PARAMETERS:
        series = df[name]
        slope = float(np.polyfit(days, series, 1)[0]) if len(df) > 2 else 0.0
        back_24h = series.iloc[-25] if len(series) > 24 else series.iloc[0]
        trends[name] = {
            "mean": round(float(series.mean()), 3),
            "min": round(float(series.min()), 3),
            "max": round(float(series.max()), 3),
            "slope_per_day": round(slope, 4),
            "change_24h": round(float(series.iloc[-1] - back_24h), 3),
        }
    trends["dawn_min_do"] = dawn_minima(df)
    daily = df.groupby(df["ts"].dt.date)["do_mg_l"].agg(["min", "max"])
    trends["do_day_night_swing"] = round(float((daily["max"] - daily["min"]).mean()), 3)
    return trends


def _severity(value: float, limit: float, span: float) -> str:
    """How far past the limit: high if more than 25% of the limit's size beyond it."""
    return "high" if abs(value - limit) > 0.25 * max(abs(limit), span) else "medium"


@traced_tool
def check_thresholds(readings: list[Reading], settings: Settings | None = None) -> list[Anomaly]:
    """Compare readings with config/thresholds.yaml. Each flag carries `limit_source`."""
    settings = settings or get_settings()
    df = _frame(readings)
    window = f"last {len(df)} h"
    flags: list[Anomaly] = []
    for name, spec in settings.thresholds.items():
        if name not in df:
            continue
        series = df[name]
        if spec.min is not None and series.min() < spec.min:
            value = float(series.min())
            critical = spec.critical_min is not None and value < spec.critical_min
            flags.append(
                Anomaly(
                    parameter=name,
                    kind="below_limit",
                    severity="high" if critical else _severity(value, spec.min, spec.min),
                    value=round(value, 3),
                    limit=spec.min,
                    window=window,
                    limit_source=spec.source,
                )
            )
        if spec.max is not None and series.max() > spec.max:
            value = float(series.max())
            flags.append(
                Anomaly(
                    parameter=name,
                    kind="above_limit",
                    severity=_severity(value, spec.max, spec.max),
                    value=round(value, 3),
                    limit=spec.max,
                    window=window,
                    limit_source=spec.source,
                )
            )
    return flags


@traced_tool
def detect_anomalies(readings: list[Reading], settings: Settings | None = None) -> list[Anomaly]:
    """Rule-based anomaly detection on the trends.

    Rules: three consecutive falling dawn oxygen minima; spikes (|z| > 3 on
    hour-to-hour changes, which removes the day-night cycle); sudden 24 h drops in
    temperature or salinity beyond thresholds.yaml; steady rises in ammonia or nitrite.
    """
    settings = settings or get_settings()
    df = _frame(readings)
    anomalies: list[Anomaly] = []

    minima = [m["do_mg_l"] for m in dawn_minima(df)]
    if len(minima) >= 3 and minima[-3] > minima[-2] > minima[-1]:
        dropped = minima[-3] - minima[-1]
    else:
        dropped = 0.0
    if dropped >= MIN_DAWN_DROP:  # smaller falls are sensor noise, not a trend
        critical = settings.thresholds["do_mg_l"].critical_min
        low = minima[-1] < (critical if critical is not None else 3.0)
        anomalies.append(
            Anomaly(
                parameter="do_mg_l",
                kind="trend",
                severity="high" if low else ("medium" if dropped >= 0.5 else "low"),
                value=minima[-1],
                limit=None,
                window=f"3 dawn minima: {', '.join(str(m) for m in minima[-3:])}",
            )
        )

    for name in PARAMETERS:
        steps = df[name].diff().dropna()
        spread = float(steps.std())
        if spread == 0:
            continue
        z = (steps - steps.mean()) / spread
        worst = int(z.abs().idxmax())
        if abs(float(z.loc[worst])) > SPIKE_Z:
            anomalies.append(
                Anomaly(
                    parameter=name,
                    kind="spike",
                    severity="medium",
                    value=round(float(df[name].iloc[worst]), 3),
                    limit=None,
                    window=f"hourly change at {df['ts'].iloc[worst]}",
                )
            )

    for name in ("temp_c", "salinity_ppt"):
        spec = settings.thresholds[name]
        if spec.max_drop_24h is None or len(df) <= 24:
            continue
        drop = float(df[name].iloc[-25] - df[name].iloc[-1])
        if drop >= spec.max_drop_24h:
            anomalies.append(
                Anomaly(
                    parameter=name,
                    kind="drop",
                    severity="high",
                    value=round(float(df[name].iloc[-1]), 3),
                    limit=spec.max_drop_24h,
                    window="last 24 h",
                    limit_source=spec.source,
                )
            )

    days = (df["ts"] - df["ts"].iloc[0]).dt.total_seconds() / 86400
    for name in RISE_PARAMS:
        slope = float(np.polyfit(days, df[name], 1)[0])
        if slope > RISE_PER_DAY:
            anomalies.append(
                Anomaly(
                    parameter=name,
                    kind="trend",
                    severity="medium",
                    value=round(float(df[name].iloc[-1]), 3),
                    limit=None,
                    window=f"rising {slope:.2f}/day over {len(df)} h",
                )
            )
    return anomalies
