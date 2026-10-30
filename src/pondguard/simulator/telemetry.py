"""Hourly pond telemetry: a healthy baseline plus event injectors (spec §6.6).

Agentic concept: reproducible, labelled test data. The simulator gives the Monitor
agent real tool inputs without real sensors. Every scenario has a fixed seed, so
the same recipe always produces the same readings.

Baselines use day-night cycles plus Gaussian noise. Dissolved oxygen is low before
dawn (minimum about 04:00) and high mid-afternoon. The numbers are tuned so a
healthy pond stays inside config/thresholds.yaml.

Event injectors are pure functions: they take a DataFrame and return a changed copy.
Injectors: do_crash, ammonia_buildup, nitrite_buildup, temp_drop, salinity_drop.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PARAMETERS = [
    "do_mg_l",
    "ph",
    "temp_c",
    "salinity_ppt",
    "tan_mg_l",
    "nitrite_mg_l",
    "alkalinity_mg_l",
]
WINDOW_DAYS = 7
DAWN_HOUR = 4  # baseline oxygen minimum


def baseline(
    pond_id: str, end: pd.Timestamp, day_of_culture: int, seed: int, days: int = WINDOW_DAYS
) -> pd.DataFrame:
    """`days` of hourly readings ending at `end`, for a healthy pond."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range(end=end, periods=days * 24, freq="h")
    h = ts.hour.to_numpy() + ts.minute.to_numpy() / 60
    day = day_of_culture - (end - ts).days  # day of culture for each row
    n = len(ts)
    df = pd.DataFrame(
        {
            "pond_id": pond_id,
            "ts": ts,
            "do_mg_l": 6.5 + 1.8 * np.sin(2 * np.pi * (h - 10) / 24) + rng.normal(0, 0.12, n),
            "ph": 8.0 + 0.25 * np.sin(2 * np.pi * (h - 10) / 24) + rng.normal(0, 0.03, n),
            "temp_c": 29 + 1.5 * np.sin(2 * np.pi * (h - 9) / 24) + rng.normal(0, 0.15, n),
            "salinity_ppt": 18 + rng.normal(0, 0.3, n),
            "tan_mg_l": 0.3 + 0.002 * day + rng.normal(0, 0.02, n),
            "nitrite_mg_l": 0.1 + rng.normal(0, 0.01, n),
            "alkalinity_mg_l": 120 + rng.normal(0, 2, n),
        }
    )
    return clip_to_physical_ranges(df)


def clip_to_physical_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Keep values inside the ranges the Reading model accepts."""
    out = df.copy()
    limits = {
        "do_mg_l": (0, 20),
        "ph": (4, 11),
        "temp_c": (10, 40),
        "salinity_ppt": (0, 60),
        "tan_mg_l": (0, 20),
        "nitrite_mg_l": (0, 20),
        "alkalinity_mg_l": (0, 400),
    }
    for name, (low, high) in limits.items():
        out[name] = out[name].clip(low, high)
    return out


def do_crash(df: pd.DataFrame, start: str, nights: int, depth: float) -> pd.DataFrame:
    """Progressive dawn-oxygen crash.

    Night k (0-based) of `nights`, starting at `start`, lowers dissolved oxygen
    around the dawn minimum by depth * (k+1) / nights mg/L, so the dawn minima
    fall night after night and the last one is `depth` below baseline.
    """
    out = df.copy()
    first = pd.Timestamp(start)
    for k in range(nights):
        centre = first + pd.Timedelta(days=k, hours=DAWN_HOUR)
        hours_away = (out["ts"] - centre).dt.total_seconds() / 3600
        bump = np.exp(-0.5 * (hours_away / 2.5) ** 2)
        out["do_mg_l"] = out["do_mg_l"] - depth * (k + 1) / nights * bump
    return clip_to_physical_ranges(out)


def _hours_since(df: pd.DataFrame, start: str) -> pd.Series:
    """Hours elapsed since `start` for each row (0 before it starts)."""
    return ((df["ts"] - pd.Timestamp(start)).dt.total_seconds() / 3600).clip(lower=0)


def _buildup(df: pd.DataFrame, column: str, start: str, days: float, slope: float) -> pd.DataFrame:
    """Linear rise of `slope` per day for `days` days from `start`, then held."""
    out = df.copy()
    elapsed = _hours_since(out, start).clip(upper=days * 24)
    out[column] = out[column] + slope * elapsed / 24
    return clip_to_physical_ranges(out)


def ammonia_buildup(df: pd.DataFrame, start: str, days: float, slope: float) -> pd.DataFrame:
    """Total ammonia nitrogen climbs `slope` mg/L per day for `days` days from `start`."""
    return _buildup(df, "tan_mg_l", start, days, slope)


def nitrite_buildup(df: pd.DataFrame, start: str, days: float, slope: float) -> pd.DataFrame:
    """Nitrite climbs `slope` mg/L per day for `days` days from `start`."""
    return _buildup(df, "nitrite_mg_l", start, days, slope)


def _drop(df: pd.DataFrame, column: str, start: str, hours: float, delta: float) -> pd.DataFrame:
    """Falls by `delta` over `hours` hours from `start`, then stays lower."""
    out = df.copy()
    progress = (_hours_since(out, start) / hours).clip(upper=1.0)
    out[column] = out[column] - delta * progress
    return clip_to_physical_ranges(out)


def temp_drop(df: pd.DataFrame, start: str, hours: float, delta: float) -> pd.DataFrame:
    """Water temperature falls by `delta` C over `hours` hours from `start` (a cold snap)."""
    return _drop(df, "temp_c", start, hours, delta)


def salinity_drop(df: pd.DataFrame, start: str, hours: float, delta: float) -> pd.DataFrame:
    """Salinity falls by `delta` ppt over `hours` hours from `start` (heavy rain)."""
    return _drop(df, "salinity_ppt", start, hours, delta)


INJECTORS = {
    "do_crash": do_crash,
    "ammonia_buildup": ammonia_buildup,
    "nitrite_buildup": nitrite_buildup,
    "temp_drop": temp_drop,
    "salinity_drop": salinity_drop,
}


def apply_events(df: pd.DataFrame, events: list[dict]) -> pd.DataFrame:
    for event in events:
        params = {k: v for k, v in event.items() if k != "type"}
        injector = INJECTORS.get(event["type"])
        if injector is None:
            raise ValueError(f"unknown event type: {event['type']}")
        df = injector(df, **params)
    return df
