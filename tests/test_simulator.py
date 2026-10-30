"""The simulator: healthy baselines stay inside limits, injectors leave their signature,
seeds reproduce, and scenario files carry labels."""

import json

import numpy as np
import pandas as pd
import pytest

from pondguard.config import load_settings
from pondguard.models import FarmerLog, Reading
from pondguard.paths import CONFIG_DIR
from pondguard.simulator.scenarios import load_specs, simulate
from pondguard.simulator.telemetry import baseline, do_crash

END = pd.Timestamp("2026-10-20 07:00")
SPECS = CONFIG_DIR / "scenario_specs.yaml"


def _dawn_minima(df: pd.DataFrame) -> list[float]:
    """Lowest DO between 00:00 and 08:00 of each day."""
    early = df[df["ts"].dt.hour < 8]
    return [g["do_mg_l"].min() for _, g in early.groupby(early["ts"].dt.date)]


@pytest.mark.parametrize("seed", [1, 2, 3, 703, 1201])
@pytest.mark.parametrize("day", [5, 42, 100, 140])
def test_baselines_sit_inside_the_thresholds(seed, day):
    limits = load_settings(env={}).thresholds
    df = baseline("P-1", END, day, seed)
    checks = {
        "do_mg_l": "do_mg_l",
        "ph": "ph",
        "temp_c": "temp_c",
        "salinity_ppt": "salinity_ppt",
        "tan_mg_l": "tan_mg_l",
        "nitrite_mg_l": "nitrite_mg_l",
        "alkalinity_mg_l": "alkalinity_mg_l",
    }
    for column, name in checks.items():
        spec = limits[name]
        if spec.min is not None:
            assert df[column].min() >= spec.min, f"{column} below {spec.min}"
        if spec.max is not None:
            assert df[column].max() <= spec.max, f"{column} above {spec.max}"


def test_baseline_has_a_day_night_oxygen_cycle():
    df = baseline("P-1", END, 42, 1)
    by_hour = df.groupby(df["ts"].dt.hour)["do_mg_l"].mean()
    assert by_hour.idxmin() in (3, 4, 5)  # low before dawn
    assert by_hour.idxmax() in (15, 16, 17)  # high mid-afternoon


def test_seeds_reproduce_and_differ():
    a, b, c = (baseline("P-1", END, 42, s) for s in (703, 703, 704))
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a["do_mg_l"], c["do_mg_l"])


def test_do_crash_makes_three_falling_dawn_minima():
    base = baseline("P-7", END, 42, 703)
    crashed = do_crash(base, start="2026-10-18T00:00", nights=3, depth=1.5)
    before, after = _dawn_minima(base)[-3:], _dawn_minima(crashed)[-3:]
    assert after[0] > after[1] > after[2]  # falling night after night
    assert after[2] < before[2] - 1.0  # the last night is about 1.5 mg/L lower
    assert after[2] < 4.0  # below the limit in thresholds.yaml
    pd.testing.assert_series_equal(base["ph"], crashed["ph"])  # only DO changes
    assert len(base) == len(crashed)


def test_injector_is_pure():
    base = baseline("P-7", END, 42, 703)
    snapshot = base.copy()
    do_crash(base, start="2026-10-18T00:00", nights=3, depth=1.5)
    pd.testing.assert_frame_equal(base, snapshot)


def test_scenarios_are_written_with_labels_and_valid_data(tmp_path):
    written = simulate(SPECS, tmp_path / "scenarios", tmp_path / "telemetry")
    assert len(written) == 41  # 40 golden scenarios plus the demo-only CRITIC-DEMO-01
    assert {"EHP-03", "HEALTHY-01", "AMBIG-01", "CRITIC-DEMO-01"} <= {p.stem for p in written}
    ehp = json.loads((tmp_path / "scenarios" / "EHP-03.json").read_text())
    assert ehp["pond_id"] == "P-07" and ehp["day_of_culture"] == 42
    assert ehp["label"]["top_condition"] == "white_faeces"
    assert ehp["label"]["action_level"] == "L2" and ehp["label"]["should_abstain"] is False
    FarmerLog(**ehp["farmer_logs"][0])  # validates symptoms and ranges
    rows = pd.read_csv(tmp_path / "telemetry" / "EHP-03.csv", parse_dates=["ts"])
    assert len(rows) == 7 * 24 and rows["ts"].iloc[-1] == END
    for record in rows.to_dict("records"):
        Reading(**record)  # every row is physically valid


def test_scenario_files_regenerate_identically(tmp_path):
    simulate(SPECS, tmp_path / "a", tmp_path / "ta")
    simulate(SPECS, tmp_path / "b", tmp_path / "tb")
    for name in ("EHP-03.json", "HEALTHY-01.json"):
        a = json.loads((tmp_path / "a" / name).read_text())
        b = json.loads((tmp_path / "b" / name).read_text())
        a.pop("telemetry_csv"), b.pop("telemetry_csv")  # paths differ by temp folder
        assert a == b
    assert (tmp_path / "ta" / "EHP-03.csv").read_bytes() == (
        tmp_path / "tb" / "EHP-03.csv"
    ).read_bytes()


def test_labels_cover_the_demo_behaviours():
    labels = {s["scenario_id"]: s["label"] for s in load_specs(SPECS)}
    assert (
        labels["HEALTHY-01"]["risk_level"] == "normal"
        and labels["HEALTHY-01"]["action_level"] == "L0"
    )
    assert labels["AMBIG-01"]["should_abstain"] is True
    assert labels["CRITIC-DEMO-01"]["top_condition"] == "vibriosis"


# --- the 40-scenario golden set ----------------------------------------------------------

import collections  # noqa: E402

from pondguard.simulator.telemetry import apply_events  # noqa: E402


def _golden():
    return [s for s in load_specs(SPECS) if s.get("golden", True)]


def test_golden_set_is_40_scenarios_five_per_group():
    golden = _golden()
    assert len(golden) == 40
    groups = collections.Counter(s["group"] for s in golden)
    assert set(groups.values()) == {5} and len(groups) == 8
    assert {
        "wssv",
        "ehp",
        "white_faeces",
        "ahpnd",
        "vibriosis",
        "water_quality_stress",
        "healthy",
        "ambiguous",
    } == set(groups)


def test_ids_ponds_and_seeds_are_unique_and_the_demo_scenario_is_outside_the_set():
    specs = load_specs(SPECS)
    for key in ("scenario_id", "pond_id", "seed"):
        values = [s[key] for s in specs]
        assert len(values) == len(set(values)), key
    demo = [s for s in specs if not s.get("golden", True)]
    assert [s["scenario_id"] for s in demo] == ["CRITIC-DEMO-01"]


def test_labels_follow_the_group_rules_of_the_spec():
    for s in _golden():
        label, group = s["label"], s["group"]
        assert label["risk_level"] in {"normal", "watch", "alert", "critical"}
        assert label["action_level"] in label["acceptable_action_levels"]
        assert label["should_abstain"] is (group == "ambiguous")
        if group in ("wssv", "ahpnd"):
            assert label["risk_level"] == "critical" and label["action_level"] == "L3"
        if group in ("ehp", "white_faeces"):
            assert label["risk_level"] == "alert" and label["action_level"] == "L2"
        if group == "healthy":
            assert label["risk_level"] == "normal" and label["top_condition"] is None
            assert not s["farmer_log"]["symptoms"] and not s["events"]
        if group == "ambiguous":
            assert label["risk_level"] == "watch" and label["top_condition"] is None
        if label["top_condition"]:
            assert label["top_condition"] in label["acceptable_conditions"]


def test_every_recipe_generates_valid_readings(tmp_path):
    simulate(SPECS, tmp_path / "s", tmp_path / "t")
    for csv in sorted((tmp_path / "t").glob("*.csv")):
        rows = pd.read_csv(csv, parse_dates=["ts"])
        assert len(rows) == 168
        for record in rows.to_dict("records"):
            Reading(**record)


@pytest.mark.parametrize(
    "event,column,check",
    [
        (
            {"type": "ammonia_buildup", "start": "2026-10-17T00:00", "days": 3, "slope": 0.25},
            "tan_mg_l",
            "rise",
        ),
        (
            {"type": "nitrite_buildup", "start": "2026-10-18T00:00", "days": 2, "slope": 0.5},
            "nitrite_mg_l",
            "rise",
        ),
        (
            {"type": "temp_drop", "start": "2026-10-19T10:00", "hours": 6, "delta": 4.5},
            "temp_c",
            "fall",
        ),
        (
            {"type": "salinity_drop", "start": "2026-10-19T08:00", "hours": 8, "delta": 7},
            "salinity_ppt",
            "fall",
        ),
    ],
)
def test_new_injectors_leave_their_signature_and_change_only_their_column(event, column, check):
    base = baseline("P-9", END, 40, 5)
    out = apply_events(base, [event])
    change = out[column] - base[column]
    assert (change.iloc[-1] > 0.5) if check == "rise" else (change.iloc[-1] < -3.0)
    assert change.iloc[0] == 0  # nothing happens before the event starts
    others = [c for c in base.columns if c not in (column, "ts", "pond_id")]
    pd.testing.assert_frame_equal(base[others], out[others])
