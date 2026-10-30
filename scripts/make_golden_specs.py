"""Append the 37 new golden-set recipes to config/scenario_specs.yaml.

The golden set is 40 scenarios, 5 per group (spec §6.15): WSSV, EHP, white faeces, AHPND,
vibriosis, water-quality stress, healthy and ambiguous. EHP-03, HEALTHY-01 and AMBIG-01
already exist from Review 3; this script writes the other 37 as explicit YAML, so the
labels are readable and reviewable in the file itself.

LABELS COME FROM THE RECIPE AND THE SPEC'S LABEL RULES (§3.4, §3.5, §6.15), NOT FROM ANY
AGENT OUTPUT. They are committed before any evaluation run. The team reviews them (TODO).

Idempotent: scenarios already in the file are skipped. Run from pondguard/:
    .venv\\Scripts\\python scripts\\make_golden_specs.py
"""

from __future__ import annotations

import json
from pathlib import Path

SPECS = Path("config/scenario_specs.yaml")
DAY = "2026-10-20"


def ev(kind: str, **kw) -> dict:
    return {"type": kind, **kw}


# Group-level labels (spec §6.15). Water-quality stress is labelled per scenario below.
LABELS = {
    "wssv": dict(risk="critical", top="wssv", ok=["wssv"], action="L3", levels=["L3", "L4"]),
    "ehp": dict(risk="alert", top="ehp", ok=["ehp", "white_faeces"], action="L2", levels=["L2"]),
    "white_faeces": dict(
        risk="alert", top="white_faeces", ok=["white_faeces", "ehp"], action="L2", levels=["L2"]
    ),
    "ahpnd": dict(risk="critical", top="ahpnd", ok=["ahpnd"], action="L3", levels=["L3", "L4"]),
    "vibriosis": dict(
        risk="alert", top="vibriosis", ok=["vibriosis"], action="L2", levels=["L1", "L2"]
    ),
    "healthy": dict(risk="normal", top=None, ok=[], action="L0", levels=["L0"]),
    "ambiguous": dict(risk="watch", top=None, ok=[], action="L0", levels=["L0", "L1"]),
}

# (id, pond, day_of_culture, seed, events, feed %, deaths, symptoms, farmer note)
RECIPES: dict[str, list[tuple]] = {
    "wssv": [
        (
            "WSSV-01",
            "P-21",
            55,
            2101,
            [ev("temp_drop", start="2026-10-19T10:00", hours=6, delta=4.5)],
            40,
            120,
            ["white_spots", "red_discoloration", "gathering_at_edges"],
            "white spots on the shell, shrimp at the pond edge",
        ),
        (
            "WSSV-02",
            "P-22",
            62,
            2201,
            [ev("salinity_drop", start="2026-10-19T08:00", hours=8, delta=7)],
            35,
            260,
            ["white_spots", "lethargy", "sudden_mortality"],
            "heavy rain yesterday, many dead this morning",
        ),
        (
            "WSSV-03",
            "P-23",
            48,
            2301,
            [
                ev("temp_drop", start="2026-10-19T14:00", hours=4, delta=4.0),
                ev("do_crash", start="2026-10-19T00:00", nights=2, depth=0.8),
            ],
            45,
            90,
            ["white_spots", "red_discoloration", "reduced_feeding"],
            "reddish shrimp with white spots",
        ),
        (
            "WSSV-04",
            "P-24",
            70,
            2401,
            [],
            30,
            400,
            ["white_spots", "gathering_at_edges", "lethargy", "sudden_mortality"],
            "mass deaths, white spots, no change in water readings",
        ),
        (
            "WSSV-05",
            "P-25",
            40,
            2501,
            [ev("temp_drop", start="2026-10-19T09:00", hours=5, delta=5.0)],
            50,
            150,
            ["red_discoloration", "gathering_at_edges", "lethargy"],
            "red shrimp swimming at the edges after the cold night",
        ),
    ],
    "ehp": [
        (
            "EHP-01",
            "P-31",
            45,
            3101,
            [],
            78,
            1,
            ["slow_growth", "size_variation"],
            "sizes very uneven at sampling",
        ),
        (
            "EHP-02",
            "P-32",
            60,
            3201,
            [],
            72,
            2,
            ["slow_growth", "size_variation", "reduced_feeding"],
            "growth well behind schedule",
        ),
        (
            "EHP-04",
            "P-34",
            55,
            3401,
            [ev("do_crash", start="2026-10-19T00:00", nights=2, depth=1.0)],
            75,
            2,
            ["slow_growth", "white_faecal_strings"],
            "slow growth and a few white strings",
        ),
        (
            "EHP-05",
            "P-35",
            70,
            3501,
            [],
            80,
            1,
            ["size_variation", "slow_growth", "lethargy"],
            "uneven sizes, a few slow shrimp",
        ),
    ],
    "white_faeces": [
        (
            "WFS-01",
            "P-41",
            38,
            4101,
            [],
            70,
            2,
            ["white_faecal_strings", "reduced_feeding"],
            "white strings on the surface at feeding",
        ),
        (
            "WFS-02",
            "P-42",
            44,
            4201,
            [ev("ammonia_buildup", start="2026-10-18T00:00", days=2, slope=0.3)],
            65,
            3,
            ["white_faecal_strings", "reduced_feeding"],
            "white strings, trays left uneaten",
        ),
        (
            "WFS-03",
            "P-43",
            52,
            4301,
            [ev("do_crash", start="2026-10-19T00:00", nights=2, depth=1.2)],
            72,
            1,
            ["white_faecal_strings"],
            "white strings floating near the aerator",
        ),
        (
            "WFS-04",
            "P-44",
            41,
            4401,
            [],
            68,
            2,
            ["white_faecal_strings", "lethargy"],
            "white faecal strings, some slow shrimp",
        ),
        (
            "WFS-05",
            "P-45",
            58,
            4501,
            [ev("nitrite_buildup", start="2026-10-18T00:00", days=2, slope=0.5)],
            74,
            1,
            ["white_faecal_strings", "reduced_feeding"],
            "white strings, feed intake down",
        ),
    ],
    "ahpnd": [
        (
            "AHPND-01",
            "P-51",
            15,
            5101,
            [],
            40,
            250,
            ["empty_gut", "pale_hepatopancreas", "sudden_mortality"],
            "early crop, many dead, guts empty",
        ),
        (
            "AHPND-02",
            "P-52",
            22,
            5201,
            [],
            35,
            400,
            ["sudden_mortality", "empty_gut"],
            "heavy mortality in the first month",
        ),
        (
            "AHPND-03",
            "P-53",
            18,
            5301,
            [ev("do_crash", start="2026-10-19T00:00", nights=2, depth=0.8)],
            45,
            180,
            ["pale_hepatopancreas", "sudden_mortality", "lethargy"],
            "pale shrunken organs in dead shrimp",
        ),
        (
            "AHPND-04",
            "P-54",
            27,
            5401,
            [],
            30,
            300,
            ["empty_gut", "pale_hepatopancreas", "reduced_feeding"],
            "no feed eaten, empty guts",
        ),
        (
            "AHPND-05",
            "P-55",
            12,
            5501,
            [ev("ammonia_buildup", start="2026-10-18T00:00", days=2, slope=0.25)],
            38,
            220,
            ["sudden_mortality", "empty_gut", "pale_hepatopancreas"],
            "sudden deaths in a young crop",
        ),
    ],
    "vibriosis": [
        (
            "VIBRIO-01",
            "P-61",
            48,
            6101,
            [ev("do_crash", start="2026-10-19T00:00", nights=2, depth=1.0)],
            72,
            6,
            ["red_discoloration", "luminescence", "lethargy"],
            "water glows at night, reddish shrimp",
        ),
        (
            "VIBRIO-02",
            "P-62",
            44,
            6201,
            [],
            70,
            8,
            ["luminescence", "lethargy"],
            "shrimp glow in the dark",
        ),
        (
            "VIBRIO-03",
            "P-63",
            52,
            6301,
            [ev("ammonia_buildup", start="2026-10-17T00:00", days=3, slope=0.25)],
            68,
            10,
            ["red_discoloration", "luminescence"],
            "red shrimp, glowing at night, dirty water",
        ),
        (
            "VIBRIO-04",
            "P-64",
            50,
            6401,
            [ev("do_crash", start="2026-10-18T00:00", nights=3, depth=1.2)],
            74,
            5,
            ["luminescence", "reduced_feeding", "lethargy"],
            "glowing at night, feed down",
        ),
        (
            "VIBRIO-05",
            "P-65",
            46,
            6501,
            [],
            66,
            12,
            ["red_discoloration", "luminescence", "lethargy", "reduced_feeding"],
            "red, glowing, sluggish shrimp",
        ),
    ],
    "healthy": [
        ("HEALTHY-02", "P-81", 25, 8102, [], 99, 0, [], "trays clean, shrimp active"),
        ("HEALTHY-03", "P-82", 48, 8203, [], 96, 0, [], "feeding well, no problems seen"),
        ("HEALTHY-04", "P-83", 62, 8304, [], 100, 0, [], "normal day"),
        ("HEALTHY-05", "P-84", 80, 8405, [], 94, 1, [], "one dead shrimp at the sampling net"),
    ],
    "ambiguous": [
        (
            "AMBIG-02",
            "P-91",
            42,
            9102,
            [],
            82,
            0,
            ["reduced_feeding"],
            "feed trays slightly fuller than usual",
        ),
        (
            "AMBIG-03",
            "P-92",
            55,
            9203,
            [],
            90,
            1,
            ["loose_shell"],
            "a few shrimp with loose shells",
        ),
        (
            "AMBIG-04",
            "P-93",
            47,
            9304,
            [],
            80,
            2,
            ["lethargy", "reduced_feeding"],
            "shrimp slow, eating less",
        ),
        (
            "AMBIG-05",
            "P-94",
            66,
            9405,
            [],
            85,
            1,
            ["loose_shell", "lethargy"],
            "loose shells and some slow shrimp",
        ),
    ],
}

# Water-quality stress: severity decides the label (spec §3.5, §6.15).
WQ = [
    (
        "WQ-01",
        "P-71",
        60,
        7101,
        [ev("ammonia_buildup", start="2026-10-17T00:00", days=3, slope=0.25)],
        85,
        0,
        [],
        "feed a little slower",
        dict(
            risk="watch",
            action="L1",
            levels=["L1"],
            notes="one mild anomaly: ammonia just above its limit",
        ),
    ),
    (
        "WQ-02",
        "P-72",
        58,
        7201,
        [ev("do_crash", start="2026-10-19T00:00", nights=2, depth=1.0)],
        90,
        0,
        [],
        "nothing unusual in the pond",
        dict(
            risk="watch",
            action="L1",
            levels=["L1"],
            notes="one mild anomaly: dawn oxygen dip below its limit",
        ),
    ),
    (
        "WQ-03",
        "P-73",
        64,
        7301,
        [ev("do_crash", start="2026-10-18T00:00", nights=3, depth=1.5)],
        75,
        2,
        ["surfacing_at_dawn", "reduced_feeding"],
        "shrimp come to the surface at dawn",
        dict(
            risk="alert",
            action="L1",
            levels=["L1", "L2"],
            notes="falling dawn oxygen plus surfacing and reduced feeding",
        ),
    ),
    (
        "WQ-04",
        "P-74",
        66,
        7401,
        [
            ev("ammonia_buildup", start="2026-10-17T00:00", days=3, slope=0.5),
            ev("nitrite_buildup", start="2026-10-18T00:00", days=2, slope=0.5),
        ],
        65,
        4,
        ["reduced_feeding"],
        "water smells bad, feeding poor",
        dict(
            risk="critical",
            action="L3",
            levels=["L2", "L3", "L4"],
            notes="ammonia and nitrite far above their limits: a water-quality crisis",
        ),
    ),
    (
        "WQ-05",
        "P-75",
        72,
        7501,
        [ev("do_crash", start="2026-10-18T00:00", nights=3, depth=3.0)],
        60,
        8,
        ["surfacing_at_dawn", "reduced_feeding"],
        "shrimp gasping at the surface every dawn",
        dict(
            risk="critical",
            action="L3",
            levels=["L2", "L3", "L4"],
            notes="dawn oxygen crash below the critical limit: crisis",
        ),
    ),
]


def flow(d: dict) -> str:
    return json.dumps(d, ensure_ascii=False)


def block(
    sid, group, pond, day, seed, events, feed, dead, symptoms, note, label, label_note
) -> str:
    lines = [
        f"  - scenario_id: {sid}",
        f"    group: {group}",
        f"    pond_id: {pond}",
        f'    date: "{DAY}"',
        f"    day_of_culture: {day}",
        f"    seed: {seed}",
    ]
    lines.append("    events: []" if not events else "    events:")
    lines += [f"      - {flow(e)}" for e in events]
    lines += [
        "    farmer_log:",
        f"      feed_consumed_pct: {feed}",
        f"      mortality_count: {dead}",
        f"      symptoms: {json.dumps(symptoms)}",
        f"      notes: {json.dumps(note)}",
        "    history: []",
        "    label:",
        f"      risk_level: {label['risk']}",
        f"      top_condition: {label['top'] if label['top'] else 'null'}",
        f"      acceptable_conditions: {json.dumps(label['ok'])}",
        f"      action_level: {label['action']}",
        f"      acceptable_action_levels: {json.dumps(label['levels'])}",
        f"      should_abstain: {'true' if group == 'ambiguous' else 'false'}",
        f"      notes: {json.dumps(label_note)}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    text = SPECS.read_text(encoding="utf-8")
    existing = {
        line.split(":", 1)[1].strip() for line in text.splitlines() if "scenario_id:" in line
    }
    out = []
    for group, rows in RECIPES.items():
        label = LABELS[group]
        for sid, pond, day, seed, events, feed, dead, symptoms, note in rows:
            if sid not in existing:
                out.append(
                    block(
                        sid,
                        group,
                        pond,
                        day,
                        seed,
                        events,
                        feed,
                        dead,
                        symptoms,
                        note,
                        label,
                        f"{group} recipe: {', '.join(symptoms) or 'no signs'}",
                    )
                )
    for sid, pond, day, seed, events, feed, dead, symptoms, note, lab in WQ:
        if sid not in existing:
            label = dict(
                risk=lab["risk"],
                top="water_quality_stress",
                ok=["water_quality_stress"],
                action=lab["action"],
                levels=lab["levels"],
            )
            out.append(
                block(
                    sid,
                    "water_quality_stress",
                    pond,
                    day,
                    seed,
                    events,
                    feed,
                    dead,
                    symptoms,
                    note,
                    label,
                    lab["notes"],
                )
            )
    if out:
        SPECS.write_text(text.rstrip("\n") + "\n\n" + "\n".join(out), encoding="utf-8")
    print(
        f"added {len(out)} recipes; file now has {text.count('scenario_id:') + len(out)} scenarios"
    )


if __name__ == "__main__":
    main()
