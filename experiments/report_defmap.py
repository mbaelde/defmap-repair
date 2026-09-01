"""Turn the Def-MAP journals into the table the article 2 claims are read off.

    uv run python experiments/report_defmap.py scratch/logs/defmap/*.jsonl
    uv run python experiments/report_defmap.py --check

One row per (probe, method, rule, dictionary size, source). Two columns decide,
both paired track by track because MUSDB tracks differ in difficulty by far more
than the margins being measured:

    d(mix)   mean of sdr - sdr_mixture, the gain over doing nothing. This is what
             the article's "+12.9 dB" style numbers are, and the only column that
             can be compared to a published separation figure.
    d(m*)    mean of sdr - sdr_best_real, the distance to the ceiling of the
             masking class. `d(IRM)` is the same distance to the IRM, kept because
             that is what the literature usually calls the ceiling: the two differ
             by ~6 dB, and every article 2 claim phrased against IRM2 understates
             its distance to what a mask can do by that much.

The oracle-to-criterion gap needs no column of its own: two rules of the same
cell are averaged over the same tracks, so subtracting their `d(mix)` (or their
`d(m*)`) *is* the paired mean of the per-track gap. Read `oracle` minus `defmap`,
`ramp-oracle` minus `ramp`, `capacity k=n` minus `local k=n`.

Oracles come from the `method: oracle` records of the same journal, matched on
(track, source). A method record whose oracle is missing is dropped rather than
compared against another track, and every drop is printed.
"""

from __future__ import annotations

import statistics
import sys
from collections import defaultdict
from pathlib import Path

from report_ceiling import load, render_dropped, silent_tracks

# the corpus knobs, printed once per probe so a row is readable without its filename
KNOBS = ("corpus", "regime", "n_train", "n_test", "test_seconds", "max_train_frames", "seed")


def rows(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """The aggregated rows, and the method records dropped for want of an oracle."""
    silent = silent_tracks(records)
    kept = [r for r in records if (r["probe"], r["track"]) not in silent]
    oracle = {
        (r["probe"], r["track"], r["source"]): r for r in kept if r["method"] == "oracle"
    }

    groups: dict[tuple, list[tuple[dict, dict]]] = defaultdict(list)
    orphans = []
    for r in kept:
        if r["method"] == "oracle":
            continue
        ref = oracle.get((r["probe"], r["track"], r["source"]))
        if ref is None:
            orphans.append(r)
            continue
        groups[(r["probe"], r["method"], r["dict_size"], r["rule"], r["source"])].append((r, ref))

    out = []
    for (probe, method, size, rule, source), pairs in sorted(groups.items(), key=_order):
        fit, refs = [p[0] for p in pairs], [p[1] for p in pairs]
        mean = statistics.fmean
        out.append(
            {
                "probe": probe,
                "knobs": " ".join(f"{key}={refs[0][key]}" for key in KNOBS),
                "method": method,
                "rule": rule,
                "dict_size": size,
                "source": source,
                "tracks": len(pairs),
                "sdr": mean([r["sdr"] for r in fit]),
                "si_sdr": mean([r["si_sdr"] for r in fit]),
                "d_mixture": mean([r["sdr"] - o["sdr_mixture"] for r, o in pairs]),
                "d_best_real": mean([r["sdr"] - o["sdr_best_real"] for r, o in pairs]),
                "d_irm": mean([r["sdr"] - o["sdr_irm"] for r, o in pairs]),
                "phase_deg": mean([r["phase_median_deg"] for r in fit]),
                "gain_above_one": mean([r["gain_above_one"] for r in fit]),
                "minutes": mean([r["seconds"] for r in fit]) / 60,
            }
        )
    return out, orphans


def _order(item: tuple[tuple, list]) -> tuple:
    """Sort key: dictionary size before rule, so a rule's trend reads down a column."""
    probe, method, size, rule, source = item[0]
    return (probe, method, source, size, rule)


def render(rows_: list[dict]) -> str:
    width = max([len(r["rule"]) for r in rows_] + [4])
    header = (
        f"{'probe':<14} {'method':<6} {'rule':<{width}} {'dict':>5} {'src':>3} {'n':>2} "
        f"{'SDR':>7} {'SI-SDR':>7} {'d(mix)':>7} {'d(m*)':>7} {'d(IRM)':>7} "
        f"{'phase':>6} {'g>1':>5} {'min':>5}"
    )
    lines = [header, "-" * len(header)]
    knobs = {}
    for r in rows_:
        knobs[r["probe"]] = r["knobs"]
        lines.append(
            f"{r['probe']:<14} {r['method']:<6} {r['rule']:<{width}} "
            f"{r['dict_size']:>5} {r['source']:>3} {r['tracks']:>2} "
            f"{r['sdr']:>7.2f} {r['si_sdr']:>7.2f} {r['d_mixture']:>+7.2f} "
            f"{r['d_best_real']:>+7.2f} {r['d_irm']:>+7.2f} "
            f"{r['phase_deg']:>5.1f}d {r['gain_above_one']:>5.2f} {r['minutes']:>5.1f}"
        )
    lines += ["", "probes:"] + [f"  {p:<14} {k}" for p, k in sorted(knobs.items())]
    return "\n".join(lines)


def render_orphans(orphans: list[dict]) -> str:
    """An orphan is a cell whose oracle line never made it: say so, or the table
    silently reports fewer tracks than the journal contains."""
    if not orphans:
        return "no record dropped for a missing oracle"
    listed = sorted({(r["probe"], r["track"], r["method"]) for r in orphans})
    lines = "\n".join(f"  {p:<14} {t:<28} {m}" for p, t, m in listed)
    return f"dropped, no oracle line for that (track, source):\n{lines}"


def demo() -> None:
    """The pairing and the silence guard are the whole point of this file."""
    knobs = {
        "corpus": "musdb", "regime": "vocals", "n_train": 25, "n_test": 10,
        "test_seconds": 5, "max_train_frames": 20000, "seed": 0,
    }
    witness = {"phase_median_deg": 0.0, "gain_above_one": 0.0, "seconds": 60.0, "si_sdr": 0.0}
    records = [
        # easy and hard track deliberately far apart: an unpaired mean would mix
        # their difficulty into the margin and hide it
        {"probe": "p", "track": "easy", "source": 0, "method": "oracle",
         "sdr_best_real": 20.0, "sdr_irm": 14.0, "sdr_mixture": 3.0, **knobs},
        {"probe": "p", "track": "hard", "source": 0, "method": "oracle",
         "sdr_best_real": 8.0, "sdr_irm": 2.0, "sdr_mixture": -7.0, **knobs},
        # one stem silent: infinite ceiling, drops both sources of the track
        {"probe": "p", "track": "mute", "source": 0, "method": "oracle",
         "sdr_best_real": float("inf"), "sdr_irm": float("inf"),
         "sdr_mixture": float("-inf"), **knobs},
        {"probe": "p", "track": "easy", "source": 0, "method": "ramp", "rule": "ramp",
         "dict_size": 50, "sdr": 12.0, **witness, **knobs},
        {"probe": "p", "track": "hard", "source": 0, "method": "ramp", "rule": "ramp",
         "dict_size": 50, "sdr": 2.0, **witness, **knobs},
        {"probe": "p", "track": "easy", "source": 0, "method": "ramp",
         "rule": "ramp-oracle", "dict_size": 50, "sdr": 14.0, **witness, **knobs},
        {"probe": "p", "track": "hard", "source": 0, "method": "ramp",
         "rule": "ramp-oracle", "dict_size": 50, "sdr": 5.0, **witness, **knobs},
        {"probe": "p", "track": "mute", "source": 0, "method": "ramp", "rule": "ramp",
         "dict_size": 50, "sdr": float("-inf"), **witness, **knobs},
        # no oracle line for this track: dropped, and said out loud
        {"probe": "p", "track": "orphan", "source": 0, "method": "ramp", "rule": "ramp",
         "dict_size": 50, "sdr": 99.0, **witness, **knobs},
    ]
    out, orphans = rows(records)
    assert silent_tracks(records) == {("p", "mute")}, silent_tracks(records)
    assert [r["rule"] for r in out] == ["ramp", "ramp-oracle"], out
    assert [r["track"] for r in orphans] == ["orphan"], orphans

    ramp, oracle = out
    assert ramp["tracks"] == 2, ramp["tracks"]
    assert abs(ramp["sdr"] - 7.0) < 1e-9, ramp["sdr"]
    assert abs(ramp["d_mixture"] - 9.0) < 1e-9, ramp["d_mixture"]  # (12-3 + 2+7)/2
    assert abs(ramp["d_best_real"] + 7.0) < 1e-9, ramp["d_best_real"]  # (12-20 + 2-8)/2
    assert abs(ramp["d_irm"] + 1.0) < 1e-9, ramp["d_irm"]  # (12-14 + 2-2)/2
    assert abs(ramp["minutes"] - 1.0) < 1e-9, ramp["minutes"]
    # the diagnostic gap is a difference of two columns, and it must equal the
    # paired mean of the per-track gaps: (14-12 + 5-2)/2 = 2.5
    assert abs((oracle["d_mixture"] - ramp["d_mixture"]) - 2.5) < 1e-9
    assert abs((oracle["d_best_real"] - ramp["d_best_real"]) - 2.5) < 1e-9
    assert "test_seconds=5" in ramp["knobs"], ramp["knobs"]
    print("demo ok: paired margins, silence guard, orphans reported, gap = column difference")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] == "--check":
        demo()
    else:
        records = load([Path(a) for a in args])
        aggregated, orphans = rows(records)
        print(render(aggregated))
        print()
        print(render_dropped(silent_tracks(records)))
        print(render_orphans(orphans))
