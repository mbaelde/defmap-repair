"""The article's figures, drawn from the journals its tables are read off.

    uv run --group figures python experiments/figures.py scratch/logs/lot3/defmap/*.jsonl
    uv run --group figures python experiments/figures.py --check      # no journal, no corpus

One PDF per figure into `FIG_DIR` (default `figures/`), vector so the paper
includes them without a resolution decision. Every quality figure goes through
`report_defmap.rows`, the same aggregation the table uses, so a figure and a
table can never disagree: paired margins, silent tracks dropped, cells without
an oracle line dropped and named.

Two ceiling lines are derived rather than re-averaged. For any rule of any cell,
`d(mix) - d(m*)` is the paired mean of `sdr_best_real - sdr_mixture` over exactly
the tracks that rule was measured on, and `d(mix) - d(IRM)` the same for the IRM.
So the ceilings are read off the rows of the cell they bound, which is what makes
them comparable to the curve they sit above.

Figure 5 calls `bench_inference_cost.measure` rather than reading a journal:
latency has to be measured on the target machine and there is no corpus in it.
Figure 2 is an illustration of the alignment step and needs neither, but it runs
the deployed `_align`, so what it shows is the code's behaviour and not a drawing
of the idea.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # isort: skip
from matplotlib.axes import Axes  # isort: skip
from matplotlib.figure import Figure  # isort: skip
import numpy as np  # isort: skip

from bench_inference_cost import BENCH_SIZES, BUDGET_MS, measure  # isort: skip
from defmap_phase_ramp import _align, _fit_pairs  # isort: skip
from defmap_protocol import N_FREQ_BINS, NFFT  # isort: skip
from report_ceiling import load, silent_tracks  # isort: skip
from report_defmap import rows  # isort: skip

# The paper is IEEEtran, so its body is Times (Nimbus Roman No9 L in the built PDF)
# and a figure drawn in the matplotlib default sans-serif reads as a foreign object
# dropped into the column. STIXGeneral is second because it ships with matplotlib and
# is Times-metric, which keeps a figure drawn on Linux the same size as one drawn here.
# `pdf.fonttype` is not cosmetic: left at 3, matplotlib emits Type 3 fonts, which the
# IEEE Xplore PDF check rejects outright.
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        # 8 pt is the IEEEtran caption size, and the figures are drawn at the width
        # they are included at, so nothing is rescaled and this is the size on paper
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.frameon": True,
        "legend.edgecolor": "0.8",
        "legend.borderpad": 0.35,
        "legend.handlelength": 1.8,
        "savefig.pad_inches": 0.02,
    }
)

FIG_DIR = Path(os.environ.get("FIG_DIR", "figures"))
# pdf for the paper, png when a figure has to be looked at rather than included
FIG_FORMAT = os.environ.get("FIG_FORMAT", "pdf")
SOURCES = {0: "voice", 1: "accompaniment"}
# one column of IEEEtran is 3.5 in, the text width 7.16: a two-panel figure gets
# the full width and its fonts stay at the paper's size without being rescaled
WIDE = (7.16, 2.9)
NARROW = (3.5, 2.6)
STYLE = {"defmap": "o-", "magnitude": "s--", "oracle": "^-", "ramp": "o-", "ramp-oracle": "^-"}


Row = dict[str, Any]  # a journal row, as `defmap_protocol` writes it
Index = dict[tuple[str, str, int, int], Row]


def index(records: list[Row]) -> Index:
    """The aggregated rows, keyed on (probe, rule, dictionary size, source)."""
    aggregated, orphans = rows(records)
    if orphans:
        dropped = sorted({(r["probe"], r["track"]) for r in orphans})
        print(f"dropped for want of an oracle line: {dropped}", file=sys.stderr)
    for track in sorted(silent_tracks(records)):
        print(f"dropped, silent stem: {track}", file=sys.stderr)
    return {(r["probe"], r["rule"], r["dict_size"], r["source"]): r for r in aggregated}


def sizes(idx: Index, probe: str, rule: str, source: int = 0) -> list[int]:
    """Dictionary sizes a rule was actually measured at, ascending."""
    return sorted({k[2] for k in idx if k[0] == probe and k[1] == rule and k[3] == source})


def curve(idx: Index, probe: str, rule: str, source: int, field: str = "d_mixture") -> tuple[list[int], list[float]]:
    """One rule's trend against dictionary size. Missing cells are absent, never zero."""
    found = [(s, idx[(probe, rule, s, source)][field]) for s in sizes(idx, probe, rule, source)]
    return [s for s, _ in found], [v for _, v in found]


def ceilings(idx: Index, probe: str, rule: str, source: int) -> dict[str, list[float]]:
    """`best_real` and IRM, in gain over the mixture, on the tracks of that rule.

    A difference of two of the rule's own columns: `d(mix) - d(m*)` is the paired
    mean of `sdr_best_real - sdr_mixture`, the rule's own SDR cancelling.
    """
    _, mixture = curve(idx, probe, rule, source, "d_mixture")
    _, star = curve(idx, probe, rule, source, "d_best_real")
    _, irm = curve(idx, probe, rule, source, "d_irm")
    return {
        "best_real": [m - s for m, s in zip(mixture, star, strict=True)],
        "irm": [m - i for m, i in zip(mixture, irm, strict=True)],
    }


def _finish(axes: Axes, xlabel: str, ylabel: str, title: str, legend: bool = True) -> None:
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.grid(True, alpha=0.25, linewidth=0.5)
    if legend:
        axes.legend(framealpha=0.9)


def _log_axis(axes: Axes, values: list[int], base: int = 10) -> None:
    """A log x-axis labelled at the values measured and nowhere else.

    The minor locator has to be silenced explicitly: left alone it writes its own
    `6 x 10^1` on top of the ticks set here, which is how two labels end up
    printed over each other."""
    axes.set_xscale("log", base=base)
    axes.set_xticks(values, [str(v) for v in values])
    axes.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def _save(figure: Figure, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.{FIG_FORMAT}"
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)
    print(f"wrote {path}")
    return path


def fig1_diagnostic(idx: Index) -> Path:
    """Def-MAP's criterion against its own oracle, per dictionary size.

    The central image of the article: the oracle climbs with the dictionary and
    the published criterion falls, so the dictionary is not what is wrong.
    """
    figure, panels = plt.subplots(1, 2, figsize=WIDE, sharey=True)
    for panel, (source, label) in zip(panels, SOURCES.items(), strict=True):
        for rule in ("oracle", "magnitude", "defmap"):
            x, y = curve(idx, "defmap_pair", rule, source)
            if x:
                panel.plot(x, y, STYLE[rule], label=rule, markersize=4, linewidth=1.2)
        bounds = ceilings(idx, "defmap_pair", "defmap", source)
        x = sizes(idx, "defmap_pair", "defmap", source)
        for key, style in (("best_real", ":"), ("irm", "-.")):
            if x:
                panel.plot(x, bounds[key], style, color="0.4", linewidth=1.0,
                           label="$m^\\ast$" if key == "best_real" else "IRM")
        _log_axis(panel, x)
        # one legend for the two panels, on the right one where the curves leave
        # room: the same five entries printed twice is five entries of clutter
        _finish(panel, "atoms per source", "SDR gain over the mixture (dB)" if source == 0 else "",
                label, legend=source == 1)
    return _save(figure, "fig1_diagnostic")


def fig2_ramp(delay: float = 3.4, gain: float = 0.8) -> Path:
    """What the alignment step does to one atom, run through the deployed `_align`.

    An illustration and labelled as one: the atom is drawn rather than learned, so
    that the delay and the gain are known and the figure can show the residual
    that remains when both are removed exactly.

    The planted gain is real on purpose. `_align` maximises the cross-correlation
    of two real frames, so a complex gain rotates that correlation and displaces
    its peak by about its phase over the atom's centre frequency: at `0.8 - 0.3j`
    this construction returns 3.89 samples for a planted 3.4. That coupling is a
    property of the estimator, documented where it lives, and not something a
    figure of the mechanism should be asked to carry.
    """
    rng = np.random.default_rng(0)
    bins = np.arange(N_FREQ_BINS)
    envelope = np.exp(-bins / 120.0) + 0.05
    atom = np.asarray(envelope * np.exp(2j * np.pi * rng.random(N_FREQ_BINS)), dtype=np.complex128)
    other = np.asarray(0.35 * envelope * np.exp(2j * np.pi * rng.random(N_FREQ_BINS)), dtype=np.complex128)
    mixture = np.asarray(gain * atom * np.exp(-2j * np.pi * delay * bins / NFFT) + other, dtype=np.complex128)

    aligned, found = _align(atom[None, :], mixture)
    fitted, _, _ = _fit_pairs(aligned, other[None, :], mixture)
    residual = mixture - fitted[0, 0] * aligned[0]

    figure, panels = plt.subplots(1, 3, figsize=(7.16, 2.4))
    panels[0].plot(bins, np.unwrap(np.angle(mixture) - np.angle(atom)), linewidth=0.8, color="C0")
    panels[0].plot(bins, -2 * np.pi * found[0] * bins / NFFT + np.angle(fitted[0, 0]), "--",
                   linewidth=1.0, color="C3", label=f"fitted ramp, $\\tau$ = {found[0]:.2f}")
    _finish(panels[0], "frequency bin", "phase difference (rad)", "before alignment")

    panels[1].plot(bins, np.unwrap(np.angle(mixture) - np.angle(aligned[0])), linewidth=0.8,
                   color="C0", label="residual phase")
    panels[1].axhline(np.angle(fitted[0, 0]), linestyle="--", linewidth=1.0, color="C3",
                      label="gain phase")
    _finish(panels[1], "frequency bin", "phase difference (rad)", "after alignment")

    panels[2].semilogy(bins, np.abs(mixture), linewidth=0.8, label="mixture")
    panels[2].semilogy(bins, np.abs(fitted[0, 0] * aligned[0]), linewidth=0.8, label="aligned atom")
    panels[2].semilogy(bins, np.abs(residual), linewidth=0.8, color="0.4", label="residual")
    _finish(panels[2], "frequency bin", "magnitude", "what is left")
    return _save(figure, "fig2_ramp")


def fig3_constraint(idx: Index) -> Path:
    """The oracle-to-criterion gap, and the slope against the dictionary, in both
    regimes. Constraining the deformation to a delay and a gain closes the gap and
    turns the slope positive, which is the repair of section 5 in one figure."""
    figure, (left, right) = plt.subplots(1, 2, figsize=WIDE)
    regimes = (("defmap_pair", "oracle", "defmap", "free"), ("defmap_ramp", "ramp-oracle", "ramp", "constrained"))
    ticks: set[int] = set()
    for source, label in SOURCES.items():
        for probe, oracle, rule, regime in regimes:
            x, top = curve(idx, probe, oracle, source)
            _, bottom = curve(idx, probe, rule, source)
            if not x:
                continue
            ticks |= set(x)
            # the two free curves of the left panel land on each other, the gap
            # being ~6 dB on both sources: dash patterns per source and hollow
            # markers keep the one drawn first visible under the other
            kwargs = {"color": f"C{source}", "markersize": 4, "linewidth": 1.2,
                      "label": f"{label}, {regime}"}
            if regime == "constrained":
                kwargs |= {"marker": "o", "linestyle": "-"}
            else:
                kwargs |= {"marker": "s", "markerfacecolor": "none",
                           "dashes": (5, 2) if source == 0 else (1.5, 2)}
            left.plot(x, [t - b for t, b in zip(top, bottom, strict=True)], **kwargs)
            right.plot(x, bottom, **kwargs)
    for panel in (left, right):
        _log_axis(panel, sorted(ticks))
    _finish(left, "atoms per source", "oracle $-$ criterion (dB)", "selection gap")
    _finish(right, "atoms per source", "SDR gain over the mixture (dB)", "what the criterion delivers",
            legend=False)
    return _save(figure, "fig3_constraint")


def _families(idx: Index, probe: str, source: int) -> dict[str, dict[int, float]]:
    """Every `<name> k=N` rule of a probe, grouped by name, as {k: gain}.

    Written as a scan of the journal's rule names rather than a list of the rules
    expected, so a selection rule added to experiment 2.5 appears in the figure
    without this file being touched.
    """
    out: dict[str, dict[int, float]] = {}
    for (found, rule, _size, src), row in idx.items():
        match = re.fullmatch(r"(.+) k=(\d+)", rule)
        if found != probe or src != source or match is None:
            continue
        out.setdefault(match.group(1), {})[int(match.group(2))] = row["d_mixture"]
    return out


def fig4_selection(idx: Index) -> Path:
    """Capacity, delivered quality, and the selection gap between them, against `k`.

    The most telling figure of the article: the model class gains ~4.5 dB of
    capacity between `k = 1` and `k = 16` and delivers none of it. The hatch is
    what experiment 2.5's rules are asked to close, and they are drawn in it.
    """
    figure, panels = plt.subplots(1, 2, figsize=WIDE, sharey=True)
    for panel, (source, label) in zip(panels, SOURCES.items(), strict=True):
        families = _families(idx, "defmap_local", source)
        capacity, quality = families.pop("capacity", {}), families.get("local", {})
        shared = sorted(set(capacity) & set(quality))
        if shared:
            panel.fill_between(shared, [quality[k] for k in shared], [capacity[k] for k in shared],
                               hatch="///", facecolor="none", edgecolor="0.6", linewidth=0.0,
                               label="absorbed by selection")
            panel.plot(shared, [capacity[k] for k in shared], "^-", markersize=4, linewidth=1.2,
                       label="capacity")
        for name, series in sorted(families.items()):
            style = "o-" if name == "local" else "s--"
            ks = sorted(series)
            panel.plot(ks, [series[k] for k in ks], style, markersize=4, linewidth=1.2, label=name)
        free = curve(idx, "defmap_pair", "oracle", source)[1]
        if free:
            panel.axhline(free[-1], linestyle=":", color="0.4", linewidth=1.0,
                          label="free deformation oracle")
        if shared:
            _log_axis(panel, shared, base=2)
        _finish(panel, "atoms per source in the combination, $k$",
                "SDR gain over the mixture (dB)" if source == 0 else "", label, legend=source == 1)
    return _save(figure, "fig4_selection")


def fig5_cost(sizes_: tuple[int, ...] = BENCH_SIZES) -> Path:
    """Per-frame cost against dictionary size, against the real-time budget.

    Measured here and now, on the machine drawing the figure, which is the only
    honest way to publish a latency: the numbers of a table computed elsewhere
    would not be reproducible by the reader on their own machine either.
    """
    rng = np.random.default_rng(0)
    measured = {size: measure(size, rng) for size in sizes_}
    figure, panel = plt.subplots(figsize=NARROW)
    rules = ["defmap", "ramp", "local k=16", "greedy k=16", "nmf i=300", "nmf i=25"]
    for rule in [r for r in rules if r in measured[sizes_[0]]]:
        panel.loglog(sizes_, [measured[s][rule] for s in sizes_], "o-", markersize=3.5,
                     linewidth=1.2, label=rule)
    panel.axhline(BUDGET_MS, linestyle="--", color="0.3", linewidth=1.0,
                  label=f"{BUDGET_MS:.0f} ms budget")
    panel.set_yscale("log")
    # a decade of headroom, or the seven-entry legend lands on Def-MAP's curve:
    # every rule is below it and the cleared band is the only free space left
    top = max(measured[sizes_[-1]][r] for r in rules if r in measured[sizes_[-1]])
    panel.set_ylim(top=top * 30)
    _log_axis(panel, list(sizes_))
    _finish(panel, "atoms per source", "milliseconds per frame", "", legend=False)
    panel.legend(framealpha=0.9, loc="upper left", ncol=2)
    return _save(figure, "fig5_cost")


def demo() -> None:
    """The two computations that are not plotting: the derived ceilings and the
    rule-family scan. Then every figure drawn once, on rows shaped like the
    journals', since a figure that raises is a figure nobody notices is missing."""
    knobs = {
        "corpus": "musdb", "regime": "unseen", "n_train": 25, "n_test": 2,
        "test_seconds": 5, "max_train_frames": 20000, "seed": 0,
    }
    witness = {"phase_median_deg": 0.0, "gain_above_one": 0.0, "seconds": 60.0, "si_sdr": 0.0}
    records: list[Row] = []
    for probe, rules in (
        ("defmap_pair", ["defmap", "magnitude", "oracle"]),
        ("defmap_ramp", ["ramp", "ramp-oracle"]),
        ("defmap_local", ["local k=1", "local k=4", "capacity k=1", "capacity k=4", "greedy k=4"]),
    ):
        for track, offset in (("easy", 4.0), ("hard", -4.0)):
            for source in (0, 1):
                records.append({"probe": probe, "track": track, "source": source,
                                "method": "oracle", "sdr_mixture": offset - 3.0,
                                "sdr_irm": offset + 8.0, "sdr_best_real": offset + 14.0, **knobs})
                for n, rule in enumerate(rules):
                    for size in (50, 300):
                        records.append({"probe": probe, "track": track, "source": source,
                                        "method": probe, "rule": rule, "dict_size": size,
                                        "sdr": offset + n + size / 300.0, **witness, **knobs})
    idx = index(records)

    # the ceiling is the mixture-to-ceiling distance of the tracks of that rule,
    # here 14 - (-3) = 17 dB whatever the rule and the size
    bounds = ceilings(idx, "defmap_pair", "defmap", 0)
    assert np.allclose(bounds["best_real"], 17.0), bounds
    assert np.allclose(bounds["irm"], 11.0), bounds
    assert idx[("defmap_pair", "defmap", 300, 0)]["tracks"] == 2

    families = _families(idx, "defmap_local", 0)
    assert set(families) == {"local", "capacity", "greedy"}, families
    assert sorted(families["local"]) == [1, 4], families

    # figure 2's alignment: the delay injected must come back out of `_align`,
    # or the figure draws a fitted ramp that is not the one the separator uses
    rng = np.random.default_rng(0)
    bins = np.arange(N_FREQ_BINS)
    atom = np.exp(2j * np.pi * rng.random(N_FREQ_BINS))
    shifted = atom * np.exp(-2j * np.pi * 3.4 * bins / NFFT)
    _, found = _align(atom[None, :], shifted)
    # a tenth of a sample of slack: the refinement fits a parabola to a peak that
    # is not one, and that bias is a property of the deployed rule rather than of
    # the figure. What this catches is a sign error or a bin/sample mix-up.
    assert abs(found[0] - 3.4) < 0.15, found

    for path in (fig1_diagnostic(idx), fig2_ramp(), fig3_constraint(idx), fig4_selection(idx),
                 fig5_cost((50,))):
        assert path.stat().st_size > 1000, path
    print("demo ok: ceilings derived, families scanned, delay recovered, five figures drawn")


def main() -> None:
    paths = [Path(a) for a in sys.argv[1:] if not a.startswith("-")]
    idx = index(load(paths))
    fig1_diagnostic(idx)
    fig2_ramp()
    fig3_constraint(idx)
    fig4_selection(idx)
    fig5_cost()


if __name__ == "__main__":
    demo() if "--check" in sys.argv else main()
