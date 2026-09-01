"""What one frame of inference costs, per parameterization, against the 23 ms budget.

The separation quality numbers come from `defmap_oracle_pair.py` and
`defmap_phase_ramp.py`; this measures the other axis of the thesis' table, the
per-frame latency, and it measures it *separately from the diagnostic rules*
those two scripts also compute. That distinction matters: the 132 ms/frame
printed by the phase-ramp sweep includes an oracle and a capacity bound that a
deployed separator never evaluates.

Cost depends only on the shapes (K1, K2, B), never on the audio, so this runs on
random spectra and needs no corpus. Which is the point: latency has to be
measured on the target machine, and the machine that holds the dataset is not
necessarily it.

Rows per dictionary size:

  defmap    Def-MAP's per-bin complex solve over every pair. Extrapolated from a
            sample of pairs, since 300 atoms/source means 90k Python-level calls
            per frame and timing them all is itself minutes of work.
  ramp      the repaired inference: align every atom on the mixture, solve the
            2x2 gains of every pair, take the argmin, rebuild one pair.
  local k=N the k-atom local combination: the same alignment, no pair search.
  greedy k=N  joint selection by deflation, which refits at every one of its 2k
            steps and is where experiment 2.5 spends its latency.
  nmf       the supervised baseline of `nmf_baseline.py`, activations of a fixed
            2*size-component basis on one magnitude frame. Its quality is only
            comparable to the rules above if its cost is, so it is timed here on
            the same frame and against the same budget.
  ramp/parts  where the ramp's time actually goes, which decides whether the
            remaining factor to the budget is a micro-optimization or needs the
            approximate search of the paper's section 2.4.

    uv run python experiments/bench_inference_cost.py
    BENCH_SIZES=50,100,300,1000 uv run python experiments/bench_inference_cost.py
"""

from __future__ import annotations

import os
import time

import numpy as np

from defmap_local_combination import _combine, _greedy
from defmap_phase_ramp import _align, _fit_pairs
from gasm.rase.defmap import solve_complex_deformation
from nmf_baseline import FAST_ITER, MAX_ITER, _activate

from defmap_oracle_pair import N_FREQ_BINS, SEED  # isort: skip

BENCH_SIZES = tuple(int(k) for k in os.environ.get("BENCH_SIZES", "50,100,300,1000").split(","))
BENCH_K = tuple(int(k) for k in os.environ.get("BENCH_K", "4,16").split(","))
BUDGET_MS = 23.0  # thesis' real-time constraint: one 1024-sample frame at 44.1 kHz
DEFMAP_SAMPLE = 200  # pairs actually timed before extrapolating to K1*K2
REPEATS = 5


def _best(fn, repeats: int = REPEATS) -> float:
    """Milliseconds of the fastest run. Best-of, not mean: a slower run measured
    the machine's other tenants, not the code."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000.0)
    return min(times)


def _ramp_frame(dict1, dict2, mix) -> None:
    """One frame of the deployed rule, and nothing else."""
    x, _ = _align(dict1, mix)
    y, _ = _align(dict2, mix)
    gain_x, gain_y, residual = _fit_pairs(x, y, mix)
    i, j = np.unravel_index(int(np.argmin(residual)), residual.shape)
    _ = gain_x[i, j] * x[i], gain_y[i, j] * y[j]


def _local_frame(dict1, dict2, mix, k: int) -> None:
    """One frame of the k-atom local combination: same alignment, no pair search."""
    x, _ = _align(dict1, mix)
    y, _ = _align(dict2, mix)
    _ = _combine(x, y, mix, k)


def _greedy_frame(dict1, dict2, mix, k: int) -> None:
    """One frame of experiment 2.5's joint selection, alignment included."""
    x, _ = _align(dict1, mix)
    y, _ = _align(dict2, mix)
    _ = _greedy(x, y, mix, k)


def _nmf_frame(basis, mix, iters: int) -> None:
    """One frame of the supervised baseline: activations of a fixed basis.

    One frame and not the whole spectrogram, even though the sweep solves it in
    one call: with the basis fixed the objective separates over frames, so the
    batched call is an implementation detail and this is the deployed cost.
    """
    _ = _activate(np.abs(mix)[None, :], basis, iters)


def measure(size: int, rng: np.random.Generator) -> dict[str, float]:
    """Milliseconds per frame of every timed rule, at one dictionary size.

    Returned rather than printed so that figure 5 plots the same numbers the
    table quotes instead of a second measurement of its own, taken on whatever
    machine happened to draw the figure.
    """
    shape = (size, N_FREQ_BINS)
    dict1 = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    dict2 = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    mix = rng.normal(size=N_FREQ_BINS) + 1j * rng.normal(size=N_FREQ_BINS)

    sample = _best(
        lambda: [
            solve_complex_deformation(dict1[k % size], dict2[k // size % size], mix)
            for k in range(DEFMAP_SAMPLE)
        ]
    )
    x, _ = _align(dict1, mix)
    y, _ = _align(dict2, mix)
    basis = np.abs(np.concatenate((dict1, dict2)))

    out = {
        "defmap": sample * size * size / DEFMAP_SAMPLE,
        "ramp": _best(lambda: _ramp_frame(dict1, dict2, mix)),
        "align": _best(lambda: (_align(dict1, mix), _align(dict2, mix))),
        "fit": _best(lambda: _fit_pairs(x, y, mix)),
    }
    for k in BENCH_K:
        out[f"local k={k}"] = _best(lambda: _local_frame(dict1, dict2, mix, k))
    for k in BENCH_K:
        out[f"greedy k={k}"] = _best(lambda: _greedy_frame(dict1, dict2, mix, k))
    for iters in (MAX_ITER, FAST_ITER):
        out[f"nmf i={iters}"] = _best(lambda: _nmf_frame(basis, mix, iters))
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    print(f"budget {BUDGET_MS:.0f} ms/frame, {N_FREQ_BINS} bins, best of {REPEATS}\n")

    for size in BENCH_SIZES:
        ms = measure(size, rng)
        print(f"{size} atoms/source ({size * size} pairs):")
        print(f"    defmap  {ms['defmap']:9.1f} ms  ({ms['defmap'] / BUDGET_MS:6.1f}x budget, extrapolated)")
        for rule in ["ramp"] + [f"local k={k}" for k in BENCH_K] + [f"greedy k={k}" for k in BENCH_K] \
                + [f"nmf i={iters}" for iters in (MAX_ITER, FAST_ITER)]:
            print(f"    {rule:<11s}{ms[rule]:7.1f} ms  ({ms[rule] / BUDGET_MS:6.1f}x budget)")
        print(
            f"    parts   align {ms['align']:.1f} ms + fit {ms['fit']:.1f} ms"
            f"   [speedup vs defmap: {ms['defmap'] / ms['ramp']:.0f}x]\n"
        )


if __name__ == "__main__":
    main()
