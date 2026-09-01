"""Def-MAP repair 2.1: constrain the deformation's phase to a delay and a global offset.

`defmap_oracle_pair.py` established that the dictionary carries the information
(oracle at +12.9 dB over the mixture, 83% of the IRM2 ceiling) and that Def-MAP's
own criterion recovers barely a third of it, degrading as the dictionary grows.
Those two figures predate `defmap_protocol` and are quoted against IRM2, which
sits ~6 dB *below* the masking class' real ceiling: the fraction-of-ceiling half
is to be reread off a rerun, the gap half is a difference and survives.
The suspected cause is the deformation itself: Def-MAP fits an independent
complex factor per bin, i.e. 2B = 1026 free real parameters per candidate, so any
candidate explains any mixture and the fit residual stops ranking pairs.

The proposed repair makes the invariance explicit instead of leaving it to the
deformation. A sub-sample misalignment between a training frame and a test frame
rotates the short-term phase proportionally to frequency, without changing what
is heard, so the deformation only needs an affine phase:

    c(f) -> g * c(f) * exp(-2j*pi*f*tau)      g complex, tau a delay in samples

3 real parameters per candidate (gain, global phase, delay) instead of 1026. The
question this answers is in two halves, and they are measured separately:

  capacity   can a *single* atom per source, fitted this way to the true source,
             still reach the dictionary's reachable quality? If restricting the
             phase costs most of the +12.9 dB oracle, the repair is too tight and
             2.1 fails regardless of how well selection then works.
  selection  does the fit residual against the mixture now rank pairs properly?
             That is the whole point of removing degrees of freedom: a model that
             cannot explain everything tells the truth about what it explains.
             `ramp` selects on that residual, `ramp-oracle` selects on the true
             error with the same estimator, and the gap between them is the same
             diagnostic quantity as in 2.0.

The delay comes from the peak of the cross-correlation between candidate and
target (one batched inverse FFT per frame and source), refined to sub-sample by
parabolic interpolation, then the complex gains of a pair solve a 2x2 Hermitian
system in closed form. Cost per frame is therefore K FFTs plus one K1 x K2
matmul, against Def-MAP's K1 x K2 x B per-bin solve: the complexity reduction of
lot 2.4 falls out of the reparameterization rather than being bolted on.

Capacity did come out short (see the `ramp+r*` rows), so the residual term of the
plan is here as well, in the cheapest form that spans both parameterizations: the
selected pair absorbs a fraction alpha of what the ramp left unexplained, split
between the sources by their energy ratio, alpha=0 being the pure ramp and
alpha=1 Def-MAP's own structure. Selection stays on the alpha=0 residual, so the
sweep isolates what the extra freedom is worth once the ranking is already sound
-- which is exactly the part Def-MAP could never measure, its own criterion being
computed under the freedom it was supposed to judge.

ponytail: alpha is a global constant, not per-frame or per-bin. A per-bin
shrinkage learned on training data is the obvious upgrade, and only worth it if
the alpha curve turns out to have an interior optimum.

Corpus, metric, oracles and journal are `defmap_protocol`'s, so a row here is
comparable to a row of 2.0 and to article 1's. Usage, in a container that has
ffmpeg (musdb imports stempeg):

    MUSDB_ROOT=/path/to/musdb18-7s python experiments/defmap_phase_ramp.py
    DICT_SIZES=50 TEST_SECONDS=5 N_TEST=2 MUSDB_ROOT=... python experiments/...
    sh experiments/run.sh defmap        # the resumable form, one cell per log
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

# run as a script from python/, so experiments/ is sys.path[0] and this resolves
from defmap_oracle_pair import DICT_SIZES
from defmap_protocol import (
    NFFT,
    N_FREQ_BINS,
    SEED,
    ComplexArray,
    FloatArray,
    config,
    corpus,
    dictionary,
    done_cells,
    emit,
    note,
    oracle_rows,
    score,
    test_spectra,
)

# The hypothesis is about sub-frame misalignment. A lag beyond a few ms would align
# the atom onto a different acoustic event, and the phase ramp on a windowed frame
# is a circular shift, which stops approximating a delay once the shift is large.
MAX_LAG = int(os.environ.get("MAX_LAG", "128"))  # samples, 2.9 ms at 44.1 kHz
RIDGE = 1e-8  # relative, keeps the 2x2 solve finite for near-colinear candidates
PAIR_RULES = ("ramp", "ramp-oracle")
# Dial between the two parameterizations, applied *after* selection: the chosen pair
# absorbs a fraction alpha of what the ramp could not explain, split between the two
# sources by their energy ratio. alpha=0 is the pure ramp, alpha=1 puts the whole
# residual back and reproduces Def-MAP's structure (estimates summing to the mixture,
# hence err1 = -err2). Selection always uses the alpha=0 residual, which is the point:
# the model that ranks pairs stays rigid even when the model that reconstructs is not.
ALPHAS = tuple(float(a) for a in os.environ.get("ALPHAS", "0,0.25,0.5,0.75,1").split(","))
METHOD = "ramp"  # the `method` field of every row this script writes


def _align(atoms: ComplexArray, target: ComplexArray) -> tuple[ComplexArray, FloatArray]:
    """Each atom phase-shifted onto `target`, and the delays used, in samples.

    argmax over lag of irfft(conj(atom) * target) is the delay that best aligns
    the atom with the target, parabolically refined to sub-sample precision.

    On a Hanning-windowed frame the ramp is a *circular* shift, which only
    approximates a delay, and the approximation is why lags are capped: within a
    few samples the wrapped tail sits under the window's near-zero edges.

    The delay estimate is coupled to the gain that follows it. `irfft` over a half
    spectrum is the cross-correlation of two *real* frames, so its peak is the
    delay only when the gain relating them is real; a complex gain rotates the
    correlation and moves the peak by about its phase over the atom's centre
    frequency, half a sample at 0.3 rad on a low-passed atom. The gains are fitted
    after the delay and are genuinely complex, so this bias is present in every
    number this script reports. Decoupling it means refining on the correlation's
    analytic envelope instead of its real part, which is a change of rule and not
    of implementation, so it is measured as one or not at all.
    """
    correlation = np.fft.irfft(np.conj(atoms) * target, n=NFFT, axis=-1)
    lags = np.concatenate((np.arange(MAX_LAG + 1), np.arange(-MAX_LAG, 0)))
    window = correlation[:, lags]
    peak = np.argmax(window, axis=-1)

    left = window[np.arange(len(atoms)), (peak - 1) % len(lags)]
    right = window[np.arange(len(atoms)), (peak + 1) % len(lags)]
    centre = window[np.arange(len(atoms)), peak]
    curvature = left - 2.0 * centre + right
    offset = np.where(curvature < 0, 0.5 * (left - right) / np.where(curvature < 0, curvature, 1.0), 0.0)
    delay = lags[peak] + np.clip(offset, -0.5, 0.5)

    bins = np.arange(N_FREQ_BINS)
    return atoms * np.exp(-2j * np.pi * np.outer(delay, bins) / NFFT), delay


def _fit_pairs(
    x: ComplexArray, y: ComplexArray, mix: ComplexArray
) -> tuple[ComplexArray, ComplexArray, FloatArray]:
    """Complex gains of every (x, y) pair least-squares fitted to `mix`, and the residual.

    Normal equations of min_g || g1 x + g2 y - m ||^2, solved in closed form:
    with xx = ||x||^2, yy = ||y||^2, xy = x^H y, bx = x^H m, by = y^H m,

        det = xx*yy - |xy|^2
        g1 = (yy*bx - xy*by) / det        g2 = (xx*by - conj(xy)*bx) / det
        residual = ||m||^2 - Re(conj(g1)*bx + conj(g2)*by)
    """
    xx = np.sum(np.abs(x) ** 2, axis=-1)[:, None]
    yy = np.sum(np.abs(y) ** 2, axis=-1)[None, :]
    xy = np.conj(x) @ y.T
    bx = (np.conj(x) @ mix)[:, None]
    by = (np.conj(y) @ mix)[None, :]

    # ridge on the diagonal, so two near-colinear atoms give a finite (and small) pair
    # of gains instead of a blow-up that would win the argmin on numerical noise
    a11 = xx + RIDGE * (xx + yy)
    a22 = yy + RIDGE * (xx + yy)
    det = a11 * a22 - np.abs(xy) ** 2 + np.finfo(float).tiny
    gain_x = (a22 * bx - xy * by) / det
    gain_y = (a11 * by - np.conj(xy) * bx) / det
    residual = np.sum(np.abs(mix) ** 2) - np.real(np.conj(gain_x) * bx + np.conj(gain_y) * by)
    return gain_x, gain_y, residual


def _pair_error(
    gain: ComplexArray, atoms: ComplexArray, truth: ComplexArray, axis: int
) -> FloatArray:
    """|| gain * atom - truth ||^2, expanded so no (K1, K2, B) array is ever formed."""
    energy = np.sum(np.abs(atoms) ** 2, axis=-1)
    overlap = np.conj(atoms) @ truth
    shape = (-1, 1) if axis == 0 else (1, -1)
    return (
        np.abs(gain) ** 2 * energy.reshape(shape)
        - 2.0 * np.real(np.conj(gain) * overlap.reshape(shape))
        + np.sum(np.abs(truth) ** 2)
    )


def _self_check() -> None:
    """The closed forms must match numpy's own solvers, and the delay estimator must
    recover a delay that was put there on purpose."""
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(3, N_FREQ_BINS)) + 1j * rng.normal(size=(3, N_FREQ_BINS))
    y = rng.normal(size=(4, N_FREQ_BINS)) + 1j * rng.normal(size=(4, N_FREQ_BINS))
    mix = rng.normal(size=N_FREQ_BINS) + 1j * rng.normal(size=N_FREQ_BINS)

    gain_x, gain_y, residual = _fit_pairs(x, y, mix)
    for i in range(len(x)):
        for j in range(len(y)):
            expected, *_ = np.linalg.lstsq(np.stack((x[i], y[j]), axis=1), mix, rcond=None)
            assert np.allclose((gain_x[i, j], gain_y[i, j]), expected, rtol=1e-5, atol=1e-8)
            fit = gain_x[i, j] * x[i] + gain_y[i, j] * y[j]
            assert np.isclose(residual[i, j], np.sum(np.abs(fit - mix) ** 2))
            explicit = np.sum(np.abs(gain_x[i, j] * x[i] - mix) ** 2)
            assert np.isclose(_pair_error(gain_x, x, mix, axis=0)[i, j], explicit)

    signal = rng.normal(size=NFFT) * np.hanning(NFFT)
    atom = np.fft.rfft(signal)
    for true_delay in (-37, 0, 12):
        shifted = np.fft.rfft(np.roll(signal, true_delay))
        _, estimated = _align(atom[None, :], shifted)
        assert abs(estimated[0] - true_delay) < 0.6, (true_delay, estimated[0])

    # capacity is a *bound*: no ranked pair may beat the best single atom fitted
    # to the truth, or the row labelled `capacity` is not one and the 2.1 verdict
    # (the ramp is too tight / tight enough) is read off the wrong line.
    best = _capacity(x, mix)
    assert np.sum(np.abs(best - mix) ** 2) <= np.min(
        _pair_error(gain_x, x, mix, axis=0)
    ) + 1e-8, "capacity is not a bound"
    note("self-check: 2x2 fit matches lstsq, delay estimator recovers known shifts,")
    note("            capacity bounds every fitted pair")


def _capacity(atoms: ComplexArray, truth: ComplexArray) -> ComplexArray:
    """Estimate of the single atom the ramp model fits best to the *true* source.

    Upper bound of the parameterization itself, selection set aside: it asks what
    the dictionary can still reach once phase is only a delay plus an offset.
    """
    aligned, _ = _align(atoms, truth)
    energy = np.sum(np.abs(aligned) ** 2, axis=-1)
    overlap = np.conj(aligned) @ truth
    gain = overlap / np.where(energy > 0, energy, 1.0)
    best = int(np.argmin(_pair_error(gain[:, None], aligned, truth, axis=0)))
    return gain[best] * aligned[best]


def _run(
    dict1: ComplexArray,
    dict2: ComplexArray,
    mix: ComplexArray,
    truth1: ComplexArray,
    truth2: ComplexArray,
) -> dict[str, list[ComplexArray]]:
    """Per rule, the two estimated spectrograms of the pairs that rule selects.

    Spectrograms rather than accumulated error power: the metric is time-domain,
    so the estimate has to survive an overlap-add round trip, and that projection
    is not a constant offset on an inconsistent per-frame estimate. Cost is one
    (frames, bins) complex array per rule and source, `TEST_SECONDS` bounding it.
    """
    rules = (*PAIR_RULES, "capacity", *(f"ramp+r{alpha:g}" for alpha in ALPHAS if alpha > 0))
    out = {rule: [np.zeros_like(mix), np.zeros_like(mix)] for rule in rules}

    for t in range(len(mix)):
        # ponytail: each atom's delay is estimated against the *mixture*, so the other
        # source contaminates it. Iterating (align, fit, subtract, re-align) would clean
        # it up; do that only if `ramp` lands far below `capacity` for this reason.
        x, _ = _align(dict1, mix[t])
        y, _ = _align(dict2, mix[t])
        gain_x, gain_y, residual = _fit_pairs(x, y, mix[t])
        err1 = _pair_error(gain_x, x, truth1[t], axis=0)
        err2 = _pair_error(gain_y, y, truth2[t], axis=1)

        for rule, criterion in (("ramp", residual), ("ramp-oracle", err1 + err2)):
            i, j = np.unravel_index(int(np.argmin(criterion)), criterion.shape)
            # recomputed from the atoms, the closed forms above only ranked pairs
            estimate1 = gain_x[i, j] * x[i]
            estimate2 = gain_y[i, j] * y[j]
            out[rule][0][t] = estimate1
            out[rule][1][t] = estimate2

            if rule != "ramp":
                continue
            unexplained = mix[t] - estimate1 - estimate2
            energy1, energy2 = np.abs(estimate1) ** 2, np.abs(estimate2) ** 2
            share = energy1 / np.maximum(energy1 + energy2, 1e-30)
            for alpha in ALPHAS:
                if alpha == 0.0:
                    continue
                key = f"ramp+r{alpha:g}"
                out[key][0][t] = estimate1 + alpha * share * unexplained
                out[key][1][t] = estimate2 + alpha * (1 - share) * unexplained

        out["capacity"][0][t] = _capacity(dict1, truth1[t])
        out["capacity"][1][t] = _capacity(dict2, truth2[t])

    return out


def main() -> None:
    _self_check()
    rng = np.random.default_rng(SEED)
    pools, test_items = corpus(rng)
    note(f"pool: {[len(p) for p in pools]} candidate frames per source, max lag {MAX_LAG} samples")

    picks = [dictionary(pool, max(DICT_SIZES), rng) for pool in pools]
    base = {**config(), "method": METHOD, "max_lag": MAX_LAG, "ridge": RIDGE}
    done = done_cells()

    for name, references in test_items:
        if (name, "oracle") not in done:
            for row in oracle_rows(name, references):
                emit(row)
        spectra, mixture = test_spectra(references)
        note(f"track {name!r}: {len(mixture)} frames")
        for size in sorted(DICT_SIZES):
            cell = f"d{size}"
            if (name, cell) in done:
                note(f"  {cell} already in the journal, skipped")
                continue
            started = time.time()
            result = _run(
                pools[0][picks[0][:size]],
                pools[1][picks[1][:size]],
                mixture,
                spectra[0],
                spectra[1],
            )
            elapsed = time.time() - started
            for rule, estimates in result.items():
                for source, scores in enumerate(score(references, estimates, mixture)):
                    emit({
                        **base, "track": name, "cell": cell, "source": source, "rule": rule,
                        "dict_size": size, "frames": int(len(mixture)), "seconds": elapsed,
                        **scores,
                    })
            note(
                f"  {cell}: {elapsed:.0f}s, "
                f"{1000 * elapsed / len(mixture):.0f} ms/frame all rules included"
            )


if __name__ == "__main__":
    _self_check() if "--check" in sys.argv else main()
