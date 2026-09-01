"""Def-MAP diagnostic: is the dictionary the problem, or the selection rule?

Def-MAP (`gasm.rase.defmap`, thesis Proposition 2) reaches ~14 dB SDR on its
training set and collapses to 3.4 dB on test (thesis ch. 4, results table).
Two incompatible explanations, and every repair effort depends on which holds:

  A. the training library *does* contain a pair of spectra that reconstructs the
     test frame, and Def-MAP's own loss picks the wrong one -> the inference is
     broken, and reparameterizing the deformation is worth the work;
  B. no pair in the library reconstructs the test frame -> the dictionary itself
     does not generalize, and no search rule can save the method.

This measures A against B directly: on the held-out tracks of `defmap_protocol`,
it runs three selection rules over the *same* candidate pairs.

  defmap     argmin of Def-MAP's own loss(T1)+loss(T2). The method as it stands.
  magnitude  argmin of || |c1|+|c2| - |mix| ||^2, a phase-blind rule. Not part of
             the thesis: it pre-tests the hypothesis that the [Re;Im] feature
             makes distances meaningless, by asking how much of the oracle gap a
             rule that ignores phase already recovers.
  oracle     argmin of the true reconstruction error, the sources being known.
             Upper bound of every possible selection rule on this dictionary.

Reference lines are the five oracles of `evaluation.oracle_spectra`, emitted once
per test track: the mixture as its own estimate (floor), the IRM and the oracle
Wiener filter (what the literature usually calls the ceiling), and `best_real`,
the actual ceiling of the masking class, which sits ~6 dB above them. Def-MAP's
imaginary part *is* a mask on Im(mix) (see `_chunk_terms`), so the class ceiling
is the honest reference and the two mask oracles only say where the usual claim
would have landed.

Read the verdict as the **oracle-to-defmap gap**, not against the thesis' 12-14
dB / 5-6 dB thresholds: those were measured on the A-Volute voice+detonation
corpus (lost) with BSS-eval, this is MUSDB18 vocals+accompaniment. A large gap
means explanation A, a small gap at a low absolute level means explanation B.

The search space here is *more generous* than Def-MAP's own: `DefMAPSeparator`
indexes its library as [sound, frame], so a candidate for test frame t is the
training sound's own frame t, while this pools every training frame and lets any
atom pair with any test frame. An oracle on the flat pool therefore upper-bounds
the oracle on Def-MAP's indexed library, which is what the B verdict needs.

One JSONL row per (track, dictionary size, rule, source) on stdout, progress on
stderr, and a (track, cell) already in `RESUME_FROM` is skipped, so a sweep can be
relaunched rather than finished in one go. Cost is dominated by the largest
dictionary but the scores are evaluated for every test frame in one matrix product
(`_pair_scores_frames`), which puts a cell at ~7 s for 50 atoms/source and ~1 min
for 300 on one core of a Zen 3, against 75 s and 46 min for the per-frame form it
replaced. `N_TEST` and `TEST_SECONDS` are the cost knobs. Usage, in the container
that has ffmpeg:

    MUSDB_ROOT=/path/to/musdb18-7s python experiments/defmap_oracle_pair.py
    DICT_SIZES=50,100 TEST_SECONDS=5 N_TEST=2 ... python experiments/...
    sh experiments/run.sh defmap        # the resumable form, one cell per log
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

from defmap_protocol import (
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
from gasm.rase.defmap import (  # isort: skip
    _deformation_loss,
    apply_deformation,
    solve_complex_deformation,
)

# The STFT, the corpus and the metric all come from `defmap_protocol`, so this
# experiment and article 1 measure the same quantity on the same material, and its
# defaults are the thesis values (Delta t = 1024, delta t = 512 at 44.1 kHz).
# Def-MAP's thesis dictionary held ~10k spectrum pairs, i.e. ~100 atoms per source,
# so the sizes below bracket the thesis scale rather than chase a bigger library.
DICT_SIZES = tuple(int(k) for k in os.environ.get("DICT_SIZES", "50,100,300").split(","))

_CHUNK_ELEMENTS = 4_000_000  # rows per chunk sized to keep each temporary around 32 MB
# candidates per rule and per chunk that the fast form hands to the exact one. The
# gap it has to absorb is the fast form's own error, not a modelling margin, so a
# few dozen is already far past what the replay against the journal needs.
SHORTLIST = int(os.environ.get("SHORTLIST", "32"))
RULES = ("defmap", "magnitude", "oracle")
METHOD = "pair"  # the `method` field of every row this script writes


def _safe_share(num: FloatArray, den: FloatArray) -> FloatArray:
    """num/den, zero where den vanishes, matching `_safe_div`'s identity transform."""
    return np.divide(num, den, out=np.zeros_like(den), where=den > 0)


def _terms(c1: ComplexArray, c2: ComplexArray) -> dict[str, FloatArray]:
    """Frame-independent part of the three scores, for two broadcastable candidate sets.

    Called two ways: `_chunk_terms` shapes it into every pair of a chunk, and
    `_run` calls it on aligned candidate lists to rescore a shortlist exactly.

    Closed forms, derived from `solve_complex_deformation` + `apply_deformation`
    and asserted against them in `_self_check`. With a = Re(c1), b = Re(c2),
    m = Re(mix) and the real residual r = m - a - b:

        est1.real = a + a^2 * r / (a^2+b^2)      est2.real = b + b^2 * r / (a^2+b^2)
        loss real = sum_bins r^2 / (a^2+b^2)

    and with p = Im(c1), q = Im(c2):

        est1.imag = p^2 * Im(m) / (p^2+q^2)      est2.imag = q^2 * Im(m) / (p^2+q^2)
        loss imag = sum_bins Im(m)^2 / (p^2+q^2)

    Two structural facts fall out and are worth more than the speedup. The
    imaginary part of the estimate is a Wiener mask on the mixture's imaginary
    part: it carries no candidate phase beyond an energy ratio, and always takes
    the sign of Im(mix). And the imaginary loss *decreases* with the candidates'
    imaginary energy without ever comparing them to the mixture, so the criterion
    is biased towards loud atoms.

    Zero denominators follow `_safe_div`: identity transform, hence a zero loss
    contribution and an estimate equal to the candidate. That makes a silent pair
    look perfect to the criterion, which is why the dictionary is energy-filtered.
    """
    a, p = c1.real, c1.imag
    b, q = c2.real, c2.imag
    den_r, den_i = a * a + b * b, p * p + q * q
    ones = np.ones_like(den_r)
    return {
        "sum_real": a + b,
        "inv_den_r": _safe_share(ones, den_r),
        "inv_den_i": _safe_share(np.ones_like(den_i), den_i),
        "share1_r": _safe_share(a * a, den_r),
        "share2_r": _safe_share(b * b, den_r),
        "share1_i": _safe_share(p * p, den_i),
        "share2_i": _safe_share(q * q, den_i),
        "atom1_real": a * ones,
        "atom2_real": b * ones,
        "abs_sum": np.abs(c1) + np.abs(c2),
    }


def _chunk_terms(dict1: ComplexArray, dict2: ComplexArray) -> dict[str, FloatArray]:
    """`_terms` shaped into every pair of the chunk: (dict1, dict2, bins)."""
    return _terms(dict1[:, None, :], dict2[None, :, :])


def _pair_scores_frames(
    dict1: ComplexArray,
    dict2: ComplexArray,
    mix: ComplexArray,
    truth1: ComplexArray,
    truth2: ComplexArray,
) -> dict[str, FloatArray]:
    """The three scores of every pair of the chunk, for *every* test frame at once.

    Same quantities as `_pair_scores`, which stays as the reference implementation
    the self-check pins to the library. Every score is a sum over bins of a
    pair-dependent factor times a frame-dependent one, so expanding the squares
    turns the frame loop into a matrix product `(pairs, bins) @ (bins, frames)`.
    Def-MAP's real part, with s = a+b and r = m_r - s:

        sum_bins r^2 / den = inv @ m_r^2 - 2 (inv*s) @ m_r + sum_bins inv*s^2

    and the oracle's four error components expand the same way, err1_r being
    (a - share1_r*s) + share1_r*m_r - truth1_r, i.e. a pair-constant plus a
    pair-weighted frame term. Sixteen products cover the three rules.

    This is worth the algebra rather than being premature: the per-frame form
    re-streams ten (chunk_pairs, bins) arrays for each of the ~430 frames, so it
    is bound by memory bandwidth, while gemm keeps its panels in cache. Measured
    on the 300-atom cell, 429 frames: 51.7 s against 2790 s, scores agreeing to
    1e-15 relative and selecting the same pair on every frame.
    """
    n1, freqbin = dict1.shape
    n2 = dict2.shape[0]

    def spread(x: FloatArray) -> FloatArray:
        """A per-pair factor as a contiguous (pairs, bins) matrix, gemm's left operand."""
        return np.ascontiguousarray(np.broadcast_to(x, (n1, n2, freqbin)).reshape(n1 * n2, freqbin))

    a, p = dict1.real[:, None, :], dict1.imag[:, None, :]
    b, q = dict2.real[None, :, :], dict2.imag[None, :, :]
    ones = np.ones((n1, n2, freqbin))
    inv_r = spread(_safe_share(ones, a * a + b * b))
    inv_i = spread(_safe_share(ones, p * p + q * q))
    sum_r = spread(a + b)
    abs_sum = spread(np.abs(dict1)[:, None, :] + np.abs(dict2)[None, :, :])

    # (bins, frames), so the right operand of every product is shared across chunks
    mix_r, mix_i = mix.real.T.copy(), mix.imag.T.copy()
    mix_r2, mix_i2, abs_mix = mix_r * mix_r, mix_i * mix_i, np.abs(mix).T.copy()

    defmap = inv_r @ mix_r2 - 2.0 * (inv_r * sum_r) @ mix_r + inv_i @ mix_i2
    defmap += (inv_r * sum_r * sum_r).sum(axis=1)[:, None]

    magnitude = (abs_sum * abs_sum).sum(axis=1)[:, None] - 2.0 * abs_sum @ abs_mix
    magnitude += (abs_mix * abs_mix).sum(axis=0)[None, :]

    oracle = np.zeros_like(defmap)
    for atom_r, share_r, share_i, truth in ((a, a * a, p * p, truth1), (b, b * b, q * q, truth2)):
        weight_r, weight_i = spread(share_r) * inv_r, spread(share_i) * inv_i
        true_r, true_i = truth.real.T.copy(), truth.imag.T.copy()
        constant = spread(atom_r) - weight_r * sum_r
        oracle += (constant * constant).sum(axis=1)[:, None]
        oracle += (weight_r * weight_r) @ mix_r2 + 2.0 * (constant * weight_r) @ mix_r
        oracle -= 2.0 * (constant @ true_r + weight_r @ (mix_r * true_r))
        oracle += (weight_i * weight_i) @ mix_i2 - 2.0 * weight_i @ (mix_i * true_i)
        oracle += (true_r * true_r + true_i * true_i).sum(axis=0)[None, :]
    return {"defmap": defmap, "magnitude": magnitude, "oracle": oracle}


def _pair_scores(
    terms: dict[str, FloatArray],
    spec_mix: ComplexArray,
    truth1: ComplexArray,
    truth2: ComplexArray,
) -> dict[str, FloatArray]:
    """The three selection scores of every pair in the chunk, for one test frame.

    Reference implementation: `_self_check` pins it to `gasm.rase.defmap` and
    `_pair_scores_frames` to it, `_run` calling only the latter.
    """
    residual = spec_mix.real - terms["sum_real"]
    defmap = (residual * residual * terms["inv_den_r"]).sum(axis=-1)
    defmap += (spec_mix.imag**2 * terms["inv_den_i"]).sum(axis=-1)

    err1_r = terms["atom1_real"] + terms["share1_r"] * residual - truth1.real
    err1_i = terms["share1_i"] * spec_mix.imag - truth1.imag
    err2_r = terms["atom2_real"] + terms["share2_r"] * residual - truth2.real
    err2_i = terms["share2_i"] * spec_mix.imag - truth2.imag
    oracle = (err1_r**2 + err1_i**2 + err2_r**2 + err2_i**2).sum(axis=-1)

    magnitude = ((terms["abs_sum"] - np.abs(spec_mix)) ** 2).sum(axis=-1)
    return {"defmap": defmap, "magnitude": magnitude, "oracle": oracle}


def _self_check() -> None:
    """`_pair_scores` must agree with the library it shortcuts, or every number
    below is measuring something other than Def-MAP."""
    rng = np.random.default_rng(SEED)
    freqbin = 7
    dict1 = rng.normal(size=(3, freqbin)) + 1j * rng.normal(size=(3, freqbin))
    dict2 = rng.normal(size=(4, freqbin)) + 1j * rng.normal(size=(4, freqbin))
    dict1[2] = 0.0  # vanishing denominators, i.e. the identity-transform branch
    truth1 = rng.normal(size=freqbin) + 1j * rng.normal(size=freqbin)
    truth2 = rng.normal(size=freqbin) + 1j * rng.normal(size=freqbin)
    spec_mix = truth1 + truth2

    scores = _pair_scores(_chunk_terms(dict1, dict2), spec_mix, truth1, truth2)
    for i in range(len(dict1)):
        for j in range(len(dict2)):
            t1, t2 = solve_complex_deformation(dict1[i], dict2[j], spec_mix)
            est1 = apply_deformation(t1, dict1[i])
            est2 = apply_deformation(t2, dict2[j])
            assert np.isclose(scores["defmap"][i, j], _deformation_loss(t1) + _deformation_loss(t2))
            expected = float(
                np.sum(np.abs(est1 - truth1) ** 2) + np.sum(np.abs(est2 - truth2) ** 2)
            )
            assert np.isclose(scores["oracle"][i, j], expected)

    # the aligned shaping `_run` rescores with must give the outer form's numbers on
    # the same pairs, a shape slip there being invisible in the results
    rows, cols = np.divmod(np.arange(len(dict1) * len(dict2)), len(dict2))
    aligned = _pair_scores(_terms(dict1[rows], dict2[cols]), spec_mix, truth1, truth2)
    for rule in RULES:
        assert np.array_equal(aligned[rule].reshape(len(dict1), len(dict2)), scores[rule])

    # and the gemm form must agree with the reference it replaces, on two frames so
    # that a frame-axis mix-up cannot pass. Expanding the squares costs a few digits
    # to cancellation, hence a relative tolerance rather than isclose's default.
    frames = np.stack((spec_mix, truth1 - truth2))
    fast = _pair_scores_frames(dict1, dict2, frames, np.stack((truth1, truth1)),
                               np.stack((truth2, truth2)))
    slow = [_pair_scores(_chunk_terms(dict1, dict2), f, truth1, truth2) for f in frames]
    for rule in RULES:
        for t, one in enumerate(slow):
            got = fast[rule][:, t].reshape(len(dict1), len(dict2))
            assert np.allclose(got, one[rule], rtol=1e-9, atol=1e-9 * one[rule].max())

    # through `note`, not print: main() runs this before writing rows, and stdout
    # is the journal
    note("self-check: fast pair scores match gasm.rase.defmap, gemm form matches both")


def _run(
    dict1: ComplexArray,
    dict2: ComplexArray,
    mix: ComplexArray,
    truth1: ComplexArray,
    truth2: ComplexArray,
) -> dict[str, list[ComplexArray]]:
    """Per rule, the two estimated spectrograms of the pairs that rule selects.

    Spectrograms rather than accumulated error power, because the metric is now
    time-domain: the estimate has to be inverted, and inverting it frame by frame
    is not the same thing as inverting the sequence (overlap-add projects the
    inconsistent per-frame estimate back onto a real signal). Memory is one
    (frames, bins) complex array per rule and source, ~3.5 MB per five seconds
    of test audio, which is why `TEST_SECONDS` is the knob that bounds it.
    """
    n_frames, freqbin = mix.shape
    best_pair = {rule: np.zeros((n_frames, 2), dtype=np.intp) for rule in RULES}
    rows_per_chunk = max(1, _CHUNK_ELEMENTS // (len(dict2) * freqbin))

    # rank with the gemm form, decide with the exact one. Expanding the squares
    # costs relative precision exactly where a score is small against its terms,
    # i.e. on the pairs that win, and Def-MAP's 1/(a^2+b^2) amplifies it: deciding
    # on the gemm score alone moved a track's SDR by 0.055 dB through tie flips,
    # while the two rules without that factor stayed within 1e-6 dB. Ranking
    # tolerates the error, so the shortlist is taken from the fast form and the
    # winner from the reference one, which reproduces the per-frame kernel exactly.
    shortlist: dict[str, list[tuple[np.ndarray, FloatArray]]] = {rule: [] for rule in RULES}
    for start in range(0, len(dict1), rows_per_chunk):
        stop = min(start + rows_per_chunk, len(dict1))
        scores = _pair_scores_frames(dict1[start:stop], dict2, mix, truth1, truth2)
        for rule in RULES:
            keep = min(SHORTLIST, len(scores[rule]))
            top = np.argpartition(scores[rule], keep - 1, axis=0)[:keep]
            value = np.take_along_axis(scores[rule], top, axis=0)
            shortlist[rule].append((top + start * len(dict2), value))

    # pruned across chunks too, so the exact pass costs the same at any dictionary
    # size instead of growing with the chunk count
    candidates = {}
    for rule, blocks in shortlist.items():
        index = np.concatenate([block for block, _ in blocks], axis=0)
        value = np.concatenate([values for _, values in blocks], axis=0)
        keep = min(SHORTLIST, len(index))
        best = np.argpartition(value, keep - 1, axis=0)[:keep]
        candidates[rule] = np.take_along_axis(index, best, axis=0)

    for t in range(n_frames):
        # one exact pass per frame over the union, so a rule can only gain from
        # another's candidates, and the lowest flat index still breaks ties
        flat = np.unique(np.concatenate([candidates[rule][:, t] for rule in RULES]))
        rows, cols = np.divmod(flat, len(dict2))
        exact = _pair_scores(_terms(dict1[rows], dict2[cols]), mix[t], truth1[t], truth2[t])
        for rule in RULES:
            best = int(exact[rule].argmin())
            best_pair[rule][t] = (rows[best], cols[best])

    # the reported estimates go back through the library, the kernel only ranked pairs
    out: dict[str, list[ComplexArray]] = {}
    for rule in RULES:
        estimates = [np.zeros_like(mix), np.zeros_like(mix)]
        for t in range(n_frames):
            i, j = best_pair[rule][t]
            t1, t2 = solve_complex_deformation(dict1[i], dict2[j], mix[t])
            estimates[0][t] = apply_deformation(t1, dict1[i])
            estimates[1][t] = apply_deformation(t2, dict2[j])
        out[rule] = estimates
    return out


def main() -> None:
    _self_check()
    rng = np.random.default_rng(SEED)
    pools, test_items = corpus(rng)
    note(f"pool: {[len(p) for p in pools]} candidate frames per source, {N_FREQ_BINS} bins")

    # nested draws, so a bigger dictionary is a superset and the trend against
    # dictionary size is monotone by construction rather than by luck
    picks = [dictionary(pool, max(DICT_SIZES), rng) for pool in pools]
    base = {**config(), "method": METHOD, "dict_pool": [int(len(p)) for p in pools]}
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
            note(f"  {cell}: {size * size} pairs per frame, {elapsed:.0f}s")


if __name__ == "__main__":
    _self_check() if "--check" in sys.argv else main()
