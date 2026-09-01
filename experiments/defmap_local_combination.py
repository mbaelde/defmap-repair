"""Def-MAP repair 2.3: one atom per source is the wall, so use k of them.

`defmap_phase_ramp.py` left the method in an awkward place. Constraining the
phase to a delay repaired the selection (the oracle-to-criterion gap fell from
8.5 dB to 0.9 dB) and brought inference under the real-time budget, but two
numbers spoiled the story: the pure ramp caps at +7.0 dB against the +12.9 dB the
free deformation could reach, and once the unexplained residual is reabsorbed the
quality stops depending on the dictionary at all, drifting *down* from +8.0 dB at
50 atoms to +7.6 dB at 300. A method built on learned exemplars whose quality
ignores its exemplars has a hole in its argument.

The suspected cause is the one-atom-per-source assumption, which is the classic
weakness of exemplar methods: the solution is tied to a single stored spectrum,
weakly deformed. The classic remedy is a local combination, k aligned atoms per
source rather than one, which is also what puts the method in the same family as
NMF and gives the paper a natural comparison axis.

Two questions, the same pair as before, and the second one matters more:

  does quality climb back towards the +12.9 dB oracle as k grows?
  does the slope against dictionary size turn positive again? A method whose
  quality improves with more training data has an argument; one whose quality
  drifts down does not, whatever its absolute level.

Pair search disappears here, which makes this cheaper than the pair rule it
replaces: the k best atoms per source are read off the alignment correlation
already computed, and one joint complex least squares of dimension k1+k2 fits
them all to the mixture at once. k=1 must reproduce the previous rule exactly,
and `_self_check` asserts it against `_fit_pairs`.

Experiment 2.5, added once the k sweep was measured, lives here too. That sweep
answered the second question above with a *no*: capacity climbs from +7.8 to
+12.3 dB between k=1 and k=16 while delivered quality stays flat around +5.5 and
falls back at k=16, so the entire benefit of a larger k is absorbed by the
selection rule. The diagnosis is the cross term of section 4.5,

    <c, x> = <c, s1> + <c, s2>

which makes an atom of source 1 that resembles source 2 score well on the
mixture. Two repairs, both selecting exactly k atoms per source so that they
live in the same model class as `local k=N` and share its `capacity k=N`
ceiling, which is what makes the gap directly comparable:

  `xtalk`   the matched-filter score discounted by each atom's intrinsic
            ambiguity, its best-delay coherence with the *other* source's
            dictionary. Computed once per dictionary pair rather than per frame,
            which is what keeps selection linear in the dictionary.
  `greedy`  joint selection by deflation, one atom at a time against the current
            residual under a per-source quota. Once the content of s2 is in the
            model the residual stops rewarding atoms of source 1 that explain it,
            so this is the one that attacks the cross term head on.

Each has a control point that must reproduce the unrepaired rule: `l=0` for the
first, and for the second a monotone residual plus an exact quota.

Corpus, metric, oracles and journal are `defmap_protocol`'s, so a row here is
comparable to a row of 2.0 / 2.1 and to article 1's.

    MUSDB_ROOT=/path/to/musdb18-7s python experiments/defmap_local_combination.py
    K_VALUES=1,4 DICT_SIZES=50 TEST_SECONDS=5 N_TEST=2 MUSDB_ROOT=... python ...
    sh experiments/run.sh defmap        # the resumable form, one cell per log
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

from defmap_phase_ramp import ALPHAS, MAX_LAG, RIDGE, _align, _fit_pairs

from defmap_oracle_pair import DICT_SIZES  # isort: skip
from defmap_protocol import (  # isort: skip
    NFFT,
    N_FREQ_BINS,
    SEED,
    ComplexArray,
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

K_VALUES = tuple(int(k) for k in os.environ.get("K_VALUES", "1,2,4,8,16").split(","))
# experiment 2.5: strength of the cross-talk discount. 0 is the control point and is
# always run, being exactly the unrepaired rule. 1 is the other end worth reading, an
# atom the other dictionary holds outright being struck off; above 1 the discount goes
# negative and the ranking stops being an attenuation of the score, so the sweep stops
# there.
XTALK_LAMBDAS = tuple(
    float(v) for v in os.environ.get("XTALK_LAMBDAS", "0.5,1").split(",") if v
)
# the two new families carry only these: selection is read at alpha=0 by definition,
# and 0.75 is the measured optimum of the quality line. Carrying all of ALPHAS would
# put peak memory near a gigabyte for no reading.
XTALK_ALPHAS = (0.75,)
METHOD = "local"  # the `method` field of every row this script writes


def _top_k(
    atoms: ComplexArray, target: ComplexArray, k: int, discount: np.ndarray | None = None
) -> np.ndarray:
    """Indices of the k atoms most correlated with `target`, energy-normalized.

    |x^H t| / ||x|| is the matched-filter score, and normalizing is what stops the
    ranking from being a loudness contest -- the bias `defmap_oracle_pair.py`
    found baked into Def-MAP's own imaginary loss.

    `discount` multiplies the score per atom, which is how experiment 2.5's
    cross-talk penalty enters. None means the plain rule, and a discount of all
    ones must give the same picks.
    """
    norm = np.sqrt(np.sum(np.abs(atoms) ** 2, axis=-1))
    score = np.abs(np.conj(atoms) @ target) / np.maximum(norm, 1e-30)
    if discount is not None:
        score = score * discount
    return np.argpartition(-score, min(k, len(atoms) - 1))[:k]


def _ambiguity(atoms: ComplexArray, other: ComplexArray, block: int = 16) -> np.ndarray:
    """Per atom of `atoms`, its best-delay coherence with the *other* dictionary.

    max over j and over the searched lags of the cross-correlation between atom i
    and the other source's atom j, normalized by both autocorrelations at zero lag,
    so this is a cosine in [-1, 1] worth exactly 1 for an atom the other dictionary
    holds up to a delay. `np.fft.irfft` of conj(a) * b is the cross-correlation of
    the two real signals the half spectra stand for, so Cauchy-Schwarz gives the
    bound and the normalization needs no convention constant.

    Delay-invariant by construction, hence computable once per dictionary pair
    instead of once per frame -- that is the whole point, a per-frame version would
    reintroduce the K1*K2 term the ramp rule was built to drop and would break the
    complexity claim of section 5.
    """
    lags = np.concatenate((np.arange(MAX_LAG + 1), np.arange(-MAX_LAG, 0)))
    energy_a = np.fft.irfft(np.abs(atoms) ** 2, n=NFFT, axis=-1)[:, 0]
    energy_o = np.fft.irfft(np.abs(other) ** 2, n=NFFT, axis=-1)[:, 0]
    out = np.empty(len(atoms))
    for start in range(0, len(atoms), block):  # (K1, K2, NFFT) at once would be ~700 MB
        chunk = atoms[start : start + block]
        corr = np.fft.irfft(np.conj(chunk)[:, None, :] * other, n=NFFT, axis=-1)
        peak = corr[:, :, lags].max(axis=-1)
        scale = np.sqrt(np.outer(energy_a[start : start + block], energy_o))
        out[start : start + block] = (peak / np.maximum(scale, 1e-30)).max(axis=-1)
    return np.clip(out, 0.0, 1.0)


def _joint_fit(columns: ComplexArray, target: ComplexArray) -> ComplexArray:
    """Complex gains of min || sum_i g_i * columns[i] - target ||^2.

    Normal equations with the same relative ridge as `_fit_pairs`, so k=1 per
    source reproduces the pair rule bit for bit rather than approximately.
    """
    gram = np.conj(columns) @ columns.T
    ridge = RIDGE * np.real(np.trace(gram))
    gram = gram + ridge * np.eye(len(columns))
    return np.linalg.solve(gram, np.conj(columns) @ target)


def _combine(
    aligned1: ComplexArray,
    aligned2: ComplexArray,
    target: ComplexArray,
    k: int,
    discounts: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[ComplexArray, ComplexArray]:
    """Estimates of both sources from their k best aligned atoms, fitted jointly.

    `discounts` are experiment 2.5's per-atom cross-talk penalties, one array per
    dictionary. None is the unrepaired rule.
    """
    penalty1, penalty2 = discounts if discounts is not None else (None, None)
    pick1 = _top_k(aligned1, target, k, penalty1)
    pick2 = _top_k(aligned2, target, k, penalty2)
    x, y = aligned1[pick1], aligned2[pick2]
    gains = _joint_fit(np.concatenate((x, y)), target)
    return gains[: len(x)] @ x, gains[len(x) :] @ y


def _greedy(
    aligned1: ComplexArray, aligned2: ComplexArray, target: ComplexArray, k: int
) -> tuple[ComplexArray, ComplexArray]:
    """Joint selection by deflation under a per-source quota of k atoms.

    Both dictionaries compete at every step for the atom best correlated with the
    *current residual*, and a source drops out of the race once it holds k atoms.
    Deflation is what answers section 4.5: the independent per-source top-k ranks
    atoms on <c, x> = <c, s1> + <c, s2>, so an atom of source 1 that resembles
    source 2 wins on a term that is not its own; once s2's content is in the model
    the residual no longer pays for explaining it.

    The quota keeps the selected set in the same model class as `_combine` at the
    same k, which is what makes `capacity k=N` the shared ceiling of both rules and
    the gap to it the quantity of interest. 2k steps, each with a refit of size at
    most 2k, so the cost is quadratic in k and linear in the dictionary.
    """
    columns = np.concatenate((aligned1, aligned2))
    from_first = np.arange(len(columns)) < len(aligned1)
    norm = np.maximum(np.sqrt(np.sum(np.abs(columns) ** 2, axis=-1)), 1e-30)
    residual, chosen, held = target.copy(), [], [0, 0]
    for _ in range(2 * k):
        score = np.abs(np.conj(columns) @ residual) / norm
        score[chosen] = -1.0  # no atom twice
        if held[0] >= k:  # a source that reached its quota leaves the race
            score[from_first] = -1.0
        if held[1] >= k:
            score[~from_first] = -1.0
        pick = int(np.argmax(score))
        held[0 if from_first[pick] else 1] += 1
        chosen.append(pick)
        gains = _joint_fit(columns[chosen], target)  # needed for the next residual
        residual = target - gains @ columns[chosen]
    picked, first = np.asarray(chosen), from_first[chosen]
    return gains[first] @ columns[picked][first], gains[~first] @ columns[picked][~first]


def _spread(
    estimate1: ComplexArray, estimate2: ComplexArray, target: ComplexArray, alpha: float
) -> tuple[ComplexArray, ComplexArray]:
    """The residual dial: reabsorb alpha of what neither estimate explains, split by
    their energies. alpha=1 restores exact summation, alpha=0 leaves selection alone,
    which is why every selection reading in the paper is taken at alpha=0."""
    unexplained = target - estimate1 - estimate2
    energy1, energy2 = np.abs(estimate1) ** 2, np.abs(estimate2) ** 2
    share = energy1 / np.maximum(energy1 + energy2, 1e-30)
    return (
        estimate1 + alpha * share * unexplained,
        estimate2 + alpha * (1 - share) * unexplained,
    )


def _capacity(aligned: ComplexArray, truth: ComplexArray, k: int) -> ComplexArray:
    """Best k-atom combination *for this source alone*.

    Atoms aligned on the truth and chosen against the truth, so this is the
    ceiling of the model class at this k, selection set aside.
    """
    picked = aligned[_top_k(aligned, truth, k)]
    return _joint_fit(picked, truth) @ picked


def _self_check() -> None:
    """k=1 per source must be the previous rule, and the joint solve must be a
    least squares. Without the first assertion the k sweep has no control point."""
    rng = np.random.default_rng(SEED)
    bins = 41
    x = rng.normal(size=(3, bins)) + 1j * rng.normal(size=(3, bins))
    y = rng.normal(size=(4, bins)) + 1j * rng.normal(size=(4, bins))
    target = rng.normal(size=bins) + 1j * rng.normal(size=bins)

    gain_x, gain_y, _ = _fit_pairs(x, y, target)
    for i in range(len(x)):
        for j in range(len(y)):
            joint = _joint_fit(np.stack((x[i], y[j])), target)
            assert np.allclose(joint, (gain_x[i, j], gain_y[i, j]), rtol=1e-6, atol=1e-10)

    columns = np.concatenate((x, y))
    expected, *_ = np.linalg.lstsq(columns.T, target, rcond=None)
    assert np.allclose(_joint_fit(columns, target), expected, rtol=1e-5, atol=1e-8)

    # experiment 2.5. Full-width atoms, the ambiguity being an irfft over NFFT.
    wide1 = rng.normal(size=(5, N_FREQ_BINS)) + 1j * rng.normal(size=(5, N_FREQ_BINS))
    wide2 = rng.normal(size=(6, N_FREQ_BINS)) + 1j * rng.normal(size=(6, N_FREQ_BINS))
    mixture = wide1[0] + wide2[3]

    # an atom the other dictionary holds scores 1, and holding it *up to a delay* still
    # scores 1: that invariance is what licenses computing this once per cell
    assert np.allclose(_ambiguity(wide1, wide1), 1.0)
    twin = wide1[0] * np.exp(-2j * np.pi * 7 * np.arange(N_FREQ_BINS) / NFFT)
    assert abs(_ambiguity(wide1, np.concatenate((wide2, twin[None])))[0] - 1.0) < 1e-6
    assert _ambiguity(wide1, wide2).max() < 0.99  # unrelated atoms must not saturate

    # lambda=0 must be the unrepaired rule exactly, or the sweep has no control point
    plain = _combine(wide1, wide2, mixture, 2)
    zero = _combine(wide1, wide2, mixture, 2, (np.ones(len(wide1)), np.ones(len(wide2))))
    assert all(np.allclose(a, b) for a, b in zip(plain, zero))

    # greedy's quota: both sources get atoms, and each estimate stays inside its own
    # dictionary's span -- the bug that would silently invalidate the whole comparison
    for k in (1, 2, 3):
        for estimate, atoms in zip(_greedy(wide1, wide2, mixture, k), (wide1, wide2)):
            assert np.linalg.norm(estimate) > 0
            _, residuals, *_ = np.linalg.lstsq(atoms.T, estimate, rcond=None)
            assert residuals[0] < 1e-12 * np.linalg.norm(estimate) ** 2 + 1e-24
    note("self-check: k=1 = pair rule, lstsq, ambiguity invariant, greedy in span")


def _run(
    dict1: ComplexArray,
    dict2: ComplexArray,
    mix: ComplexArray,
    truth1: ComplexArray,
    truth2: ComplexArray,
) -> dict[str, list[ComplexArray]]:
    """Per rule and per k, the two estimated spectrograms of that rule.

    Spectrograms rather than accumulated error power, the metric being time-domain
    now (see `defmap_protocol`). This is the memory-hungry script of the three: with
    experiment 2.5's families the default sweep holds 60 rules times two sources,
    ~3.5 MB each per five seconds of test audio, so ~420 MB at `TEST_SECONDS=5`.
    Raising `TEST_SECONDS` scales that linearly, and 30 would be near 2.5 GB.

    ponytail: the frame loop stays outside the k loop, so alignment (the dominant
    cost) is computed once for every k. Moving the k loop out would cut peak
    memory by len(ks) and cost the same factor in alignment time, worth doing only
    if the full protocol turns out to be memory-bound rather than time-bound.
    """
    ceiling = min(len(dict1), len(dict2))
    ks = sorted({min(k, ceiling) for k in K_VALUES})  # a small dictionary caps k
    rules = [f"local k={k}" for k in ks]
    rules += [f"local k={k} +r{a:g}" for k in ks for a in ALPHAS if a > 0]
    rules += [f"xtalk k={k} l={lam:g}" for k in ks for lam in XTALK_LAMBDAS]
    rules += [
        f"xtalk k={k} l={lam:g} +r{a:g}"
        for k in ks
        for lam in XTALK_LAMBDAS
        for a in XTALK_ALPHAS
        if a > 0
    ]
    rules += [f"greedy k={k}" for k in ks]
    rules += [f"greedy k={k} +r{a:g}" for k in ks for a in XTALK_ALPHAS if a > 0]
    rules += [f"capacity k={k}" for k in ks]
    out = {rule: [np.zeros_like(mix), np.zeros_like(mix)] for rule in rules}

    # experiment 2.5's discounts: delay-invariant, so once per cell rather than per
    # frame. This is the line that keeps the rule linear in the dictionary.
    amb1, amb2 = _ambiguity(dict1, dict2), _ambiguity(dict2, dict1)
    discounts = {lam: (1 - lam * amb1, 1 - lam * amb2) for lam in XTALK_LAMBDAS}
    note(f"  ambiguity: mean {amb1.mean():.3f} / {amb2.mean():.3f} per source")

    for t in range(len(mix)):
        # alignment is the dominant cost and does not depend on k, so it stays out here
        aligned1, _ = _align(dict1, mix[t])
        aligned2, _ = _align(dict2, mix[t])
        capacity1, _ = _align(dict1, truth1[t])
        capacity2, _ = _align(dict2, truth2[t])

        for k in ks:
            variants = [(f"local k={k}", _combine(aligned1, aligned2, mix[t], k), ALPHAS)]
            variants += [
                (
                    f"xtalk k={k} l={lam:g}",
                    _combine(aligned1, aligned2, mix[t], k, discounts[lam]),
                    XTALK_ALPHAS,
                )
                for lam in XTALK_LAMBDAS
            ]
            variants += [
                (f"greedy k={k}", _greedy(aligned1, aligned2, mix[t], k), XTALK_ALPHAS)
            ]
            for label, (estimate1, estimate2), alphas in variants:
                out[label][0][t], out[label][1][t] = estimate1, estimate2
                for alpha in alphas:
                    if alpha == 0.0:
                        continue
                    key = f"{label} +r{alpha:g}"
                    out[key][0][t], out[key][1][t] = _spread(
                        estimate1, estimate2, mix[t], alpha
                    )

            out[f"capacity k={k}"][0][t] = _capacity(capacity1, truth1[t], k)
            out[f"capacity k={k}"][1][t] = _capacity(capacity2, truth2[t], k)

    return out


def main() -> None:
    _self_check()
    rng = np.random.default_rng(SEED)
    pools, test_items = corpus(rng)
    note(f"pool: {[len(p) for p in pools]} candidate frames per source, k in {K_VALUES}")

    picks = [dictionary(pool, max(DICT_SIZES), rng) for pool in pools]
    base = {**config(), "method": METHOD, "ridge": RIDGE}
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
                # the k of a rule is a column of its own, so the reporter can plot
                # against k without parsing the label back apart
                k = int(rule.split("k=")[1].split(" ")[0])
                for source, scores in enumerate(score(references, estimates, mixture)):
                    emit({
                        **base, "track": name, "cell": cell, "source": source, "rule": rule,
                        "dict_size": size, "k": k, "frames": int(len(mixture)),
                        "seconds": elapsed, **scores,
                    })
            note(f"  {cell}: {len(result)} rules, {elapsed:.0f}s")


if __name__ == "__main__":
    _self_check() if "--check" in sys.argv else main()
