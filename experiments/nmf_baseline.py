"""Supervised NMF on the same protocol, the baseline the repaired method owes.

The three Def-MAP probes are read against oracles, which tells how much of the
masking ceiling a rule reaches but not whether a standard method reaches more for
the same money. This is that comparison, and it is the one a reviewer asks for
first: NMF is the exemplar method's own family, learned non-negative spectra per
source, and its supervised form runs frame by frame against a fixed basis, so its
latency is comparable rather than merely of the same order.

    train    one basis per source, `size` components, learned on exactly the
             frames Def-MAP's dictionary is drawn from: the loud half of that
             source's pool, `defmap_protocol.dictionary` deciding which. Both
             methods therefore see the same training data, and the silence that
             biases Def-MAP's criterion is kept out of the basis too.
    infer    activations of the concatenated basis against the mixture magnitude,
             basis held fixed, then the two partial reconstructions define a mask
             and the mask is applied to the mixture with its own phase.

Activations are solved for the whole test spectrogram in one call, which is not a
lookahead: with the basis fixed the loss separates over frames, so the solution of
frame t is the same whether its neighbours are in the call or not. That is what
makes the deployed cost `bench_inference_cost.py` reports the honest one.

Four rules per dictionary size:

  nmf           the Wiener form, mask on the ratio of *powers*, which is what the
                repo's oracle Wiener uses, so the baseline and its own oracle are
                built the same way
  nmf p=1       the ratio of magnitudes, the other convention in the literature.
                Both are here because which one wins is a measurement, and a
                baseline argued down by its weaker variant proves nothing.
  nmf i=N       the same rule held to `NMF_FAST_ITER` multiplicative updates
                instead of MAX_ITER. Activation iterations are the whole per-frame
                cost, so this is the row that fixes the latency rather than
                letting it follow convergence: `bench_inference_cost.py` measures
                the converged solve at 19 ms per frame with 100 components per
                source and 40 ms with 300, against 12 and 42 for the local
                combination, so the comparison is at comparable latency either
                way and this row says what a strictly cheaper NMF delivers.
  nmf capacity  activations fitted against the *true* source magnitude rather than
                the mixture, this basis' ceiling under the mixture phase. The
                analogue of `capacity k=N` in `defmap_local_combination.py`, and
                the line that says whether NMF loses the same way Def-MAP does,
                in selection rather than in capacity.

    MUSDB_ROOT=/path/to/musdb18 python experiments/nmf_baseline.py
    DICT_SIZES=50 N_TEST=2 TEST_SECONDS=5 MUSDB_ROOT=... python experiments/nmf_baseline.py
    python experiments/nmf_baseline.py --check          # self-check, no corpus
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
from sklearn.decomposition import NMF, non_negative_factorization

from defmap_oracle_pair import DICT_SIZES  # isort: skip
from defmap_protocol import (  # isort: skip
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

METHOD = "nmf"
# KL on magnitudes, the separation literature's loss, and not the Euclidean
# default: the squared error is dominated by the loud broadband source, which
# hands it the whole mask and costs the quiet source ~20 dB. Measured, not
# assumed, on a four-track smoke run. `mu` is the only solver KL has.
LOSS = {"beta_loss": "kullback-leibler", "solver": "mu"}
MAX_ITER = 300  # what the basis fit and the unconstrained activation solve get
FAST_ITER = int(os.environ.get("NMF_FAST_ITER", "25"))  # the budget-compatible arm
FLOOR = 1e-30


def _basis(loud: ComplexArray, size: int) -> FloatArray:
    """`size` non-negative spectra learned on one source's training magnitudes.

    Not nested across sizes, unlike the Def-MAP dictionary: each size is its own
    fit, so the trend against size carries NMF's own initialization noise. That is
    NMF, not a choice made here.
    """
    model = NMF(
        n_components=size,
        init="nndsvda",  # deterministic, and no zeros for the multiplicative update
        max_iter=MAX_ITER,
        random_state=SEED,
        **LOSS,
    )
    model.fit(np.abs(loud))
    return np.asarray(model.components_, dtype=np.float64)


def _activate(
    magnitude: FloatArray, basis: FloatArray, iters: int = MAX_ITER
) -> FloatArray:
    """Activations of a fixed basis, frames by components.

    `iters` is the whole latency of the deployed baseline: coordinate descent on
    the activations is what runs per frame, and nothing else does.
    """
    weights, _, _ = non_negative_factorization(
        magnitude,
        H=basis,
        n_components=len(basis),
        update_H=False,
        init="custom",
        max_iter=iters,
        random_state=SEED,
        **LOSS,
    )
    return np.asarray(weights, dtype=np.float64)


def _separate(
    mixture: ComplexArray,
    basis1: FloatArray,
    basis2: FloatArray,
    power: float,
    iters: int = MAX_ITER,
) -> list[ComplexArray]:
    """Both sources, by masking the mixture with the two partial reconstructions.

    The mask is what keeps the baseline inside the same class the oracles bound,
    and it makes the two estimates sum back to the mixture exactly whatever the
    activations are, which the self-check asserts.
    """
    weights = _activate(np.abs(mixture), np.concatenate((basis1, basis2)), iters)
    part1 = weights[:, : len(basis1)] @ basis1
    part2 = weights[:, len(basis1) :] @ basis2
    share = part1**power / np.maximum(part1**power + part2**power, FLOOR)
    return [share * mixture, (1.0 - share) * mixture]


def _capacity(
    spectra: list[ComplexArray], mixture: ComplexArray, bases: list[FloatArray]
) -> list[ComplexArray]:
    """What this basis could deliver if the activations were fitted on the truth.

    One fit per source against its own magnitude, so no competition and no cross
    term: the ceiling of the model class, selection set aside, under the mixture
    phase the method never estimates.
    """
    out = []
    for spectrum, basis in zip(spectra, bases, strict=True):
        magnitude = _activate(np.abs(spectrum), basis) @ basis
        phase = mixture / np.maximum(np.abs(mixture), FLOOR)
        out.append(magnitude * phase)
    return out


def _self_check() -> None:
    """Sources on disjoint bands must come back nearly whole, and the two estimates
    must sum to the mixture. The first assertion is what catches a transposed basis,
    which separates badly rather than loudly."""
    rng = np.random.default_rng(SEED)
    bins, frames = 20, 8
    basis1 = np.zeros((2, bins))
    basis2 = np.zeros((2, bins))
    basis1[0, :5], basis1[1, 5:10] = 1.0, 1.0
    basis2[0, 10:15], basis2[1, 15:] = 1.0, 1.0

    gains = np.abs(rng.normal(size=(frames, 4))) + 0.1
    truth1 = gains[:, :2] @ basis1
    truth2 = gains[:, 2:] @ basis2
    phase = np.exp(2j * np.pi * rng.random((frames, bins)))
    mixture = (truth1 + truth2) * phase

    for power in (1.0, 2.0):
        estimate1, estimate2 = _separate(mixture, basis1, basis2, power)
        assert np.allclose(estimate1 + estimate2, mixture)
        error = np.abs(np.abs(estimate1) - truth1).sum() / truth1.sum()
        assert error < 1e-6, f"power {power}: disjoint bands not recovered, {error:.3g}"

    ceiling = _capacity([truth1 * phase, truth2 * phase], mixture, [basis1, basis2])
    assert np.allclose(np.abs(ceiling[0]), truth1, atol=1e-8)
    note("self-check: disjoint bands recovered, estimates sum to the mixture")


def main() -> None:
    _self_check()
    rng = np.random.default_rng(SEED)
    pools, test_items = corpus(rng)
    # the loud half, whole: `dictionary` truncates to `size`, and a size past the
    # end of the permutation is the permutation, so this is every frame the
    # Def-MAP dictionaries could have drawn
    louds = [pool[dictionary(pool, len(pool), rng)] for pool in pools]
    note(f"pool: {[len(p) for p in pools]} frames per source, {[len(p) for p in louds]} loud")

    base = {**config(), "method": METHOD, "max_iter": MAX_ITER}
    done = done_cells()
    bases: dict[int, list[FloatArray]] = {}  # trained on first use, not upfront

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
            if size not in bases:
                started = time.time()
                bases[size] = [_basis(loud, size) for loud in louds]
                note(f"  basis at {size} components: {time.time() - started:.0f}s")
            basis1, basis2 = bases[size]

            started = time.time()
            rules = {
                "nmf": _separate(mixture, basis1, basis2, 2.0),
                "nmf p=1": _separate(mixture, basis1, basis2, 1.0),
                f"nmf i={FAST_ITER}": _separate(mixture, basis1, basis2, 2.0, FAST_ITER),
                "nmf capacity": _capacity(spectra, mixture, [basis1, basis2]),
            }
            elapsed = time.time() - started
            for rule, estimates in rules.items():
                for source, scores in enumerate(score(references, estimates, mixture)):
                    emit({
                        **base, "track": name, "cell": cell, "source": source, "rule": rule,
                        "dict_size": size, "frames": int(len(mixture)),
                        "seconds": elapsed, **scores,
                    })
            note(f"  {cell}: {len(rules)} rules, {elapsed:.0f}s")


if __name__ == "__main__":
    _self_check() if "--check" in sys.argv else main()
