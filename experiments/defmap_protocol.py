"""The measurement protocol article 2 borrows from article 1, in one place.

The four Def-MAP experiments were written before `evaluation.py` existed and
measured themselves: a spectrum-domain SDR stand-in, an inline IRM2 as ceiling,
one test track, no journal. None of them imports anything from the article 1
harness, which by now carries every piece they were missing. This module is the
adapter, so that both articles report the same quantity on the same corpus and a
number can move between them.

What changes, and why each change matters more than it looks:

  metric      `sdr` / `si_sdr` are time-domain, after a WOLA round trip that is
              exact to 100+ dB. The spectral stand-in ignored the overlap-add
              projection, which is not a constant offset: a per-frame masked
              spectrogram is inconsistent and the projection removes residual,
              so the stand-in *understates* every masked estimate by a variable
              amount. `TRIM = NFFT` is not optional, the sum of squared windows
              vanishes over the first and last frame and an estimate divided by
              it explodes there (article 1 read its IRM oracle at -2.81 dB, below
              the untreated mixture, before the trim was put in).
  ceiling     the five oracles of `oracle_spectra`, `best_real` among them. The
              inline IRM2 these scripts used is *not* the ceiling of the masking
              class: `m*` sits ~6 dB above it. Every article 2 claim phrased
              against IRM2 (C1's "83 % of the ceiling", the 6.8 dB reserve)
              understates its distance to what a mask can do by about that much,
              and must be restated against `best_real`.
  corpus      `_load_musdb` and `_training_frames`, i.e. `N_TEST` held-out tracks
              instead of `mus_test.tracks[0]`, and a dictionary pool drawn under
              a per-track quota so peak memory is one track's spectra.
  resume      one journal line per (track, cell), the cell being reread and
              skipped on relaunch, exactly like `ceiling_sweep.py`. The pair
              search at 300 atoms is hours per cell on the full test split, so
              this is what makes the protocol runnable at all.

`STRIDE` is gone from the metric path and there is no replacement. Subsampling
the test frames by s leaves a spectrogram that is not the STFT of anything: Hann
at hop s*512 does not satisfy COLA, so it cannot be inverted and no time-domain
metric can be read from it. `N_TEST` and `TEST_SECONDS` are the cost knobs now.

Nothing here is imported *into* `ceiling_sweep.py`: article 1's held-out arm is
paused mid-`RESUME_FROM`, and a refactor of the script it resumes into would
invalidate the journal it resumes from. The oracle block below is therefore a
deliberate ten-line twin of the one in `main()` there, to be folded back into a
shared helper once that arm has landed.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ceiling_sweep import (  # article 1's corpus, verbatim, so the two are comparable
    HOP,
    MAX_TRAIN_FRAMES,
    N_TEST,
    N_TRAIN,
    NFFT,
    REGIME,
    SEED,
    TEST_SECONDS,
    _load_musdb,
    _training_frames,
)
from evaluation import (
    ORACLES,
    analyze,
    class_witnesses,
    oracle_spectra,
    sdr,
    si_sdr,
    spectral_ceiling,
    synthesize,
)

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

N_FREQ_BINS = NFFT // 2 + 1
RESUME_FROM = os.environ.get("RESUME_FROM", "")
_SILENCE_FLOOR = 1e-3  # of the mixture's peak magnitude, below which a gain is noise

__all__ = [
    "HOP",
    "NFFT",
    "N_FREQ_BINS",
    "SEED",
    "ComplexArray",
    "FloatArray",
    "config",
    "corpus",
    "dictionary",
    "done_cells",
    "emit",
    "note",
    "oracle_rows",
    "score",
    "test_spectra",
]


def config() -> dict[str, Any]:
    """The knobs every row of a journal carries, so a row is readable alone."""
    return {
        "corpus": "musdb",
        "regime": REGIME,
        "nfft": NFFT,
        "hop": HOP,
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "test_seconds": TEST_SECONDS,
        "max_train_frames": MAX_TRAIN_FRAMES,
        "seed": SEED,
    }


def corpus(
    rng: np.random.Generator,
) -> tuple[list[ComplexArray], list[tuple[str, list[FloatArray]]]]:
    """The candidate pool per source, and the held-out items to separate.

    The pool is what the dictionary is drawn from: training frames of one source,
    pooled over `N_TRAIN` tracks under a `MAX_TRAIN_FRAMES // N_TRAIN` quota each.
    Test items are time-domain references, because the metric is time-domain and
    the mixture must be the sum of exactly those references.
    """
    train_loaders, test_items = _load_musdb()
    return [_training_frames(loaders, rng) for loaders in train_loaders], test_items


def dictionary(spectra: ComplexArray, size: int, rng: np.random.Generator) -> NDArray[np.intp]:
    """Indices of `size` frames drawn above the pool's median frame energy.

    MUSDB vocal stems are silent for much of a track, and a silent pair scores a
    *zero* Def-MAP loss (`defmap_oracle_pair._chunk_terms`), so an unfiltered draw
    would measure the criterion's silence bias instead of the dictionary. The draw
    is a permutation of the loud half, so the sizes are nested and the trend
    against dictionary size is monotone by construction rather than by luck.
    """
    energy = (np.abs(spectra) ** 2).sum(axis=-1)
    loud = np.flatnonzero(energy >= np.median(energy))
    return rng.permutation(loud)[:size]


def test_spectra(references: list[FloatArray]) -> tuple[list[ComplexArray], ComplexArray]:
    """Analysis spectra of the references and their mixture.

    The mixture is the sum of the source *spectra* rather than the spectrum of
    the mixed signal. The two agree to numerical precision, the STFT being linear,
    and summing the spectra is what guarantees it exactly, which the oracles need:
    a mask built from sources that do not sum to the mixture is not an oracle.
    """
    spectra = [analyze(reference, NFFT, HOP) for reference in references]
    return spectra, np.asarray(np.sum(spectra, axis=0), dtype=np.complex128)


def score(
    references: list[FloatArray], estimates: list[ComplexArray], mixture: ComplexArray
) -> list[dict[str, float]]:
    """Per source: SDR, SI-SDR, and the three witnesses that it left the mask class.

    `estimates` are full spectrograms of the same shape as the mixture, one per
    source. The shape assertion is the guard against a subsampled estimate being
    synthesized as if it were a spectrogram, which reads as a plausible SDR
    rather than as an error.
    """
    floor = _SILENCE_FLOOR * float(np.abs(mixture).max())
    rows: list[dict[str, float]] = []
    for reference, estimate in zip(references, estimates, strict=True):
        assert estimate.shape == mixture.shape, f"{estimate.shape} against {mixture.shape}"
        signal = synthesize(estimate, NFFT, HOP)
        row = {"sdr": sdr(reference, signal, NFFT), "si_sdr": si_sdr(reference, signal, NFFT)}
        # A NaN is a numerical failure of the rule and never a result: written to
        # the journal it poisons the mean of its whole cell silently, which is how
        # a table gets published with a hole in it. `-inf` is left alone, being the
        # honest score of a rule that returned an empty estimate.
        assert not np.isnan([row["sdr"], row["si_sdr"]]).any(), f"NaN score: {row}"
        row.update(class_witnesses(estimate, mixture, floor))
        rows.append(row)
    return rows


def oracle_rows(name: str, references: list[FloatArray]) -> list[dict[str, Any]]:
    """The reference lines of one test item: five oracles and the analytic ceiling.

    Emitted once per item and independent of any method, so the reporter pairs a
    method row to *its own* track's ceiling. `sdr_mixture` doubles as the silence
    detector: it is the A/B energy ratio in dB, and an excerpt where one stem is
    inaudible has an infinite ceiling that would swamp any mean it enters.
    """
    spectra, _ = test_spectra(references)
    rows: list[dict[str, Any]] = []
    for index, reference in enumerate(references):
        estimates = oracle_spectra(spectra, index)
        row: dict[str, Any] = {
            **config(),
            "track": name,
            "cell": "oracle",
            "source": index,
            "method": "oracle",
            "spectral_ceiling": spectral_ceiling(spectra, index),
        }
        for oracle in ORACLES:
            signal = synthesize(estimates[oracle], NFFT, HOP)
            row[f"sdr_{oracle}"] = sdr(reference, signal, NFFT)
            row[f"si_sdr_{oracle}"] = si_sdr(reference, signal, NFFT)
        rows.append(row)
    return rows


def emit(row: dict[str, Any]) -> None:
    """One JSON object per line on stdout, so a log is directly a dataframe."""
    print(json.dumps(row), flush=True)


def note(message: str) -> None:
    """Progress goes to stderr: stdout is the journal and must stay parseable."""
    print(message, file=sys.stderr, flush=True)


def done_cells(path: str = "") -> set[tuple[str, str]]:
    """The (track, cell) pairs an earlier run of this same probe already wrote.

    Rows are appended, so a relaunch reads its own output back and skips what is
    there. A half-written last line is the normal state of a killed run, so a row
    that does not parse, or that lacks the two keys, never counts as done and is
    simply recomputed.
    """
    path = path or RESUME_FROM
    if not path or not os.path.exists(path):
        return set()
    done: set[tuple[str, str]] = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "track" in row and "cell" in row:
                done.add((row["track"], row["cell"]))
    return done


def demo() -> None:
    """The metric path checks itself against the oracle rows it will be compared to.

    If `score` and `oracle_rows` ever disagree on the same spectrogram, every
    `d(m*)` the reporter prints is the difference between two conventions rather
    than between two estimators, and it would read as a perfectly plausible gap.
    """
    import tempfile

    rng = np.random.default_rng(0)
    time = np.arange(8 * NFFT) / 16000.0
    references = [
        np.sin(2 * np.pi * 440 * time) * (1 + 0.5 * np.sin(2 * np.pi * 3 * time)),
        np.sin(2 * np.pi * 660 * time + 1.1) + 0.3 * rng.normal(size=len(time)),
    ]
    spectra, mixture = test_spectra(references)
    assert np.allclose(mixture, analyze(references[0] + references[1], NFFT, HOP)), "STFT linearity"

    (row0, row1) = oracle_rows("probe", references)
    for index, row in enumerate((row0, row1)):
        estimates = [oracle_spectra(spectra, i)["best_real"] for i in range(2)]
        measured = score(references, estimates, mixture)[index]
        assert abs(measured["sdr"] - row["sdr_best_real"]) < 1e-9, "two SDR conventions"
        # the ceiling is a ceiling: m* is above every other oracle of the class
        for oracle in ("mixture", "irm", "wiener", "clipped"):
            assert row["sdr_best_real"] >= row[f"sdr_{oracle}"] - 1e-9, oracle
        assert row["sdr_best_real"] > row["spectral_ceiling"] - 0.2, "round trip broken"

    # a mask cannot rotate phase, so the witnesses must read zero on one
    masked = [oracle_spectra(spectra, i)["clipped"] for i in range(2)]
    witnesses = score(references, masked, mixture)[0]
    assert witnesses["phase_median_deg"] < 1e-9, "a mask cannot rotate phase"
    assert witnesses["gain_above_one"] == 0.0

    # the shape guard: a subsampled estimate must raise rather than be synthesized
    try:
        score(references, [spectra[0][::4], spectra[1][::4]], mixture)
    except AssertionError:
        pass
    else:  # pragma: no cover
        raise AssertionError("a subsampled estimate was silently synthesized")

    with tempfile.TemporaryDirectory() as directory:
        log = os.path.join(directory, "probe.jsonl")
        with open(log, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"track": "a", "cell": "oracle"}) + "\n")
            handle.write(json.dumps({"track": "a", "cell": "d100"}) + "\n")
            handle.write(json.dumps({"track": "a"}) + "\n")  # no cell, not a done marker
            handle.write('{"track": "b", "ce')  # killed mid-write
        assert done_cells(log) == {("a", "oracle"), ("a", "d100")}, done_cells(log)
    assert done_cells("") == set()

    print("self-check: metric and oracle rows agree, m* above every mask,")
    print("            subsampled estimates rejected, half-written rows not done")


if __name__ == "__main__":
    demo()
