"""Open-Unmix on the same protocol: what a trained deep model delivers here.

`nmf_baseline.py` answers the reviewer who asks for a standard method at
comparable latency. This answers the other one: how far the repaired exemplar
method is from the state of the art of its task, measured on the same tracks, the
same excerpts and the same metric rather than against a number copied from a
paper whose protocol differs in the excerpt length, the reference implementation
of SDR, and the corpus split.

It is not a latency competitor and is not offered as one. Open-Unmix is a
three-layer bidirectional LSTM over the whole excerpt, so it is non-causal by
construction and its cost per frame is not defined: `bench_inference_cost.py`
deliberately has no row for it. What it bounds is quality, and the point of the
comparison is the distance, not a win.

`umxhq` is trained on the MUSDB18 *train* split and every test item here comes
from the test split, so the comparison is clean of leakage, which is exactly why
the pretrained weights are used rather than a fit of our own.

One rule, `umx n=0`: the model's magnitude estimates turned into a ratio mask on
the mixture, no EM refinement. Open-Unmix's own default is one step of
multichannel Wiener EM and `UMX_NITER` still reaches it, but it is not what is
reported, for a reason that is a property of this protocol rather than of the
model. Our references are mono and the model is stereo, so the input is one
channel duplicated, and the 2x2 spatial covariance the EM estimates from two
identical channels is singular: five of the fifty test tracks came back NaN on
both sources, and on the forty-five where it ran it cost a tenth of a dB rather
than buying one. A multichannel filter on a duplicated channel is not a
measurement, so the arm reported is the one that does not need the second
channel to exist.

`umx n=0` is not exactly inside the masking class the five oracles bound, and the
journal's witnesses say so: 3 degrees of median phase deviation and 2 % of gains
above one. The mask is Open-Unmix's, applied in Open-Unmix's own 4096-point STFT,
and what we read is the waveform reanalysed at 1024, where one of our bins spans
several of theirs. So `d(m*)` is a distance to the ceiling of *our* class and not
a fraction of a ceiling the method was held to, which is the honest reading of a
baseline that was never asked to stay in that class.

The vocals target is asked for explicitly and the second source is the
`residual`, which is the vocals/accompaniment split the protocol already uses,
`ceiling_sweep._load_musdb` building its references in that order.

Torch is imported inside the separation call and nowhere else, so `--check` runs
in the repo's own environment. The real run needs a venv with torch in it, which
`experiments/daneel_umx.sh` builds inside the container.

    MUSDB_ROOT=/path/to/musdb18 python experiments/openunmix_baseline.py
    N_TEST=2 TEST_SECONDS=5 MUSDB_ROOT=... python experiments/openunmix_baseline.py
    python experiments/openunmix_baseline.py --check          # self-check, no torch
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

import numpy as np

from ceiling_sweep import TEST_SECONDS, _load_musdb  # isort: skip
from defmap_protocol import (  # isort: skip
    HOP,
    NFFT,
    ComplexArray,
    FloatArray,
    config,
    done_cells,
    emit,
    note,
    oracle_rows,
    score,
    test_spectra,
)
from evaluation import analyze  # isort: skip

METHOD = "openunmix"
MODEL = os.environ.get("UMX_MODEL", "umxhq")
NITERS = tuple(int(n) for n in os.environ.get("UMX_NITER", "0").split(","))


def _collapse(estimate: Any) -> FloatArray:
    """One openunmix output, (1, channels, samples), back to the protocol's mono.

    The model is stereo and our references are mono, so the mono input is upmixed
    to two identical channels on the way in and the two outputs are averaged on
    the way out. Averaging and not taking channel 0: the two differ once the EM
    step runs, and half of the estimate is not the estimate.
    """
    array = np.asarray(estimate, dtype=np.float64)
    return np.asarray(array.reshape(-1, array.shape[-1]).mean(axis=0), dtype=np.float64)


def _separate(signal: FloatArray, rate: int, niter: int) -> list[FloatArray]:
    """Both sources in the time domain, vocals first, from the pretrained model."""
    import torch  # type: ignore[import-not-found]
    from openunmix import predict  # type: ignore[import-not-found]

    audio = torch.from_numpy(np.asarray(signal, dtype=np.float32)[None, :])
    estimates = predict.separate(
        audio,
        rate=rate,
        model_str_or_path=MODEL,
        targets=["vocals"],
        residual=True,
        niter=niter,
        device="cpu",
    )
    return [_collapse(estimates[key].detach().cpu()) for key in ("vocals", "residual")]


def _spectra(signals: list[FloatArray], mixture: ComplexArray) -> list[ComplexArray]:
    """Time-domain estimates into the shape `score` measures.

    The deep model returns waveforms where every other method returns
    spectrograms, so the analysis happens here rather than in the metric, and the
    metric path stays the one every other probe uses: analysis, mask or not, WOLA
    resynthesis, time-domain SDR.
    """
    spectra = [np.asarray(analyze(signal, NFFT, HOP), dtype=np.complex128) for signal in signals]
    for spectrum in spectra:
        assert spectrum.shape == mixture.shape, f"{spectrum.shape} against {mixture.shape}"
    return spectra


def _self_check() -> None:
    """The plumbing, without the model: channel collapse, then the metric path run
    on the truth. An estimate that is the reference itself must score far above any
    separator, and it does not if the analysis is misaligned with the mixture's."""
    rng = np.random.default_rng(0)
    stacked = rng.normal(size=(1, 2, 64))
    assert np.allclose(_collapse(stacked), stacked[0].mean(axis=0)), "channel collapse"

    time_axis = np.arange(8 * NFFT) / 16000.0
    references = [
        np.sin(2 * np.pi * 440 * time_axis),
        np.sin(2 * np.pi * 660 * time_axis + 1.1) + 0.3 * rng.normal(size=len(time_axis)),
    ]
    _, mixture = test_spectra(references)
    rows = score(references, _spectra(references, mixture), mixture)
    for row in rows:
        assert row["sdr"] > 60.0, f"metric path broken, truth scores {row['sdr']:.1f} dB"
    note("self-check: channels averaged, the truth round-trips through the metric")


def main() -> None:
    _self_check()
    # only the test half of the corpus: no dictionary is drawn and no basis is
    # trained, and `_load_musdb` selects its test items without consuming the rng,
    # so these are the same excerpts the other probes separate, to the byte
    _, test_items = _load_musdb()
    base = {**config(), "method": METHOD, "model": MODEL, "dict_size": 0}
    done = done_cells()

    for name, references in test_items:
        if (name, "oracle") not in done:
            for row in oracle_rows(name, references):
                emit(row)
        if (name, "umx") in done:
            note(f"track {name!r} already in the journal, skipped")
            continue
        _, mixture = test_spectra(references)
        rate = round(len(references[0]) / TEST_SECONDS)
        note(f"track {name!r}: {len(mixture)} frames at {rate} Hz")

        for niter in NITERS:
            started = time.time()
            estimates = _separate(references[0] + references[1], rate, niter)
            elapsed = time.time() - started
            rows = score(references, _spectra(estimates, mixture), mixture)
            for source, scores in enumerate(rows):
                emit({
                    **base, "track": name, "cell": "umx", "source": source,
                    "rule": f"umx n={niter}", "niter": niter,
                    "frames": int(len(mixture)), "seconds": elapsed, **scores,
                })
            # both SDRs on one line: the vocals/residual assignment cannot be
            # asserted without the model, and a swap shows up here as the voice
            # scoring where the accompaniment should
            note(f"  n={niter}: {rows[0]['sdr']:+.1f} / {rows[1]['sdr']:+.1f} dB, {elapsed:.0f}s")


if __name__ == "__main__":
    _self_check() if "--check" in sys.argv else main()
