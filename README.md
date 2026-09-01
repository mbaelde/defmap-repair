# defmap-repair

Companion code for a journal article in preparation on Def-MAP, the real-time source separation method of chapter 3 of Baelde (2019). Def-MAP separates a mixture by picking one learned spectrum per source from a dictionary and deforming both optimally to explain the observed frame. As published it separates poorly, +5.7 dB over the mixture on vocals and +0.4 dB on the accompaniment against a 17.0 dB ceiling, and costs 1037 ms per frame at 100 atoms per source against a 23 ms real-time budget, so this repository first diagnoses why and then repairs it.

The baselines are imported from [`gasm`](https://github.com/mbaelde/generative-audio-source-models), the thesis code itself, so that what is being criticised is the reference implementation and not a paraphrase of it. Section and figure numbers below refer to the article; each experiment section states which of its claims that experiment produces.

## From the article to the code

| Article | Script | Question it answers |
| --- | --- | --- |
| §III, §V-B, Fig. 2 | `defmap_oracle_pair.py` | Is the failure the dictionary or the selection criterion? |
| §IV-B, Fig. 1 | `defmap_phase_ramp.py` | Does constraining the deformation to a delay make the criterion informative? |
| §IV-D, §V-C, Fig. 3 | `defmap_phase_ramp.py` | What does the rigid class deliver, and what does partial reabsorption add? |
| §IV-C, §V-D, Fig. 4 | `defmap_local_combination.py` | Does a richer model class reach the output, or is the gain absorbed by selection? |
| §IV-E | `defmap_local_combination.py` (rules `xtalk`, `greedy`) | Can a criterion aimed at the source rather than the mixture dent the selection lock? |
| §V-E, Table I | `nmf_baseline.py` | Where does the repair sit against a learned baseline at comparable latency? |
| §V-E, Table I | `openunmix_baseline.py` | How far is the state of the art of the task, on this protocol? |
| §IV-F, §V-G, Table II | `bench_inference_cost.py` | Is the repaired rule inside the real-time budget? |
| all figures | `figures.py` | Draws the article's figures from the same journals the tables are read from. |
| §V-A | `defmap_protocol.py`, `evaluation.py` | The protocol and the metric every row above is measured with. |

Figure filenames do not follow the article's numbering, because the alignment figure was written last and placed first:

| Article | File |
| --- | --- |
| Fig. 1, the alignment step | `figures/fig2_ramp.pdf` |
| Fig. 2, the diagnosis | `figures/fig1_diagnostic.pdf` |
| Fig. 3, the selection gap | `figures/fig3_constraint.pdf` |
| Fig. 4, quality against capacity | `figures/fig4_selection.pdf` |
| not in the article | `figures/fig5_cost.pdf`, superseded by Table II |

## Layout

Measurement layer, shared verbatim with the companion article on the ceilings of time-frequency masking, whose own experiments live in [`masking-ceiling`](https://github.com/mbaelde/masking-ceiling), so that a row of this repository and a row of that one are the same measurement:

- `experiments/evaluation.py`, WOLA resynthesis, time-domain SDR and SI-SDR, the five oracles of the masking class.
- `experiments/ceiling_sweep.py`, the corpus and the excerpt selection.
- `experiments/report_ceiling.py`, journal loading and the silence guard.
- `experiments/defmap_protocol.py`, this article's own layer on top of those: one JSONL row per (track, dictionary size, rule, source), and a resumable journal.

Experiments, in the order the article reads them: `defmap_oracle_pair.py`, `defmap_phase_ramp.py`, `defmap_local_combination.py`. Baselines: `nmf_baseline.py`, `openunmix_baseline.py`. Cost: `bench_inference_cost.py`. Reporting: `report_defmap.py`, `figures.py`. Runners: `run.sh`, `daneel.sh`, `daneel_umx.sh`. Corpus tooling: `locate_clips.py`, `locate_clips.sh`.

The modules import each other flat, which is why they sit in one directory: a script is run with `experiments/` as `sys.path[0]`, and `run.sh` is what groups them into a plan. Every script carries a `_self_check()` that runs before the corpus is touched, and takes its parameters from environment variables.

## Protocol

Every number in the article comes off the whole fifty-track MUSDB18 test split at 25 training tracks, five-second excerpts, 1024-point transform, three dictionary sizes of 50, 100 and 300 atoms per source. Figures are gains over the mixture on the vocals with 300 atoms unless said otherwise, paired track by track, and read against `best_real`, the exact ceiling of the masking class, at +16.97 dB on vocals and +11.64 dB on the accompaniment. The ideal ratio mask sits about 3 dB under that ceiling, so a deficit phrased against the mask alone understates it by that margin.

## The experiments

### `defmap_oracle_pair.py`, the diagnosis

Asks whether Def-MAP fails because its dictionary is too poor or because its selection criterion is wrong, by putting an oracle that picks the pair minimising the true error inside the same candidate pool.

- The oracle reaches +13.41 dB where Def-MAP's own criterion reaches +5.71 dB, a gap of 6.23, 6.57 and 7.70 dB at the three dictionary sizes.
- Def-MAP's criterion gets *worse* as the dictionary grows, +5.84 then +6.07 then +5.71 dB, while its own oracle rises monotonically.
- A phase-blind rule on magnitude additivity, run over the same pairs, leaves a gap of 6.11 to 6.43 dB, so discarding the dictionary's phase costs nothing the criterion has.
- The gap is symmetric between the two sources to the decimal, which is the single error signal the degeneracy predicts.

The criterion is at fault: fitting 1026 free real parameters per candidate lets any pair explain any mixture, and the closed forms show the imaginary part collapsing to a Wiener mask and a silent pair scoring a perfect zero.

### `defmap_phase_ramp.py`, the rigid class

Constrains the deformation to what a sub-sample delay can produce, `c -> g * c * exp(-2i pi f tau)`, three real parameters instead of 1026, with the delay read off a cross-correlation peak refined to sub-sample resolution and the complex gains from a 2x2 Hermitian system in closed form.

- The oracle-to-criterion gap falls to 2.13, 2.29 and 2.80 dB on vocals and to 0.48, 0.55 and 0.80 dB on the accompaniment, two thirds to nine tenths of the loss, and the criterion improves with dictionary size again.
- The symmetry between sources breaks by a factor of three and a half to four and a half, which is what a class that does not interpolate is supposed to show.
- What survives grows with the dictionary, 0.67 dB on vocals between 50 and 300 atoms, while the oracle of the same class improves over the same range: the criterion now ranks, and what remains is that it ranks against the wrong target.

An `alpha` dial then lets the selected pair reabsorb a fraction of the unexplained residual, selection always staying on the rigid `alpha = 0` residual.

### `defmap_local_combination.py`, the rich class and the selection lock

Replaces the single atom per source by a local combination of `k` aligned atoms, fitted jointly by one complex least squares of dimension `k1 + k2`. `k = 1` reproduces the previous rule exactly, which the self-check asserts.

- Capacity climbs with `k`, from +8.56 dB to +12.99 dB at `k = 16`, reaching the free per-bin deformation oracle to within 0.15, 0.03 and 0.42 dB with 16 gains and 16 delays instead of 1026 free reals.
- Delivered quality does not follow: +5.81, +6.02, +6.11, +6.05 then +5.57 dB for `k = 1, 2, 4, 8, 16`, peaking at `k = 4` and falling back at `k = 16`.
- The gap between capacity and delivered quality runs 2.75, 3.31, 4.12, 5.34 and 7.42 dB on vocals over that sweep and 0.85 to 4.53 dB on the accompaniment, and grows on the other axis too, from 6.36 to 7.42 dB at `k = 16` between 50 and 300 atoms.

The entire benefit of a larger model class is absorbed by selection, hence the operating point of Table I, `k = 2` or `4` rather than the widest measured.

The same script carries experiment 2.5, two selection rules aimed at that 7.42 dB, reported in one sentence of Section IV-E. Both pick exactly `k` atoms per source, so they live in the model class of `local k=N` and are read against its `capacity k=N` ceiling.

- `xtalk k=N l=L` discounts each atom's score by its own ambiguity, the best-delay coherence with the *other* source's dictionary, computed once per dictionary pair rather than per frame so that selection stays linear in the dictionary. `l = 0` is the control point and must reproduce the unrepaired rule exactly.
- `greedy k=N` selects jointly by deflation under a per-source quota, which attacks the cross term `<c, x> = <c, s1> + <c, s2>` head on: once one source's content is in the model, the residual stops paying for atoms of the other that explain it.
- At `k = 16` and 300 atoms they raise the vocals from +5.57 dB over the mixture to +6.20 dB with `xtalk l=1` and +6.54 dB with `greedy`, against a capacity of +12.99 dB, paired standard errors of 0.04 and 0.09 dB. That is 8 to 13 per cent of the lock, quantified and dented rather than closed.

Both rules are added unconditionally to the default sweep, so the article's fifty-track campaign measured them on the same tracks, dictionaries and frames as `local` and `capacity`.

## The baselines

### `nmf_baseline.py`, the learned baseline

Supervised NMF, one basis per source learned on exactly the frames the Def-MAP dictionary is drawn from, then activations of the concatenated basis against the mixture magnitude with the basis held fixed, KL loss on magnitudes as the separation literature has it. The two partial reconstructions define a mask, so the baseline lives inside the same class the oracles bound.

Four rules per size: the Wiener form `nmf`, the magnitude ratio `nmf p=1`, a strictly cheaper `nmf i=25` that fixes the latency instead of letting it follow convergence, and `nmf capacity`, activations fitted against the true source magnitude, which is the analogue of `capacity k=N` and says whether NMF loses where Def-MAP does. Activations are solved for the whole test spectrogram in one call, which is not lookahead: with the basis fixed the objective separates over frames.

It reaches +7.72 dB converged and +7.82 dB at twenty-five updates on vocals, so at the operating point it is ahead of the repaired rule on both axes, quality and latency. What the exemplar route buys is the diagnosis and a model carrying its own phase, not the level, and this README says so for the same reason the article does.

### `openunmix_baseline.py`, the trained baseline

Pretrained `umxhq`, on the same tracks, the same excerpts and the same metric, so the distance to the state of the art of the task is measured rather than copied from a paper with another protocol. One rule, `umx n=0`, the model's magnitudes turned into a ratio mask. On the whole fifty-track split it reaches +11.6 dB over the mixture on the voice, 5.4 dB under `best_real` and 2.3 dB under the IRM, in five and a half minutes on one core.

Three things it is not, all of them stated in the article too. It is not a latency competitor and `bench_inference_cost.py` deliberately has no row for it, a bidirectional LSTM over the whole excerpt having no per-frame cost to report. Its default one step of multichannel Wiener EM is reachable through `UMX_NITER` and is not reported: our references are mono, so the stereo model gets one channel duplicated, the 2x2 spatial covariance it estimates from two identical channels is singular, five of the fifty test tracks came back NaN, and on the other forty-five the step cost a tenth of a dB rather than buying one. And it does not sit exactly inside the class the oracles bound, the journal's witnesses saying so with 3 degrees of median phase deviation and 2 % of gains above one, because the mask is applied in Open-Unmix's own 4096-point STFT and what we read is the waveform reanalysed at 1024. So `d(m*)` here is a distance to the ceiling of our class rather than a fraction of a ceiling the method was held to. The weights are trained on the MUSDB18 train split and every test item comes from the test split, which is why they are used as they are rather than refitted.

## Cost

`bench_inference_cost.py` times the deployed rules alone on random spectra, with no corpus, since cost depends only on shapes. Table II, best-of-five on one core of an AMD Ryzen 7 7435HS:

| | 100 atoms | 300 atoms |
| --- | --- | --- |
| Def-MAP, as published | 1037.2 ms | 9513.4 ms |
| pair rule | 12.2 ms | 65.7 ms |
| local combination, `k = 4` | 10.7 ms | 44.7 ms |
| local combination, `k = 16` | 11.6 ms | 45.3 ms |
| budget, one 1024-sample frame | 23 ms | 23 ms |

That is a factor of 47 to 806 on the local combination and 45 to 328 on the pair rule. The local combination is cheaper than the pair rule it replaces, the quadratic `K1 x K2` search being gone, and `k` barely registers, flat to within a millisecond over the whole range. The frame duration itself is met up to 100 atoms and exceeded by a factor of 1.9 at 300, which is the size carrying the best quality figures, so real-time feasibility holds at the smaller sizes only. Experiment 2.5's `greedy` costs two and a half to four and a half times the local combination, refitting at each of its 2k steps; the NMF baseline costs 17.9 ms converged and 3.2 ms at twenty-five updates, so the quality comparison against it is a comparison at comparable latency. Absolute values move with the machine and its load, the ratios between rules do not, which is what the script is for.

## Reporting

`report_defmap.py` is what the article's tables are read from. It pairs every rule against the oracles of the same track, drops tracks with a silent stem and says which, and reports both the gain over the mixture and the distance to the masking ceiling. Journals merge by track name, so a sharded run reads as one campaign.

`figures.py` draws the figures from those same journals, through `report_defmap.rows`, so a figure and a table cannot disagree. The two ceiling lines are derived from a rule's own columns rather than re-averaged, `d(mix) - d(m*)` being the paired mean of `sdr_best_real - sdr_mixture`, which keeps them on exactly the tracks that rule was measured on. Two figures are exceptions by design: the cost figure calls `bench_inference_cost.measure` instead of a journal, latency having to be measured on the machine that draws it, and the alignment figure runs the deployed `_align` on a synthetic atom so that it shows the code's behaviour rather than a drawing of the idea.

## Running

```
uv run python experiments/defmap_oracle_pair.py --check          # self-check, no corpus
uv run python experiments/bench_inference_cost.py
MUSDB_ROOT=/path/to/musdb18 uv run python experiments/defmap_local_combination.py
K_VALUES=1,4 DICT_SIZES=50 TEST_SECONDS=5 N_TEST=2 MUSDB_ROOT=... uv run python experiments/defmap_local_combination.py
sh experiments/run.sh defmap                                     # the four probes, resumable
LOG=/scratch/logs/lot3 sh experiments/daneel_umx.sh              # the fifth, in its own venv
SLICES="0:20 20:30 30:40 40:50" LOG=/scratch/logs/lot3 sh experiments/daneel.sh defmap   # the same plan on four cores
uv run python experiments/report_defmap.py scratch/logs/defmap/*.jsonl
uv run python experiments/report_defmap.py scratch/logs/lot3/*/defmap/*.jsonl   # sharded run, journals merge by track
uv run --group figures python experiments/figures.py scratch/logs/defmap/*.jsonl
FIG_DIR=figures/pilot FIG_FORMAT=png uv run --group figures python experiments/figures.py scratch/logs/pilot/*.jsonl
```

`sh experiments/run.sh defmap` is the form meant for a real run: one cell per script, each skipping the (track, dictionary) cells already in its journal, so a killed run is relaunched with the same command. The Open-Unmix probe is a plan of its own and a container of its own because torch must not become a dependency of this repository: `uv run python` resyncs the project environment on every call, so a torch installed into it would be removed by the next Def-MAP cell. `experiments/daneel_umx.sh` builds a second venv under `/scratch` and passes it as `PYTHON`, and its `SRC` knob lets it run against a `git worktree` of the branch while a Def-MAP plan is still reading the main checkout.

Knobs, all read from the environment:

- `N_TEST` and `TEST_SECONDS` are the cost knobs. `N_TEST` defaults to the whole fifty-track MUSDB18 test split so that no cross-probe reading has to go through an intersection. `TEST_SECONDS` is also the memory knob, `defmap_local_combination.py` holding one spectrogram per rule, about 420 MB at five seconds and 2.5 GB at thirty.
- `TEST_SLICE="a:b"` is the only way to make a plan use more than one core: a cell is one Python process, measured at 100 % of one, so the plan is bound by serialization rather than by compute. `SLICES` on `daneel.sh` runs one container per slice with a journal directory of its own. The slice is read inside the `N_TEST` selection and the training pool is fixed by `N_TRAIN` and `SEED`, so what a shard measures on a track is what the whole plan would have measured on it.
- `FIG_DIR` and `FIG_FORMAT` are where the figures land and in what format, PDF for the paper and PNG when one has to be looked at. `matplotlib` sits in its own `figures` dependency group rather than in `dev`, which `uv run` installs by default: a figure is drawn once on a laptop and every experiment container would otherwise download it for nothing.
- `musdb` reads stems through `stempeg`, which hard-fails without `ffmpeg` on the path.

### Where in the track the excerpts are cut

The article's numbers are measured inside the `musdb18-7s` clips, which are not the heads of their tracks: they start between 22.2 s and 298.2 s, median 135.2 s. `TEST_OFFSETS` counts from the start of the full track, so an offset sweep on full tracks does not overlap the measured excerpt on any of the fifty. `locate_clips.sh` decodes each (clip, full track) pair with the host's ffmpeg, the `uv` container having none, and `locate_clips.py` locates each clip in its track by normalised cross-correlation computed by FFT, writing one TSV line per track. `TEST_ANCHORS` points at that TSV, and offsets then count from each track's own anchor, so the `@0` column of an anchored run reproduces by construction what the article measures and its neighbours say whether the numbers survive a slide. Run `locate_clips.py` with no argument for its self-check; a correlation below 0.9 is flagged `SUSPECT`.

## Scope and limits

These experiments are diagnostics, sized to tell one hypothesis from another. One corpus and one metric, with the pairing over fifty tracks as the only dispersion figure: the repaired rule's paired gain over the original method is 0.9 to 1.2 dB with a between-track deviation of 1.3 to 1.4 dB, hence a paired standard error near 0.19 dB. Three limits are stated rather than hidden.

- The repaired rule stays 10.0 dB under `best_real` on vocals, so this is not state-of-the-art quality.
- On the accompaniment the exemplar model at `k = 1` sits *below* the mixture taken as its own estimate, at -1.53, -1.17 and -0.91 dB, and the positive figures reported on that source at `alpha = 0.75` come from the reabsorption term, which is a soft mask on the mixture rather than the model. Exemplar-based separation is claimed on vocals alone.
- One known bias of the repaired rule is documented in `_align` and left in place: the delay is read off the cross-correlation of two real frames, so the complex gain fitted after it displaces that correlation's peak by about its phase over the atom's centre frequency, half a sample at 0.3 rad. Decoupling it means refining on the correlation's analytic envelope, which is a change of rule rather than of implementation, so it is measured as one or not at all.
