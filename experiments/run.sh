#!/bin/sh
# Every cell of the Def-MAP plan, resumable, one plan per invocation.
#
#   sh experiments/run.sh defmap   the Def-MAP probes and the NMF baseline, hours per cell
#   sh experiments/run.sh umx      the Open-Unmix baseline, needs PYTHON with torch
#
# The ceiling experiments of the companion article live in their own
# repository, https://github.com/mbaelde/masking-ceiling, and are driven by
# the same runner there.
#
# Resumption is the point. A killed run is relaunched with the same command: a
# probe that finished is skipped on its .done marker, a probe that was cut open
# reads its own log back and recomputes only the (K, track) cells missing from
# it, and the EM fit behind those cells restarts from its last checkpoint rather
# than from scratch. Nothing is recomputed except what was actually lost.
#
set -u
# /w is where daneel.sh mounts the repo; a run outside the container gives its own
# WORKDIR, its own PYTHON, and its own LOG and CHECKPOINT_DIR
cd "${WORKDIR:-/w}"
PYTHON=${PYTHON:-uv run python}
PLAN=${1:-defmap}
LOG=${LOG:-/scratch/logs/defmap}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/scratch/checkpoints}
export CHECKPOINT_DIR
# Which script the plan drives. A probe that only re-reads the fits shares the
# whole resume protocol below and differs from the sweep in its entry point
# alone, so the entry point is the variable rather than the function.
SCRIPT=${SCRIPT:-experiments/defmap_oracle_pair.py}
mkdir -p "$LOG" "$CHECKPOINT_DIR"

run() {
    name=$1
    shift
    if [ -f "$LOG/$name.done" ]; then
        echo "=== $name already complete, skipped ===" >&2
        return
    fi
    echo "=== $name : $* ===" >&2
    date >&2
    # append, and hand the log back as RESUME_FROM so finished cells are not redone
    # shellcheck disable=SC2086
    RESUME_FROM="$LOG/$name.jsonl" \
        env "$@" $PYTHON "$SCRIPT" \
        >>"$LOG/$name.jsonl" 2>>"$LOG/$name.err" || {
        echo "=== $name interrupted, relaunch the same command to resume ===" >&2
        exit 1
    }
    touch "$LOG/$name.done"
    echo "=== $name complete, $(wc -l <"$LOG/$name.jsonl") rows ===" >&2
}

# The three Def-MAP probes, on the corpus and the metric of `defmap_protocol`, so
# a row here is comparable to a row of the companion article rather than to a
# spectral stand-in. One cell per script and one script per SCRIPT assignment, the resume
# protocol above being the same: a cell reads its own journal back and skips the
# (track, dictionary) cells already in it.
#
# The environment is fixed rather than defaulted. These cells are read against
# each other -- 2.1 against 2.0 to price the delay constraint, 2.3 against 2.1 to
# price the local combination -- and a knob inherited from the caller would make
# two of the three rows incomparable without saying so. TEST_SECONDS matters
# twice over: local_combination holds one spectrogram per rule in memory, ~420 MB
# at five seconds with experiment 2.5's families and ~2.5 GB at thirty.
#
# The one knob left to the caller is the number of test tracks, so a smoke run
# stays cheap. The publishable lot takes the default, the whole MUSDB18 test
# split: the pilot ran the quadratic probe on five tracks and the two linear ones
# on ten, which forced every cross-probe reading through the five-track
# intersection. Fifty everywhere costs the quadratic probe most of the budget and
# buys paired comparisons that need no intersection.
plan_defmap() {
    # own directory: the reporters glob *.jsonl, and $LOG is whatever arm the
    # caller configured
    LOG="$LOG/defmap"
    mkdir -p "$LOG"
    tracks=${N_TEST:-50}
    base="CORPUS=musdb REGIME=unseen NFFT=1024 N_TRAIN=25 MAX_TRAIN_FRAMES=20000"
    base="$base TEST_SECONDS=5 SEED=0 N_TEST=$tracks"

    # 2.0 is the quadratic one, k1*k2 pair scores per frame, and it exists to show
    # the pathology rather than to carry a headline. It used to be half the plan's
    # budget; since the scores are evaluated for every frame in one matrix product
    # the 300-atom cell is a minute rather than three quarters of an hour, and the
    # price of the plan is now defmap_local, nmf and defmap_ramp.
    SCRIPT=experiments/defmap_oracle_pair.py
    # shellcheck disable=SC2086
    run defmap_pair $base DICT_SIZES=50,100,300

    # 2.1 and 2.3 are linear in the dictionary.
    SCRIPT=experiments/defmap_phase_ramp.py
    # shellcheck disable=SC2086
    run defmap_ramp $base DICT_SIZES=50,100,300

    # 2.3 and 2.5 share a script: the repaired selection rules have to be read
    # against `local k=N` on the same tracks, dictionaries and frames.
    SCRIPT=experiments/defmap_local_combination.py
    # shellcheck disable=SC2086
    run defmap_local $base DICT_SIZES=50,100,300 K_VALUES=1,2,4,8,16

    # The learned baseline, last because it is the cheapest and because a killed
    # run should lose it rather than the twenty hours above. Same sizes so that
    # `size` means the same thing on both axes of the article's table: components
    # of a learned basis against atoms of a dictionary.
    SCRIPT=experiments/nmf_baseline.py
    # shellcheck disable=SC2086
    run nmf $base DICT_SIZES=50,100,300
}

# The trained baseline, its own plan rather than a fifth cell of plan_defmap:
# Open-Unmix needs torch, which is not a dependency of this repo, so it runs from
# a venv of its own and the caller passes it as PYTHON. `experiments/daneel_umx.sh`
# is what builds that venv. The environment is plan_defmap's to the letter, LOG
# included, so its journal lands beside the four others and the reporter reads it
# as a fifth probe with no intersection to take.
plan_umx() {
    LOG="$LOG/defmap"
    mkdir -p "$LOG"
    tracks=${N_TEST:-50}
    base="CORPUS=musdb REGIME=unseen NFFT=1024 N_TRAIN=25 MAX_TRAIN_FRAMES=20000"
    base="$base TEST_SECONDS=5 SEED=0 N_TEST=$tracks"

    SCRIPT=experiments/openunmix_baseline.py
    # shellcheck disable=SC2086
    run umx $base TORCH_HOME="${TORCH_HOME:-/scratch/torchhub}"
}

case "$PLAN" in
    defmap) plan_defmap ;;
    umx) plan_umx ;;
    *) echo "unknown plan: $PLAN (defmap | umx)" >&2; exit 2 ;;
esac

echo "=== plan $PLAN done ===" >&2
date >&2
