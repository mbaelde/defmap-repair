#!/bin/sh
# The Open-Unmix baseline on Daneel, detached. Run from the host.
#
#   LOG=/scratch/logs/lot3 sh experiments/daneel_umx.sh      beside the lot-3 journals
#   N_TEST=4 LOG=/scratch/logs/smoke sh experiments/daneel_umx.sh    smoke run
#
# A container of its own rather than a cell of daneel.sh, for one reason: torch is
# not a dependency of this repo and must not become one. `uv run python` syncs the
# project environment to pyproject.toml on every call, so torch installed into it
# would be removed by the next Def-MAP cell, and the Def-MAP cells must keep
# resolving to exactly what article 1 measured. The venv is therefore built beside
# the project's, under /scratch so a relaunch reuses it rather than pulling 200 MB
# of wheels again, and run.sh is told to use it through PYTHON.
#
# torch and torchaudio come from the CPU index in the *same* install command. Two
# commands is what breaks it: torchaudio resolved from PyPI against a torch from
# the CPU index loads a mismatched ABI and dies on `_torchaudio.abi3.so`.
#
# The weights live in TORCH_HOME under /scratch too, so the download happens once
# and a run with no network still has them.
#
# SRC exists so this can run while a Def-MAP plan is still going: the two
# containers share /scratch but a `git pull` in the checkout the other one is
# reading would corrupt it mid-plan, `sh` reading a script by byte offset. Point
# SRC at a `git worktree` of the branch instead, which only ever writes .git.
set -eu
NAME=defmap-umx
ROOT=${ROOT:-$HOME/data/gasm-demos}
SRC=${SRC:-$ROOT/src/defmap-repair}
CPUS=${CPUS:-3}
VENV=/scratch/venv-umx
CPU_INDEX=https://download.pytorch.org/whl/cpu
GASM="gasm @ git+https://github.com/mbaelde/generative-audio-source-models.git#subdirectory=python"

docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d --name "$NAME" --memory=12g --cpus="$CPUS" \
    -e PYTHONUNBUFFERED=1 -e MUSDB_ROOT=/data/musdb \
    -e TORCH_HOME=/scratch/torchhub \
    -e N_TEST="${N_TEST:-50}" \
    -e UMX_NITER="${UMX_NITER:-0}" \
    -e LOG="${LOG:-/scratch/logs/ceiling}" \
    -v "$SRC:/w" \
    -v "$ROOT/scratch:/scratch" \
    -v "$ROOT/musdb18-7s:/data/musdb:ro" \
    -w /w ghcr.io/astral-sh/uv:python3.13-bookworm \
    sh -c "apt-get update -qq && apt-get install -y -qq ffmpeg git >/dev/null 2>&1 && \
           if [ ! -x $VENV/bin/python ]; then \
             uv venv -q $VENV && \
             VIRTUAL_ENV=$VENV uv pip install -q torch torchaudio --index-url $CPU_INDEX && \
             VIRTUAL_ENV=$VENV uv pip install -q --no-deps openunmix && \
             VIRTUAL_ENV=$VENV uv pip install -q numpy scipy tqdm scikit-learn soundfile musdb '$GASM'; \
           fi && \
           PYTHON=$VENV/bin/python sh experiments/run.sh umx"

echo "$NAME started; follow with: docker logs -f $NAME"
