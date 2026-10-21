#!/usr/bin/env bash
# Train on a remote Linux machine (cloud VM, lab server) so it keeps running while your laptop is off.
#
#   bash scripts/remote_train.sh --name ws --minutes 120 --seeds "0 1 2" --detach
#
# For each seed: warm start (pretrain_bc.py), RL fine-tuning (train.py) for --minutes, then
# evaluate.py on the nominal pilot and 5 random pilots. Results: runs/<name>_s<seed>/, logs in
# runs/<name>.log. With --detach it runs in the background and survives logging out.
# Copy results to your laptop afterwards:  scp -r user@VM_IP:exo-RL-mujoco/runs/<name>_s* runs/
set -euo pipefail

NAME="ws_$(date +%Y%m%d_%H%M)"
MINUTES=120
SEEDS="0"
BC_STEPS=80000
ENVS=""                      # default: half the CPU cores, 2..8
DETACH=0

usage() { sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
while [ $# -gt 0 ]; do
    case "$1" in
        --name) NAME="$2"; shift 2 ;;
        --minutes) MINUTES="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --bc-steps) BC_STEPS="$2"; shift 2 ;;
        --envs) ENVS="$2"; shift 2 ;;
        --detach) DETACH=1; shift ;;
        -h|--help) usage 0 ;;
        *) echo "unknown option: $1"; usage 1 ;;
    esac
done

cd "$(dirname "$0")/.."                       # repo root
mkdir -p runs
LOG="runs/${NAME}.log"

if [ "$DETACH" = 1 ]; then                    # re-run this script in the background, then return
    nohup bash "$0" --name "$NAME" --minutes "$MINUTES" --seeds "$SEEDS" --bc-steps "$BC_STEPS" \
        ${ENVS:+--envs "$ENVS"} > "$LOG" 2>&1 &
    echo "Started in the background (PID $!). Follow progress with:  tail -f $LOG"
    exit 0
fi

# Python environment (created once). Linux puts the interpreter in venv/bin, Windows in venv/Scripts.
if [ ! -d venv ]; then
    echo "[setup] creating venv and installing requirements"
    python3 -m venv venv
    venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
    venv/bin/pip install -q -r requirements.txt
fi
PY=venv/bin/python
[ -x "$PY" ] || PY=venv/Scripts/python.exe

if [ -z "$ENVS" ]; then
    CORES=$(nproc 2>/dev/null || echo 4)
    ENVS=$(( CORES / 2 )); [ "$ENVS" -lt 2 ] && ENVS=2; [ "$ENVS" -gt 8 ] && ENVS=8
fi
echo "[run] $NAME | seeds: $SEEDS | $MINUTES min RL each | $ENVS parallel envs | started $(date)"

for SEED in $SEEDS; do
    RUN="${NAME}_s${SEED}"
    echo "[run] seed $SEED: warm start (imitation learning)"
    "$PY" scripts/pretrain_bc.py --name "${RUN}_bc" --steps "$BC_STEPS" --set seed="$SEED"

    echo "[run] seed $SEED: RL fine-tuning for $MINUTES min"
    "$PY" scripts/train.py --name "$RUN" --config "runs/${RUN}_bc/config.yaml" \
        --init-from "runs/${RUN}_bc/checkpoints/bc.pt" --init-buffer "runs/${RUN}_bc/checkpoints/bc_buffer.npz" \
        --set train.num_envs="$ENVS" --set train.updates_per_step=0.25 --set train.eval_async=true \
        --max-minutes "$MINUTES"

    CKPT=""                                    # best.pt exists once training has been evaluated
    for C in best.pt latest.pt; do
        if [ -f "runs/$RUN/checkpoints/$C" ]; then CKPT="$C"; break; fi
    done
    if [ -n "$CKPT" ]; then
        echo "[run] seed $SEED: evaluation ($CKPT)"
        "$PY" scripts/evaluate.py --run "runs/$RUN" --ckpt "$CKPT" --episodes 3 --randomized 5
    else
        echo "[run] seed $SEED: no checkpoint to evaluate yet (run longer than one train.eval_every)"
    fi
done
echo "[run] done $(date). Results: runs/${NAME}_s*/eval_best/"
