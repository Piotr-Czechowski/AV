#!/bin/bash
#SBATCH --partition=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=25G
#SBATCH --time=20:50:00
#SBATCH --gpus=1
#SBATCH --account=CHANGE_ME
#SBATCH --job-name=a3c-carla
#SBATCH --output=a3c-carla-log-%J.log
#SBATCH --error=a3c-carla-log-%J.log
#SBATCH --signal=B:USR1@120

# Apptainer pipeline: one job starts the CARLA servers and trains.
# Submit from the template root:
#   sbatch examples/apptainer/train.slurm -w 1
#   sbatch --gpus=2 --cpus-per-task=16 --mem=50G examples/apptainer/train.slurm -w 2
#   bash examples/apptainer/train.slurm -w 1     # the same pipeline without Slurm
#
# Options this script reads:
#   -w,  --workers NUM          A3C workers = CARLA servers        (default: 1)
#        --servers-per-gpu NUM  CARLA servers per GPU               (default: 1)
#        --start-port NUM       first CARLA RPC port                (default: from the job id)
#        --outdir DIR           directory of a new run
#   -r,  --resume DIR           continue the run in DIR
# Every other argument goes to train_a3c.py unchanged, e.g. --scenario 14 --steps 2000000

set -uo pipefail

GRACEFUL_SHUTDOWN_WAIT=75   # seconds training gets to write its last checkpoint

# sbatch runs a copy of this script, so the template root is the submit directory.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_DIR}" || exit 1
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

# --- 1. Arguments: only what this script needs; the rest goes to train_a3c.py ---
NUM_WORKERS=1
SERVERS_PER_GPU=""
START_PORT=""
OUTDIR=""
RESUME=""
TRAIN_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        -w|--workers|--num-workers) NUM_WORKERS="$2"; shift 2 ;;
        --servers-per-gpu) SERVERS_PER_GPU="$2"; shift 2 ;;
        --start-port) START_PORT="$2"; shift 2 ;;
        --outdir) OUTDIR="$2"; shift 2 ;;
        -r|--resume) RESUME="$2"; shift 2 ;;
        -h|--help) sed -n '/^# Apptainer pipeline/,/^$/p' "$0"; exit 0 ;;
        *) TRAIN_ARGS+=("$1"); shift ;;
    esac
done

# Default port block from the job id: two jobs on one node get different ports.
if [ -z "${START_PORT}" ] && [ -z "${CARLA_START_PORT:-}" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
    START_PORT=$((10000 + (SLURM_JOB_ID % 200) * 100))
fi

# --- 2. Environment and run directory ---
if [ -n "${TMPDIR:-}" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
    ln -sfn "$TMPDIR" "/tmp/${SLURM_JOB_ID}" || true
    export TMPDIR="/tmp/${SLURM_JOB_ID}"
fi

# Site-specific modules. Uncomment and edit for your cluster.
# module load GCCcore/11.3.0 Python/3.10.4 2>/dev/null || true
# module load CUDA/12.4.0 cuDNN/9.2.1.18-CUDA-12.4.0 2>/dev/null || true

if [ -n "${VENV:-}" ]; then
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate" || exit 1
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

OUTPUT_DIR="${RESUME:-${OUTDIR:-${PROJECT_DIR}/runs/a3c_${NUM_WORKERS}w_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}}}"
mkdir -p "${OUTPUT_DIR}" || exit 1
RUN_LOG="${OUTPUT_DIR}/a3c_training.log"
CARLA_LOG="${OUTPUT_DIR}/carla_servers.log"
if [ -n "${SLURM_JOB_ID:-}" ]; then
    ln -sfn "${PROJECT_DIR}/a3c-carla-log-${SLURM_JOB_ID}.log" "${OUTPUT_DIR}/slurm.log" 2>/dev/null || true
fi
echo "[PIPELINE] job ${SLURM_JOB_ID:-local} on $(hostname), ${NUM_WORKERS} worker(s), outdir ${OUTPUT_DIR}"

# --- 3. One trap: stop training first, then the launcher (it stops the servers) ---
LAUNCHER_PID=""
TRAIN_PID=""
TAIL_PID=""
GPU_DMON_PID=""
alive() { [ -n "$1" ] && kill -0 "$1" 2>/dev/null; }

cleanup() {
    local code=$?
    trap '' EXIT INT TERM USR1   # nothing may interrupt the cleanup
    if alive "${TRAIN_PID}"; then
        echo "[PIPELINE] stopping training (SIG${STOP_SIGNAL}); waiting up to ${GRACEFUL_SHUTDOWN_WAIT}s"
        kill -"${STOP_SIGNAL}" "${TRAIN_PID}" 2>/dev/null
        local waited=0
        while alive "${TRAIN_PID}" && [ "${waited}" -lt "${GRACEFUL_SHUTDOWN_WAIT}" ]; do
            sleep 1
            waited=$((waited + 1))
        done
        alive "${TRAIN_PID}" && kill -KILL "${TRAIN_PID}" 2>/dev/null
    fi
    for pid in "${LAUNCHER_PID}" "${TAIL_PID}" "${GPU_DMON_PID}"; do
        alive "${pid}" && kill -TERM "${pid}" 2>/dev/null
    done
    wait "${LAUNCHER_PID}" 2>/dev/null
    exit "${code}"
}
STOP_SIGNAL=TERM
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
# Slurm sends SIGUSR1 to this shell before the time limit (--signal=B:USR1@120).
trap 'echo "[SLURM] SIGUSR1: forwarding to training for the last checkpoint"; STOP_SIGNAL=USR1; exit 138' USR1

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi dmon -s pucvmet -o DT > "${OUTPUT_DIR}/gpu_dmon.log" 2>&1 &
    GPU_DMON_PID=$!
fi

# --- 4. CARLA servers: own session, so Ctrl+C does not stop them before training ---
LAUNCHER_CMD=(python -u carla_multiserver_launcher.py --runtime apptainer
    --num-servers "${NUM_WORKERS}" --outdir "${OUTPUT_DIR}")
[ -n "${SERVERS_PER_GPU}" ] && LAUNCHER_CMD+=(--servers-per-gpu "${SERVERS_PER_GPU}")
[ -n "${START_PORT}" ] && LAUNCHER_CMD+=(--start-port "${START_PORT}")
setsid "${LAUNCHER_CMD[@]}" > "${CARLA_LOG}" 2>&1 &
LAUNCHER_PID=$!

# --- 5. Training in the background; it waits for the servers itself ---
TRAIN_CMD=(python -u train_a3c.py --num-workers "${NUM_WORKERS}")
[ -n "${START_PORT}" ] && TRAIN_CMD+=(--start-port "${START_PORT}")
if [ -n "${RESUME}" ]; then
    TRAIN_CMD+=(--resume "${RESUME}")
else
    TRAIN_CMD+=(--outdir "${OUTPUT_DIR}")
fi
[ "${#TRAIN_ARGS[@]}" -gt 0 ] && TRAIN_CMD+=("${TRAIN_ARGS[@]}")
echo "[PIPELINE] ${TRAIN_CMD[*]}"
touch "${RUN_LOG}"
tail -n 0 -f "${RUN_LOG}" &
TAIL_PID=$!
"${TRAIN_CMD[@]}" >> "${RUN_LOG}" 2>&1 &
TRAIN_PID=$!

# --- 6. Wait; fail fast when the launcher dies ---
# Training runs in the background, so a trap runs within a second of a signal.
while alive "${TRAIN_PID}"; do
    if ! alive "${LAUNCHER_PID}"; then
        echo "[PIPELINE] CARLA launcher died. Last lines of ${CARLA_LOG}:" >&2
        tail -n 40 "${CARLA_LOG}" >&2
        exit 1
    fi
    sleep 1
done
wait "${TRAIN_PID}"
exit $?
