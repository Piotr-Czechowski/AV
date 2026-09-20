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
#SBATCH --signal=SIGUSR1@90

# Submit from the template root:
#   cd /path/to/A3C-CARLA-Template && sbatch examples/hpc/train.slurm -w 1
# Two workers need matching GPU/CPU/RAM, e.g.:
#   sbatch --gpus=2 --cpus-per-task=16 --mem=50G examples/hpc/train.slurm -w 2
#
# This recipe starts CARLA (unless --no-carla) then train_a3c.py.
# train_a3c.py itself never launches the simulator.

# =============================================================================
# HPC recipe: one job starts N CARLA servers, waits for ports, then trains.
#
# Usage:
#   sbatch [SBATCH_FLAGS] examples/hpc/train.slurm [OPTIONS]
#
# Common run args:
#   -w,  --workers NUM                A3C workers = CARLA servers             (default: 1)
#        --workers-per-gpu NUM        learner workers per visible GPU         (default: 1)
#        --servers-per-gpu NUM        CARLA servers per visible GPU           (default: workers-per-gpu)
#   -s,  --scenario NUM               CARLA scenario id                       (default: 14)
#        --save-episodes "EP1 EP2..." episodes whose frames get saved
#   -r,  --resume DIR                 resume from previous outdir
#        --outdir DIR                 explicit outdir override
#        --steps NUM                  override Python global step budget
#        --seed NUM                   RNG seed
#        --no-wandb                   disable W&B
#        --testing                    evaluation mode
#        --verbose-env-logs           keep per-episode CarlaEnv stdout prints
#        --no-carla                   skip launching CARLA (servers already up)
#        --start-port NUM             first CARLA rpc port                    (default: 2000)
#        --port-step NUM              port increment between servers          (default: 100)
#        --graceful-shutdown-wait NUM seconds to let Python finish on signal  (default: 75)
#        --                          pass remaining args directly to Python
# =============================================================================

set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
CONFIG_FILE="${CONFIG_FILE:-${PROJECT_DIR}/.env}"
if [[ -f "${CONFIG_FILE}" ]]; then
    export CONFIG_FILE
    set -a
    # shellcheck disable=SC1090
    source "${CONFIG_FILE}"
    set +a
fi

MULTISERVER_SCRIPT="${MULTISERVER_SCRIPT:-${PROJECT_DIR}/carla_multiserver_launcher.py}"
RUNS_DIR="${RUNS_DIR:-${PROJECT_DIR}/runs}"
VENV="${VENV:-}"
CARLA_RUNTIME="${CARLA_RUNTIME:-apptainer}"
CARLA_CONTAINER_IMAGE="${CARLA_CONTAINER_IMAGE:-}"
WANDB_PROJECT="${WANDB_PROJECT:-a3c-carla}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

NUM_WORKERS=1
WORKERS_PER_GPU=1
WORKER_GPU_START=0
SERVER_GPU_START=0
SCENARIO=14
START_PORT="${CARLA_START_PORT:-2000}"
PORT_STEP="${CARLA_PORT_STEP:-100}"
GRACEFUL_SHUTDOWN_WAIT=75
CARLA_SERVER_START_PERIOD=30

STEPS=""
LOG_STEPS=""
LOG_UPDATE_ARRAYS=""
LOG_RESOURCES=""
VERBOSE_ENV_LOGS=""
NO_WANDB=""
SAVE_EPISODES=""
TESTING=""
RESUME=""
OUTDIR_OVERRIDE=""
SEED=""
SERVERS_PER_GPU=""
NO_CARLA=""
PYTHON_EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -w|--workers) NUM_WORKERS="$2"; shift 2 ;;
        --workers-per-gpu) WORKERS_PER_GPU="$2"; shift 2 ;;
        --worker-gpu-start) WORKER_GPU_START="$2"; shift 2 ;;
        --server-gpu-start) SERVER_GPU_START="$2"; shift 2 ;;
        --servers-per-gpu) SERVERS_PER_GPU="$2"; shift 2 ;;
        -s|--scenario) SCENARIO="$2"; shift 2 ;;
        --start-port) START_PORT="$2"; shift 2 ;;
        --port-step) PORT_STEP="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        --graceful-shutdown-wait) GRACEFUL_SHUTDOWN_WAIT="$2"; shift 2 ;;
        --log-steps) LOG_STEPS="--log-steps"; shift ;;
        --log-update-arrays) LOG_UPDATE_ARRAYS="--log-update-arrays"; shift ;;
        --log-resources) LOG_RESOURCES="--log-resources"; shift ;;
        --verbose-env-logs) VERBOSE_ENV_LOGS="--verbose-env-logs"; shift ;;
        --no-wandb) NO_WANDB="--no-wandb"; shift ;;
        --save-episodes|--episodes-to-save) SAVE_EPISODES="$2"; shift 2 ;;
        --testing) TESTING="--testing"; shift ;;
        -r|--resume) RESUME="$2"; shift 2 ;;
        --outdir) OUTDIR_OVERRIDE="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --no-carla) NO_CARLA="1"; shift ;;
        --venv) VENV="$2"; shift 2 ;;
        --project-dir) PROJECT_DIR="$2"; shift 2 ;;
        --multiserver-script) MULTISERVER_SCRIPT="$2"; shift 2 ;;
        --python-arg) PYTHON_EXTRA_ARGS+=("$2"); shift 2 ;;
        --) shift; PYTHON_EXTRA_ARGS+=("$@"); break ;;

        -h|--help)
            sed -n '21,44p' "$0"; exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Use -- before rare Python-only arguments, e.g. -- --lr 5e-5" >&2
            exit 1 ;;
    esac
done

if [ -z "${SERVERS_PER_GPU}" ]; then
    SERVERS_PER_GPU="${WORKERS_PER_GPU}"
fi

if [ -n "${TMPDIR:-}" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
    ln -sfn "$TMPDIR" "/tmp/${SLURM_JOB_ID}" || true
    export TMPDIR="/tmp/${SLURM_JOB_ID}"
fi

# Site-specific modules. Uncomment and edit for your cluster.
# module load GCCcore/13.2.0 Python/3.11.5 2>/dev/null || true
# module load CUDA/12.4.0 cuDNN/9.2.1.18-CUDA-12.4.0 2>/dev/null || true

if [ -n "${VENV}" ]; then
    if [ ! -d "${VENV}" ]; then
        echo "ERROR: VENV=${VENV} does not exist" >&2
        exit 1
    fi
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
else
    echo "[SLURM] VENV not set; using the Python already on PATH"
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CARLA_SERVER_START_PERIOD

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [ -n "${RESUME}" ]; then
    OUTPUT_DIR="${RESUME}"
    echo "[SLURM] resuming from: ${OUTPUT_DIR}"
elif [ -n "${OUTDIR_OVERRIDE}" ]; then
    OUTPUT_DIR="${OUTDIR_OVERRIDE}"
    mkdir -p "${OUTPUT_DIR}"
else
    OUTPUT_DIR="${RUNS_DIR}/a3c_${NUM_WORKERS}w_${TIMESTAMP}_${SLURM_JOB_ID:-local}"
    mkdir -p "${OUTPUT_DIR}"
fi

if [ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    ln -sfn "${SLURM_SUBMIT_DIR}/a3c-carla-log-${SLURM_JOB_ID}.log" \
        "${OUTPUT_DIR}/slurm.log" 2>/dev/null || true
fi

RUN_LOG="${OUTPUT_DIR}/a3c_training.log"
GPU_LOG="${OUTPUT_DIR}/gpu_dmon.log"
CARLA_LOG="${OUTPUT_DIR}/carla_servers.log"

echo "==================================================================="
echo "=== A3C CARLA training ==="
echo "==================================================================="
echo "  job id            : ${SLURM_JOB_ID:-local}"
echo "  host              : $(hostname)"
echo "  visible GPUs      : ${CUDA_VISIBLE_DEVICES:-all}"
echo "  outdir            : ${OUTPUT_DIR}"
echo "  project dir       : ${PROJECT_DIR}"
echo "  venv              : ${VENV}"
echo "  carla runtime     : ${CARLA_RUNTIME}"
echo "-------------------------------------------------------------------"
echo "  workers           : ${NUM_WORKERS}"
echo "  workers-per-gpu   : ${WORKERS_PER_GPU}"
echo "  worker-gpu-start  : ${WORKER_GPU_START}"
echo "  server-gpu-start  : ${SERVER_GPU_START}"
echo "  servers-per-gpu   : ${SERVERS_PER_GPU}"
echo "  start-port        : ${START_PORT} (+${PORT_STEP} per server)"
echo "  scenario          : ${SCENARIO}"
echo "==================================================================="

cd "${PROJECT_DIR}"

CARLA_PID=""
GPU_DMON_PID=""
TRAIN_PID=""
TAIL_PID=""

cleanup() {
    local code="${1:-$?}"
    trap - SIGINT SIGTERM SIGUSR1 EXIT
    echo ""
    echo "[CLEANUP] exit code=${code}; stopping child processes..."

    local train_pid="${TRAIN_PID:-}"
    if [ -n "${train_pid}" ] && ps -p "${train_pid}" > /dev/null 2>&1; then
        echo "[CLEANUP] SIGTERM TRAIN_PID=${train_pid}; waiting up to ${GRACEFUL_SHUTDOWN_WAIT}s"
        kill -TERM "${train_pid}" 2>/dev/null || true
        local waited=0
        while [ "${waited}" -lt "${GRACEFUL_SHUTDOWN_WAIT}" ] && \
              ps -p "${train_pid}" > /dev/null 2>&1; do
            sleep 1
            waited=$((waited + 1))
        done
    fi

    for pid_var in TAIL_PID CARLA_PID GPU_DMON_PID; do
        local pid="${!pid_var:-}"
        if [ -n "${pid}" ] && ps -p "${pid}" > /dev/null 2>&1; then
            echo "[CLEANUP] SIGTERM ${pid_var}=${pid}"
            kill -TERM "${pid}" 2>/dev/null || true
        fi
    done

    if [ -n "${train_pid}" ] && ps -p "${train_pid}" > /dev/null 2>&1; then
        echo "[CLEANUP] SIGKILL TRAIN_PID=${train_pid}"
        kill -KILL "${train_pid}" 2>/dev/null || true
    fi

    sleep 3
    for pid_var in TAIL_PID CARLA_PID GPU_DMON_PID; do
        local pid="${!pid_var:-}"
        if [ -n "${pid}" ] && ps -p "${pid}" > /dev/null 2>&1; then
            echo "[CLEANUP] SIGKILL ${pid_var}=${pid}"
            kill -KILL "${pid}" 2>/dev/null || true
        fi
    done
    echo "[CLEANUP] done."
    exit "${code}"
}

on_preempt() {
    trap - SIGUSR1
    echo ""
    echo "[SLURM] SIGUSR1: forwarding to training for last checkpoint"
    local train_pid="${TRAIN_PID:-}"
    local code=1
    if [ -n "${train_pid}" ] && ps -p "${train_pid}" > /dev/null 2>&1; then
        kill -USR1 "${train_pid}" 2>/dev/null || kill -TERM "${train_pid}" 2>/dev/null || true
        local waited=0
        while [ "${waited}" -lt "${GRACEFUL_SHUTDOWN_WAIT}" ] && \
              ps -p "${train_pid}" > /dev/null 2>&1; do
            sleep 1
            waited=$((waited + 1))
        done
        wait "${train_pid}" 2>/dev/null
        code=$?
    fi
    cleanup "${code}"
}
trap cleanup SIGINT SIGTERM EXIT
trap on_preempt SIGUSR1

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi dmon -s pucvmet -o DT > "${GPU_LOG}" 2>&1 &
    GPU_DMON_PID=$!
    echo "[SLURM] gpu watchdog PID=${GPU_DMON_PID} -> ${GPU_LOG}"
    nvidia-smi || true
fi

if [ -z "${NO_CARLA}" ]; then
    if [ "${CARLA_RUNTIME}" != "native" ]; then
        if [ -z "${CARLA_CONTAINER_IMAGE}" ] || \
           [ "${CARLA_CONTAINER_IMAGE}" != "${CARLA_CONTAINER_IMAGE/CHANGE_ME/}" ]; then
            echo "ERROR: set CARLA_CONTAINER_IMAGE in ${CONFIG_FILE:-.env}" >&2
            echo "       (or pass --no-carla if the servers are already running)" >&2
            exit 1
        fi
    fi
    if [ ! -f "${MULTISERVER_SCRIPT}" ]; then
        echo "ERROR: MULTISERVER_SCRIPT=${MULTISERVER_SCRIPT} does not exist" >&2
        exit 1
    fi
    echo "[SLURM] starting ${NUM_WORKERS} CARLA server(s) runtime=${CARLA_RUNTIME}..."
    LAUNCHER_CMD=(python -u "${MULTISERVER_SCRIPT}"
        --runtime "${CARLA_RUNTIME}"
        --num-servers "${NUM_WORKERS}"
        --servers-per-gpu "${SERVERS_PER_GPU}"
        --server-gpu-start "${SERVER_GPU_START}"
        --start-port "${START_PORT}"
        --port-step "${PORT_STEP}"
        --outdir "${OUTPUT_DIR}"
    )
    if [ -n "${CARLA_CONTAINER_IMAGE}" ]; then
        LAUNCHER_CMD+=(--image "${CARLA_CONTAINER_IMAGE}")
    fi
    if [ -n "${CARLA_BINARY:-}" ]; then
        LAUNCHER_CMD+=(--binary "${CARLA_BINARY}")
    fi
    if [ -n "${CARLA_PATH:-}" ]; then
        LAUNCHER_CMD+=(--carla-path "${CARLA_PATH}")
    fi
    nohup "${LAUNCHER_CMD[@]}" > "${CARLA_LOG}" 2>&1 &
    CARLA_PID=$!
    echo "[SLURM] CARLA launcher PID=${CARLA_PID} -> ${CARLA_LOG}"

    EXPECTED_PORTS=()
    for ((i=0; i<NUM_WORKERS; i++)); do
        EXPECTED_PORTS+=($((START_PORT + i * PORT_STEP)))
    done
    echo "[SLURM] waiting for ports: ${EXPECTED_PORTS[*]}"

    retries=60
    delay=10
    for ((r=1; r<=retries; r++)); do
        if [ -n "${CARLA_PID}" ] && ! ps -p "${CARLA_PID}" > /dev/null 2>&1; then
            echo "[SLURM] CARLA launcher died while waiting for ports." >&2
            echo "[SLURM] last 80 lines of ${CARLA_LOG}:" >&2
            tail -n 80 "${CARLA_LOG}" >&2 || true
            exit 1
        fi
        all_ok=true
        for port in "${EXPECTED_PORTS[@]}"; do
            if ! lsof -nP -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1; then
                all_ok=false; break
            fi
        done
        if ${all_ok}; then
            echo "[SLURM] all CARLA ports LISTEN after ${r} check(s)."
            break
        fi
        if [ "${r}" -eq "${retries}" ]; then
            echo "[SLURM] FAILED to bring up all CARLA servers in time." >&2
            echo "[SLURM] last 80 lines of ${CARLA_LOG}:" >&2
            tail -n 80 "${CARLA_LOG}" >&2 || true
            exit 1
        fi
        sleep "${delay}"
    done
else
    echo "[SLURM] --no-carla: skipping CARLA startup."
fi

TRAIN_CMD=(python -u train_a3c.py
    --num-workers "${NUM_WORKERS}"
    --workers-per-gpu "${WORKERS_PER_GPU}"
    --worker-gpu-start "${WORKER_GPU_START}"
    --start-port "${START_PORT}"
    --port-step "${PORT_STEP}"
    --outdir "${OUTPUT_DIR}"
)

TRAIN_CMD+=(--scenario)
# shellcheck disable=SC2206
SCENARIO_VALUES=(${SCENARIO})
TRAIN_CMD+=("${SCENARIO_VALUES[@]}")

[ -n "${STEPS}" ] && TRAIN_CMD+=(--steps "${STEPS}")
[ -n "${SEED}" ] && TRAIN_CMD+=(--seed "${SEED}")
[ -n "${RESUME}" ] && TRAIN_CMD+=(--resume "${RESUME}")
[ -n "${LOG_STEPS}" ] && TRAIN_CMD+=(--log-steps)
[ -n "${LOG_UPDATE_ARRAYS}" ] && TRAIN_CMD+=(--log-update-arrays)
[ -n "${LOG_RESOURCES}" ] && TRAIN_CMD+=(--log-resources)
[ -n "${NO_WANDB}" ] && TRAIN_CMD+=(--no-wandb)
[ -n "${WANDB_PROJECT}" ] && TRAIN_CMD+=(--wandb-project "${WANDB_PROJECT}")
[ -n "${WANDB_RUN_NAME}" ] && TRAIN_CMD+=(--wandb-run-name "${WANDB_RUN_NAME}")
[ -n "${WANDB_ENTITY}" ] && TRAIN_CMD+=(--wandb-entity "${WANDB_ENTITY}")
[ -n "${TESTING}" ] && TRAIN_CMD+=(--testing)
[ -n "${VERBOSE_ENV_LOGS}" ] && TRAIN_CMD+=(--verbose-env-logs)

if [ -n "${SAVE_EPISODES}" ]; then
    TRAIN_CMD+=(--save-episodes)
    # shellcheck disable=SC2206
    SAVE_EPISODE_VALUES=(${SAVE_EPISODES})
    TRAIN_CMD+=("${SAVE_EPISODE_VALUES[@]}")
fi

if [ "${#PYTHON_EXTRA_ARGS[@]}" -gt 0 ]; then
    TRAIN_CMD+=("${PYTHON_EXTRA_ARGS[@]}")
fi

echo "[SLURM] training command:"
printf '  %q ' "${TRAIN_CMD[@]}"; echo ""
echo ""

touch "${RUN_LOG}"
"${TRAIN_CMD[@]}" >> "${RUN_LOG}" 2>&1 &
TRAIN_PID=$!
echo "[SLURM] training PID=${TRAIN_PID} -> ${RUN_LOG}"
echo ""
echo "==================================================================="
echo "=== tailing training log (Ctrl+C stops training AND this job) ==="
echo "==================================================================="
tail -n +1 -f "${RUN_LOG}" --pid=${TRAIN_PID} &
TAIL_PID=$!

wait "${TRAIN_PID}"
TRAIN_RC=$?
echo ""
echo "[SLURM] training exited with code ${TRAIN_RC}"

kill -TERM "${TAIL_PID}" 2>/dev/null || true

exit "${TRAIN_RC}"
