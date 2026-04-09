# Nasz algorytm, usprawniony przez Bartka — wersja z profilingiem
# Profiling wzorowany na rozwiązaniu z Chainera:
# cProfile na całym run() workera, zapis do pliku profile-{PID}.out

"""
A3C Multi-GPU Implementation — wersja z lokalnymi optymalizatorami per worker

Kluczowa zmiana względem poprzedniej wersji:
- Usunięto SharedRMSprop oraz globalne optimizer.step() (linie 496-497)
- Każdy worker trzyma WŁASNY lokalny RMSprop działający na GPU (brak shared memory)
- Po lokalnym kroku gradient aplikowany jest jako delta wag:
    delta = local_param.data - snapshot_param.data
    global_param.data.add_(delta)
  co eliminuje kolejkowanie workerów przy shared memory CPU

Dlaczego to działa:
- Lokalne optimizer.step() działa w izolacji na GPU każdego workera — brak rywalizacji
- Zapis delty do global_param.data to O(n_params) operacji addcdiv_ zamiast
  pełnego kroku RMSprop na shared tensorach — krótszy czas trzymania cache line
- Hogwild! nadal obowiązuje: zapisy do global_param.data są bez locka,
  co jest poprawne dla SGD-family (Recht et al. 2011)

Profiling:
- Każdy worker zapisuje plik profile-{PID}.out do katalogu PROFILE_DIR
- Wzorowane na Chainerze: cProfile.runctx na całej funkcji run workera
- Wyniki można analizować przez: python -m pstats profile-XXXX.out
"""

import glob
import time
import numpy as np
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import gc
import logging
from logging.handlers import RotatingFileHandler
import csv
import cProfile

from collections import namedtuple
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import settings
from torch.distributions.categorical import Categorical

from carla_env import CarlaEnv
from nets.a2c import DiscreteActor as DeepDiscreteActor
from nets.a2c import Critic as DeepCritic

from ACTIONS import ACTIONS as ac
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

LOGGING = settings.LOGGING

###########################################################################
# Configuration
###########################################################################

SEED = 52
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

ACTION_TYPE = settings.ACTION_TYPE
CAMERA_TYPE = settings.CAMERA_TYPE
MODEL_LOAD_PATH = 'A_to_B_GPU_34/PC_models/currently_trained/parallel_03_8workers.pth'
MODEL_SAVE_PATH = 'A_to_B_GPU_34/PC_models/currently_trained/parallel_03_8workers'
EXP_ID = "parallel_03_8workers.pth"

GAMMA = settings.GAMMA
LR = settings.LR
USE_ENTROPY = settings.USE_ENTROPY
SCENARIO = settings.SCENARIO
TESTING = settings.TESTING

NUM_WORKERS = 8
NUMBER_OF_SERVERS_PER_GPU = 2
n_gpus = torch.cuda.device_count()
WORKER_GPUS = ([f'cuda:{g}' for g in range(n_gpus) for _ in range(NUMBER_OF_SERVERS_PER_GPU)])[:NUM_WORKERS]
print(f'!!!!!!!!!!    WORKER_GPUS {WORKER_GPUS}')
BASE_PORT = settings.PORT

T_MAX = 20
MAX_GRAD_NORM = 40.0
ENTROPY_COEF = 0.01
VALUE_LOSS_COEF = 1.0
SAVE_INTERVAL = 20
SYNC_EVERY_N_UPDATES = 1

MAX_RETRIES = 3
CARLA_TIMEOUT_WAIT = 60
WORKER_CHECK_INTERVAL = 5

LOG_FILE = 'log.csv'

# Katalog zapisu plików profilingu — wzorowany na Chainerze
PROFILE_DIR = 'profiles'

# Limit epizodów po którym worker kończy się sam i zapisuje profil
# MAX_EPISODES = 50

Transition = namedtuple("Transition", ["s", "value_s", "a", "log_prob_a"])


def wandb_logger_process(log_queue, shutdown_event):
    if not LOGGING or not WANDB_AVAILABLE:
        return
    os.environ['WANDB_INSECURE_DISABLE_SSL'] = 'true'
    wandb.init(
        project="A_to_B",
        name="synchr_test3_11",
        resume="allow",
        id=EXP_ID,
        config={
            "learning_rate": LR,
            "num_workers": NUM_WORKERS,
            "t_max": T_MAX,
            "gamma": GAMMA,
            "entropy_coef": ENTROPY_COEF,
            "value_loss_coef": VALUE_LOSS_COEF,
            "max_grad_norm": MAX_GRAD_NORM,
        }
    )

    logger.info("wandb logger process started")

    try:
        while not shutdown_event.is_set():
            try:
                record = log_queue.get(timeout=1.0)
            except:
                continue

            if record is None:
                break

            worker_id = record.pop("worker_id", None)

            metrics = {}
            for k, v in record.items():
                if worker_id is not None and k not in ("episode", "global_step"):
                    metrics[f"worker/{worker_id}/{k}"] = v
                else:
                    metrics[k] = v

            step = metrics.pop("global_step", None)
            wandb.log(metrics, step=step)

    except Exception as e:
        logger.error(f"W&B error: {e}", exc_info=True)
    finally:
        wandb.finish()
        logger.info("wandb logger process finished")


def setup_logger(log_file='logs/a3c_training.log', level=logging.INFO):
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    file_handler = RotatingFileHandler(log_file, maxBytes=50*1024*1024, backupCount=5)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    logger = logging.getLogger('A3C')
    logger.setLevel(level)

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

logger = setup_logger()


###########################################################################
# Chainer-style Gradient Transfer and Parameter Sync
###########################################################################

def sync_params_to_local(global_model, local_model):
    """
    Kopiuje wagi z globalnego modelu (CPU, shared memory)
    do lokalnego modelu (GPU).
    Wzorowane na Chainerze: bezpośredni zapis data.copy_() — zero alokacji.
    """
    for global_param, local_param in zip(
        global_model.parameters(),
        local_model.parameters()
    ):
        local_param.data.copy_(global_param.data)


def apply_local_delta_to_global(local_model, snapshot, global_model):
    """
    Aplikuje deltę wag lokalnego modelu do globalnego modelu.

    Zamiast kopiować gradienty i wywoływać optimizer.step() na shared memory
    (co powoduje kolejkowanie workerów), obliczamy delta = local - snapshot
    i dodajemy ją bezpośrednio do global_param.data.

    Jest to poprawne w trybie Hogwild! (bez locka) — Recht et al. 2011
    pokazuje zbieżność SGD bez synchronizacji dla sparse updates.

    Args:
        local_model:  model lokalny workera po optimizer.step() (GPU)
        snapshot:     lista tensorów — kopia wag lokalnego modelu PRZED
                      obliczaniem gradientów (CPU, zwykła pamięć)
        global_model: model globalny w shared memory (CPU)
    """
    with torch.no_grad():
        for (local_param, global_param, snap_param) in zip(
            local_model.parameters(),
            global_model.parameters(),
            snapshot
        ):
            # delta obliczana na CPU aby uniknąć transferu GPU→CPU per-param
            delta = local_param.data.cpu() - snap_param
            # Hogwild!: brak locka — celowo
            global_param.data.add_(delta)


def make_snapshot(model):
    """
    Zwraca listę tensorów CPU będących kopią aktualnych wag modelu.
    Snapshot jest robiony PRZED optimizer.step() — służy do obliczenia delty.

    Używamy .cpu().clone() zamiast .clone() żeby:
    1. Nie zajmować pamięci GPU na snapshot
    2. Operacja delta = local - snap była na CPU (brak CUDA sync)
    """
    return [p.data.cpu().clone() for p in model.parameters()]


###########################################################################
# Global network
###########################################################################

class GlobalNetwork:
    """
    Global network w shared memory (CPU).

    Względem poprzedniej wersji usunięto:
    - SharedRMSprop actor_optimizer i critic_optimizer
    - _init_shared_grads() — gradienty nie są już przenoszone do globalnego modelu

    Aktualizacje wag odbywają się teraz przez apply_local_delta_to_global()
    wywoływane z każdego workera osobno.
    """

    def __init__(self, state_shape, action_shape, critic_shape):
        self.device = torch.device('cpu')

        self.actor = DeepDiscreteActor(state_shape, action_shape, 'cpu').to(self.device)
        self.critic = DeepCritic(state_shape, critic_shape, 'cpu').to(self.device)

        self.actor.share_memory()
        self.critic.share_memory()

        # Brak globalnych optimizerów — każdy worker ma własny lokalny RMSprop
        # Brak _init_shared_grads() — gradienty nie trafiają do globalnego modelu

        self.global_episode = mp.Value('i', 0)
        self.total_updates = mp.Value('i', 0)
        self.best_reward = mp.Value('d', -float('inf'))
        self.global_mean_reward = mp.Value('d', 0.0)
        self.worker_mean_rewards = mp.Array('d', [0.0] * NUM_WORKERS)

        self.stats_lock = mp.Lock()
        self.save_lock = mp.Lock()

        logger.info("=" * 80)
        logger.info("Global Network initialized")
        logger.info("Actor params: %d | Critic params: %d",
                   sum(p.numel() for p in self.actor.parameters()),
                   sum(p.numel() for p in self.critic.parameters()))
        logger.info("Optimizer: LOCAL RMSprop per worker (no shared optimizer)")
        logger.info("Update method: Hogwild! delta transfer (local - snapshot → global)")
        logger.info("LR: %.6f | T_MAX: %d | Workers: %d", LR, T_MAX, NUM_WORKERS)
        logger.info("=" * 80)

    def save(self, path):
        with self.save_lock:
            state = {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "global_mean_reward": self.global_mean_reward.value,
                "best_reward": self.best_reward.value,
                "global_episode": self.global_episode.value,
                "total_updates": self.total_updates.value
            }
            torch.save(state, path + ".pth")
            logger.info("Model saved | Ep: %d | Updates: %d | Best: %.1f | Mean: %.1f",
                       self.global_episode.value, self.total_updates.value,
                       self.best_reward.value, self.global_mean_reward.value)

    def load(self, path):
        state = torch.load(path, map_location=self.device)

        with self.save_lock:
            self.actor.load_state_dict(state["actor"])
            self.critic.load_state_dict(state["critic"])

            # Obsługa starych checkpointów zapisanych z SharedRMSprop
            # — klucze actor_optimizer / critic_optimizer są ignorowane
            if "actor_optimizer" in state:
                logger.info("Ignoring legacy actor_optimizer state (SharedRMSprop → local RMSprop)")
            if "critic_optimizer" in state:
                logger.info("Ignoring legacy critic_optimizer state (SharedRMSprop → local RMSprop)")

            self.global_mean_reward.value = state.get("global_mean_reward", 0.0)
            self.best_reward.value = state.get("best_reward", -float('inf'))
            self.global_episode.value = state.get("global_episode", 0)
            self.total_updates.value = state.get("total_updates", 0)

            logger.info("Model loaded | Ep: %d | Updates: %d | Best: %.1f | Mean: %.1f",
                       self.global_episode.value, self.total_updates.value,
                       self.best_reward.value, self.global_mean_reward.value)

    def update_stats(self, worker_id, mean_reward, episode_reward):
        is_new_best = False

        with self.stats_lock:
            if episode_reward > self.best_reward.value:
                self.best_reward.value = episode_reward
                is_new_best = True

            self.worker_mean_rewards[worker_id] = mean_reward

            active = [r for r in self.worker_mean_rewards[:] if r != 0]
            if active:
                self.global_mean_reward.value = sum(active) / len(active)

            return self.global_mean_reward.value, is_new_best

    def increment_episode(self):
        with self.stats_lock:
            self.global_episode.value += 1
            return self.global_episode.value

    def increment_updates(self):
        self.total_updates.value += 1
        return self.total_updates.value


###########################################################################
# Worker
###########################################################################

class A3CWorker(mp.Process):
    """
    A3C Worker process running on GPU.

    Względem poprzedniej wersji kluczowe zmiany:
    - Dodano self.actor_optimizer i self.critic_optimizer — lokalne, na GPU
    - Dodano self.actor_snapshot i self.critic_snapshot — kopie wag przed rollout
    - compute_and_apply_gradients() robi teraz lokalny optimizer.step()
      zamiast transfer_grads_to_shared() + global optimizer.step()
    - Po lokalnym kroku delty są przenoszone do globalnego modelu przez
      apply_local_delta_to_global()
    - Snapshot jest aktualizowany po każdym sync_with_global()
    """

    def __init__(self, worker_id, global_network, port, device,
                 log_queue=None, shutdown_event=None):
        super(A3CWorker, self).__init__()
        self.worker_id = worker_id
        self.global_network = global_network
        self.port = port
        self.device = torch.device(device)
        self.log_queue = log_queue
        self.shutdown_event = shutdown_event

        self.gamma = GAMMA

        self.trajectory = []
        self.rewards = []
        self.mean_reward = 0.0
        self.episode_count = 0
        self.local_updates = 0
        self.total_steps = 0

        self.episode_rewards_history = []

        self.actor = None
        self.critic = None

        # Lokalne optimizery — inicjalizowane po _init_networks()
        self.actor_optimizer = None
        self.critic_optimizer = None

        # Snapshoty wag — lista tensorów CPU, aktualizowane po każdym sync
        self.actor_snapshot = None
        self.critic_snapshot = None

        self._initialized = False

        logger.info("W%d initialized | Port: %d | Device: %s",
                   self.worker_id, self.port, self.device)

    def _init_networks(self):
        if self._initialized:
            return

        state_shape = [250, 250, 3]
        critic_shape = 1
        action_shape = len(ac.ACTIONS_NAMES)

        self.actor = DeepDiscreteActor(state_shape, action_shape, self.device).to(self.device)
        self.critic = DeepCritic(state_shape, critic_shape, self.device).to(self.device)

        # Lokalny RMSprop na GPU — brak shared memory, brak rywalizacji między workerami
        self.actor_optimizer = torch.optim.RMSprop(
            self.actor.parameters(), lr=LR, alpha=0.99, eps=1e-8, weight_decay=1e-2
        )
        self.critic_optimizer = torch.optim.RMSprop(
            self.critic.parameters(), lr=LR, alpha=0.99, eps=1e-8, weight_decay=1e-2
        )

        self.sync_with_global()
        self._initialized = True
        logger.info("W%d networks initialized on %s | Local RMSprop ready",
                    self.worker_id, self.device)

    def sync_with_global(self):
        """
        Kopiuje wagi z globalnego modelu do lokalnego
        i aktualizuje snapshot (punkt odniesienia dla delty).

        Snapshot musi być robiony PO sync, nie przed — interesuje nas delta
        względem stanu globalnego w momencie pobrania wag.
        """
        sync_params_to_local(self.global_network.actor, self.actor)
        sync_params_to_local(self.global_network.critic, self.critic)

        # Aktualizuj snapshot po skopiowaniu wag z globalnego modelu
        self.actor_snapshot = make_snapshot(self.actor)
        self.critic_snapshot = make_snapshot(self.critic)

    def get_action(self, obs, speed, manouver, testing=False):
        logits = self.actor(obs, speed, manouver)
        value = self.critic(obs, speed, manouver)

        distribution = Categorical(logits=logits)
        self.action_distribution = distribution

        if testing:
            action = distribution.probs.argmax(dim=-1)
        else:
            action = distribution.sample()

        log_prob = distribution.log_prob(action)
        action_np = action.cpu().numpy().squeeze(0)

        if not testing:
            self.trajectory.append(Transition(obs, value, action_np, log_prob))

        return action_np, distribution

    def compute_and_apply_gradients(self, final_state, done, final_speed, manouver):
        """
        Oblicza straty, wykonuje lokalny krok optymalizatora na GPU,
        następnie przenosi deltę wag do globalnego modelu (Hogwild!).

        Przepływ:
          1. Oblicz returns i advantage
          2. total_loss.backward() — gradienty na lokalnym modelu (GPU)
          3. clip_grad_norm_ — lokalnie
          4. actor/critic_optimizer.step() — krok RMSprop na GPU (brak rywalizacji)
          5. apply_local_delta_to_global() — delta → global shared memory
          6. Wyczyść trajectory i rewards
        """
        if len(self.trajectory) == 0:
            return

        returns = []
        with torch.no_grad():
            if done:
                R = torch.zeros(1, 1, device=self.device)
            else:
                R = self.critic(final_state, final_speed, manouver)

        for reward in reversed(self.rewards):
            R = reward + self.gamma * R
            returns.insert(0, R)

        batch = Transition(*zip(*self.trajectory))
        values = batch.value_s
        log_probs = batch.log_prob_a

        policy_loss = 0
        value_loss = 0

        for G_t, V_s, log_prob in zip(returns, values, log_probs):
            advantage = G_t - V_s.detach()
            policy_loss = policy_loss - log_prob * advantage
            value_loss = value_loss + VALUE_LOSS_COEF * F.smooth_l1_loss(V_s, G_t)

        entropy = self.action_distribution.entropy().mean()
        total_loss = policy_loss + value_loss - ENTROPY_COEF * entropy

        self.actor.zero_grad()
        self.critic.zero_grad()
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), MAX_GRAD_NORM)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD_NORM)

        # --- KLUCZOWA ZMIANA względem poprzedniej wersji ---
        # Poprzednio (bottleneck):
        #   transfer_grads_to_shared(self.actor, self.global_network.actor)
        #   transfer_grads_to_shared(self.critic, self.global_network.critic)
        #   self.global_network.actor_optimizer.step()   # ← kolejkowanie workerów
        #   self.global_network.critic_optimizer.step()  # ← kolejkowanie workerów
        #
        # Teraz: lokalny krok na GPU (izolowany per worker) + delta do globalnego

        # Krok 4: lokalny optimizer.step() — każdy worker robi to niezależnie na GPU
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        # Krok 5: przenies deltę (local_after_step - snapshot_before_step) do globalnego
        apply_local_delta_to_global(
            self.actor, self.actor_snapshot, self.global_network.actor
        )
        apply_local_delta_to_global(
            self.critic, self.critic_snapshot, self.global_network.critic
        )

        self.global_network.increment_updates()
        self.local_updates += 1

        self.trajectory.clear()
        self.rewards.clear()

    def log_episode(self, episode_num, ep_reward, episode_length, distance_from_target=None):
        self.episode_rewards_history.append(ep_reward)

        window = min(100, len(self.episode_rewards_history))
        self.mean_reward = np.mean(self.episode_rewards_history[-window:])

        global_mean, is_new_best = self.global_network.update_stats(
            self.worker_id, self.mean_reward, ep_reward
        )

        if is_new_best:
            logger.info("!!! NEW BEST REWARD: %.1f (W%d, Ep%d)",
                      ep_reward, self.worker_id, episode_num)

        if LOGGING:
            logger.info("W%d Ep%d | R: %6.1f | Local mean: %6.1f | Global mean: %6.1f | Len: %3d | Steps: %6d",
                    self.worker_id, episode_num, ep_reward, self.mean_reward,
                    global_mean, episode_length, self.total_steps)

        if self.log_queue is not None:
            try:
                self.log_queue.put_nowait({
                    "worker_id": self.worker_id,
                    "episode": episode_num,
                    "reward": ep_reward,
                    "local_mean_reward": self.mean_reward,
                    "global_mean_reward": global_mean,
                    "episode_length": episode_length,
                    "total_steps": self.total_steps,
                    "distance_from_target": distance_from_target,
                    "global_step": self.global_network.total_updates.value,
                })
            except:
                pass

        try:
            with open(LOG_FILE, 'a', newline='') as f:
                csv.writer(f).writerow([
                    self.worker_id, episode_num, ep_reward, self.mean_reward,
                    global_mean, episode_length, self.total_steps,
                    self.global_network.total_updates.value, distance_from_target
                ])
        except:
            pass

    def _run_worker(self):
        """
        Właściwa logika workera — profilowana przez cProfile.
        Wydzielona z run() analogicznie do train_loop w Chainerze.
        """
        self._init_networks()

        try:
            prev_mean = self.global_network.worker_mean_rewards[self.worker_id]
            if prev_mean != 0.0:
                self.mean_reward = prev_mean
                self.episode_rewards_history = [prev_mean] * 100
                logger.info("W%d restored mean reward from global: %.3f",
                            self.worker_id, prev_mean)
        except:
            pass

        env = None

        try:
            env = CarlaEnv(
                scenario=SCENARIO, spawn_point=False, terminal_point=False,
                mp_density=25, port=self.port,
                action_space=ACTION_TYPE, camera=CAMERA_TYPE,
                resX=250, resY=250, manual_control=False
            )

            while not (self.shutdown_event and self.shutdown_event.is_set()):
                # if self.global_network.global_episode.value >= MAX_EPISODES:
                #     logger.info(
                #         "W%d stopping — global episodes reached %d/%d",
                #         self.worker_id,
                #         self.global_network.global_episode.value,
                #         MAX_EPISODES
                #     )
                #     print(
                #         f"W{self.worker_id} stopping — global episodes "
                #         f"{self.global_network.global_episode.value}/{MAX_EPISODES}",
                #         flush=True
                #     )
                #     break

                try:
                    self.episode_count += 1
                    current_episode = self.global_network.increment_episode()

                    self.sync_with_global()

                    save_images = (current_episode % 200 == 0)
                    env.state_observer.reset()
                    state, speed = env.reset(save_image=save_images, episode=current_episode)

                    state = state / 255.0
                    speed = speed / 100.0

                    done = False
                    ep_reward = 0.0
                    step_count = 0
                    episode_step = 0
                    last_distance_from_target = None

                    maneuver_idx = 0
                    maneuver = env.car_decisions[maneuver_idx]
                    maneuver_tensor = torch.tensor([maneuver], device=self.device)

                    while not done:
                        episode_step += 1
                        self.total_steps += 1

                        if isinstance(state, np.ndarray):
                            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                        else:
                            state_tensor = state.float().to(self.device)
                            if state_tensor.dim() == 3:
                                state_tensor = state_tensor.unsqueeze(0)

                        speed_tensor = torch.tensor([[speed]], dtype=torch.float32, device=self.device)

                        _, left_junction = env.planner.on_junction(env.vehicle.get_location())
                        if left_junction:
                            maneuver_idx += 1
                            if maneuver_idx < len(env.car_decisions):
                                maneuver = env.car_decisions[maneuver_idx]
                            else:
                                maneuver = 1
                            maneuver_tensor = torch.tensor([maneuver], device=self.device)

                        action, self.last_distribution = self.get_action(
                            state_tensor, speed_tensor, maneuver_tensor, TESTING
                        )

                        if save_images:
                            env.state_observer.manouver = maneuver
                            env.state_observer.action = action
                            env.state_observer.step = episode_step
                            env.state_observer.episode = current_episode
                            env.state_observer.save_to_disk()
                            env.state_observer.draw_related_values()
                            env.state_observer.save_together()

                        env.step_apply_action(action)

                        while not env.image_queue.empty():
                            _ = env.image_queue.get()

                        env.world.tick()
                        env.step_apply_action(action)
                        env.world.tick()

                        next_state, reward, done, _, next_speed, distance_from_target = env.step(
                            save_image=save_images, episode=current_episode, step=episode_step
                        )

                        if save_images:
                            env.state_observer.reward = reward

                        next_state = next_state / 255.0
                        next_speed = next_speed / 100.0

                        self.rewards.append(reward)
                        ep_reward += reward
                        step_count += 1
                        last_distance_from_target = distance_from_target

                        if not TESTING and (step_count >= T_MAX or done):
                            if isinstance(next_state, np.ndarray):
                                next_state_tensor = torch.from_numpy(next_state).float().unsqueeze(0).to(self.device)
                            else:
                                next_state_tensor = next_state.float().to(self.device)
                                if next_state_tensor.dim() == 3:
                                    next_state_tensor = next_state_tensor.unsqueeze(0)

                            next_speed_tensor = torch.tensor([[next_speed]], dtype=torch.float32, device=self.device)

                            self.compute_and_apply_gradients(
                                next_state_tensor, done, next_speed_tensor, maneuver_tensor
                            )

                            if not done and self.local_updates % SYNC_EVERY_N_UPDATES == 0:
                                self.sync_with_global()

                            step_count = 0

                        state = next_state
                        speed = next_speed

                    gc.collect()
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()

                    self.log_episode(current_episode, ep_reward, episode_step,
                                     distance_from_target=last_distance_from_target)

                    if self.episode_count % SAVE_INTERVAL == 0:
                        self.global_network.save(MODEL_SAVE_PATH)

                except RuntimeError as e:
                    if "time-out" in str(e):
                        logger.warning("W%d CARLA timeout, reconnecting...", self.worker_id)

                        if env:
                            env.world = None
                            env.client = None

                        time.sleep(CARLA_TIMEOUT_WAIT)

                        env = CarlaEnv(
                            scenario=SCENARIO, spawn_point=False, terminal_point=False,
                            mp_density=25, port=self.port,
                            action_space=ACTION_TYPE, camera=CAMERA_TYPE,
                            resX=250, resY=250, manual_control=False
                        )
                        logger.info("W%d reconnected", self.worker_id)
                    else:
                        raise

        except KeyboardInterrupt:
            logger.info("W%d interrupted", self.worker_id)
        except Exception as e:
            logger.error("W%d crashed: %s", self.worker_id, str(e), exc_info=True)
            raise
        finally:
            if env:
                try:
                    env.world = None
                    env.client = None
                except:
                    pass
            logger.info("W%d terminated", self.worker_id)

    def run(self):
        """
        Punkt wejścia procesu workera.
        Wzorowane na Chainerze: cProfile.runctx opakowuje całą logikę workera.
        Plik profilu zapisywany do PROFILE_DIR/profile-{PID}.out
        """
        logger.info("W%d started (with cProfile)", self.worker_id)

        os.makedirs(PROFILE_DIR, exist_ok=True)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        profile_path = os.path.join(PROFILE_DIR, f'profile-worker{self.worker_id}-{timestamp}.out')

        logger.info("W%d profile will be saved to: %s", self.worker_id, profile_path)
        print(f"W{self.worker_id} profile will be saved to: {profile_path}", flush=True)

        try:
            profiler = cProfile.Profile()
            profiler.enable()
            self._run_worker()
            profiler.disable()
            profiler.dump_stats(profile_path)
            logger.info("W%d profile saved successfully: %s", self.worker_id, profile_path)
            print(f"W{self.worker_id} profile saved successfully: {profile_path}", flush=True)
        except Exception as e:
            logger.error("W%d profiling error: %s", self.worker_id, str(e), exc_info=True)
            raise


###########################################################################
# Handle workers
###########################################################################

def handle_workers(global_network, log_queue=None, shutdown_event=None):
    """Launch and monitor workers. Automatically restart crashed workers."""
    workers = {}
    restart_counts = {i: 0 for i in range(NUM_WORKERS)}

    logger.info("Launching %d workers...", NUM_WORKERS)

    for worker_id in range(NUM_WORKERS):
        port = BASE_PORT + (100 * worker_id)
        device = WORKER_GPUS[worker_id]

        worker = A3CWorker(
            worker_id=worker_id,
            global_network=global_network,
            port=port,
            device=device,
            log_queue=log_queue,
            shutdown_event=shutdown_event
        )
        worker.start()
        workers[worker_id] = worker
        logger.info("W%d launched | Port: %d | Device: %s", worker_id, port, device)

    try:
        while not (shutdown_event and shutdown_event.is_set()):
            time.sleep(WORKER_CHECK_INTERVAL)

            for worker_id in range(NUM_WORKERS):
                worker = workers[worker_id]

                if not worker.is_alive():
                    restart_counts[worker_id] += 1
                    logger.warning("W%d died (restart #%d)",
                                 worker_id, restart_counts[worker_id])

                    worker.join(timeout=2)

                    for core_file in glob.glob('core.*'):
                        try:
                            os.remove(core_file)
                        except:
                            pass

                    wait_time = float(os.getenv('CARLA_SERVER_START_PERIOD', '30.0'))
                    logger.info("W%d waiting %.1fs for CARLA...", worker_id, wait_time)
                    time.sleep(wait_time)

                    port = BASE_PORT + (100 * worker_id)
                    device = WORKER_GPUS[worker_id]

                    worker = A3CWorker(
                        worker_id=worker_id,
                        global_network=global_network,
                        port=port,
                        device=device,
                        log_queue=log_queue,
                        shutdown_event=shutdown_event
                    )
                    worker.start()
                    workers[worker_id] = worker
                    logger.info("W%d restarted (total restarts: %d)",
                              worker_id, restart_counts[worker_id])

    except KeyboardInterrupt:
        logger.info("Stopping all workers...")
        if shutdown_event:
            shutdown_event.set()
        for worker in workers.values():
            if worker.is_alive():
                worker.terminate()
        for worker in workers.values():
            worker.join(timeout=5)

    logger.info("Training finished. Restart counts: %s", restart_counts)


###########################################################################
# Main
###########################################################################

if __name__ == "__main__":
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            csv.writer(f).writerow([
                'worker_id', 'episode', 'reward', 'local_mean',
                'global_mean', 'length', 'total_steps', 'total_updates',
                'distance_from_target'
            ])

    os.makedirs(PROFILE_DIR, exist_ok=True)

    mp.set_start_method('spawn')

    shutdown_event = mp.Event()
    log_queue = mp.Queue(maxsize=1000) if LOGGING else None

    logger.info("=" * 80)
    logger.info("A3C Multi-GPU Training — local RMSprop per worker (Hogwild! delta)")
    logger.info("=" * 80)
    logger.info("Workers: %d | GPUs: %s", NUM_WORKERS, WORKER_GPUS)
    logger.info("LR: %.6f | Gamma: %.4f | T_MAX: %d steps", LR, GAMMA, T_MAX)
    logger.info("Update method: local optimizer.step() + delta transfer to global")
    logger.info("Profile output: %s/profile-{{PID}}.out", PROFILE_DIR)
    logger.info("=" * 80)

    state_shape = [250, 250, 3]
    action_shape = len(ac.ACTIONS_NAMES)
    critic_shape = 1

    global_network = GlobalNetwork(state_shape, action_shape, critic_shape)

    if os.path.isfile(MODEL_LOAD_PATH):
        global_network.load(MODEL_LOAD_PATH)
    else:
        logger.info("Starting training from scratch")

    logger_process = None
    if LOGGING and log_queue is not None and WANDB_AVAILABLE:
        logger_process = mp.Process(
            target=wandb_logger_process,
            args=(log_queue, shutdown_event),
            name="WandBLogger"
        )
        logger_process.start()
        logger.info("wandb logger process spawned")

    try:
        handle_workers(global_network, log_queue=log_queue, shutdown_event=shutdown_event)
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        shutdown_event.set()

        if log_queue is not None:
            try:
                log_queue.put(None)
            except:
                pass

        if logger_process is not None:
            logger_process.join(timeout=10)
            if logger_process.is_alive():
                logger_process.terminate()

        logger.info("=" * 80)
        logger.info("Training complete")
        logger.info("=" * 80)