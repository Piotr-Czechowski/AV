# Implementacja `new_hogwild_*` A3C dla CARLA - pełna dokumentacja po polsku

Ten dokument opisuje aktualną implementację A3C/Hogwild z katalogu
`AV/A_to_B_GPU_34/`, czyli pliki `new_hogwild_*` oraz bezpośrednie zależności,
z których ta implementacja korzysta. Opis jest pisany tak, aby dało się go
czytać bez wcześniejszej znajomości reinforcement learning, PyTorch, CUDA ani
tego repozytorium.

Dokument powstał przez analizę kodu w repozytorium `AV/`, ze szczególnym
naciskiem na:

- `new_hogwild_train.slurm`
- `new_hogwild_train_a3c_carla.py`
- `new_hogwild_run_a3c.py`
- `new_hogwild_a3c.py`
- `new_hogwild_carla_wrapper.py`
- `new_hogwild_training_logger.py`
- `new_hogwild_system_monitor.py`
- `new_hogwild_timing_utils.py`
- `new_hogwild_prepare_output_dir.py`
- `carla_env.py`
- `ACTIONS.py`
- `settings.py`
- `utils.py`
- `state_observer.py`

Stan opisu: 2026-06-06. Opis dotyczy kodu znajdującego się lokalnie w tym
repozytorium, nie ogólnej, idealnej implementacji A3C.

## Spis treści

1. Cel całego repozytorium `AV/`
2. Jak `A_to_B_GPU_34` pasuje do repozytorium
3. Co implementacja `new_hogwild_*` próbuje zrobić
4. Minimalne podstawy: tensory, PyTorch, autograd, CUDA
5. Minimalne podstawy: CARLA jako środowisko
6. Minimalne podstawy: reinforcement learning i A3C
7. Mapa plików aktualnej implementacji
8. Uruchomienie przez SLURM
9. Główne wejście Pythona: `new_hogwild_train_a3c_carla.py`
10. Globalny model i optymalizator: `GlobalNetwork`
11. Architektura sieci `SharedActorCritic`
12. Worker A3C: lokalny model, rollout, akcja, loss, gradienty
13. Hogwild w tej implementacji
14. Wrapper CARLA: obserwacje, akcje, reward i koniec epizodu
15. Niskopoziomowe środowisko `CarlaEnv`
16. Akcje dyskretne
17. Reward: legacy i shaped
18. Checkpointy, resume i rollback
19. Logowanie, monitoring i timing
20. Dokładny przepływ danych od obrazu CARLA do update'u wag
21. Konfiguracja i domyślne wartości
22. Różnice względem starszych plików A2C/A3C
23. Znane ograniczenia i miejsca ryzyka
24. Jak czytać wyniki treningu
25. Słownik pojęć

## 1. Cel całego repozytorium `AV/`

Repozytorium `AV/` dotyczy autonomicznej jazdy w symulatorze CARLA. README
opisuje dwa główne projekty:

- `A_to_B/` - jazda autonomiczna z punktu A do punktu B.
- `Chase/` - autonomiczne ściganie innego pojazdu.

W `A_to_B/` znajdują się starsze implementacje A2C, modele końcowe, modele
trenowane na PC, środowisko CARLA i nawigacja. W `Chase/` jest podobny zestaw
plików, ale dla zadania ścigania. Katalog `Gif/` zawiera przykłady wizualne
scenariuszy.

`A_to_B_GPU_34/` jest rozwiniętą wersją projektu `A_to_B`, przygotowaną pod
trening wieloprocesowy/wieloserwerowy na GPU. Zawiera starsze pliki A2C/A3C,
ale aktualna implementacja opisywana tutaj jest w plikach `new_hogwild_*`.

Ważne: repozytorium ma kilka generacji kodu. Nie wszystkie pliki są aktualnie
używane przez `new_hogwild_*`. Na przykład `nets/a2c.py` zawiera starsze
osobne sieci actora i critica, natomiast aktualny `new_hogwild_a3c.py` definiuje
własną sieć `SharedActorCritic` i jej używa.

## 2. Jak `A_to_B_GPU_34` pasuje do repozytorium

Najważniejsze katalogi w `AV/A_to_B_GPU_34/`:

| Ścieżka | Rola |
|---|---|
| `new_hogwild_*.py` | Aktualna modułowa implementacja A3C/Hogwild. |
| `new_hogwild_train.slurm` | Launcher dla klastra: start CARLA, start treningu, cleanup. |
| `carla_env.py` | Niskopoziomowe środowisko CARLA. |
| `carla_navigation/` | Planowanie trasy: global route planner, local planner, controller. |
| `ACTIONS.py` | Definicja dyskretnej przestrzeni akcji pojazdu. |
| `utils.py` | Funkcja reward dla trybu `legacy` i pomocniczy kolorowy logger. |
| `state_observer.py` | Pomocniczy zapis/rysowanie klatek. |
| `a3c.py`, `a3c_improved*.py`, `a3c_multigpu_profile.py` | Starsze wersje A3C. |
| `a2c_rgb*.py` | Starsze wersje A2C. |
| `nets/a2c.py` | Starsze definicje actora/critica. |
| `PC_models/`, `final_models/` | Modele zapisane wcześniej. |
| `runs/` | Wyniki uruchomień `new_hogwild_*`. |
| `pytorch_grad_cam/`, `grad-camera.py` | Narzędzia wizualizacji CAM, nie są rdzeniem treningu A3C. |

Katalog `runs/` jest katalogiem wyników. W analizowanym repozytorium są tam
nieśledzone artefakty treningu, np. `runs/a3c_hogwild_...`. Nie są one częścią
kodu algorytmu, ale pokazują oczekiwany układ wyników.

## 3. Co implementacja `new_hogwild_*` próbuje zrobić

Implementacja trenuje agenta do jazdy w CARLA metodą A3C:

1. Startuje kilka niezależnych serwerów CARLA, po jednym na workera.
2. Startuje jeden proces nadrzędny Pythona.
3. Proces nadrzędny tworzy globalny model na CPU w pamięci współdzielonej.
4. Tworzy wielu workerów `A3CWorker`.
5. Każdy worker:
   - łączy się ze swoim serwerem CARLA na osobnym porcie,
   - ma lokalną kopię modelu na przypisanym urządzeniu, zwykle `cuda:0`,
   - zbiera krótki rollout w środowisku,
   - liczy loss i gradienty lokalnie,
   - kopiuje gradienty do globalnego modelu na CPU,
   - wykonuje krok optymalizatora globalnego.
6. Workery nie czekają na siebie. To jest część "asynchronous" w A3C.
7. Supervisor obserwuje workery i restartuje te, które padły.
8. Checkpointy i logi są zapisywane w katalogu runa.

Najkrótszy obraz architektury:

```text
SLURM job
  |
  +-- CARLA server 0  <--->  worker 0  --\
  +-- CARLA server 1  <--->  worker 1  ---+--> globalny model CPU shared memory
  +-- CARLA server N  <--->  worker N  --/        + shared optimizer
```

Najważniejsza decyzja architektoniczna: globalny model jest na CPU, a lokalne
modele workerów mogą być na GPU. Dzięki temu parametry globalne i stan
optymalizatora można współdzielić między procesami przez PyTorch shared memory.

## 4. Minimalne podstawy: tensory, PyTorch, autograd, CUDA

### 4.1 Tensor

Tensor w PyTorch to wielowymiarowa tablica liczb. Przykłady:

- obraz z kamery: tensor o kształcie `[3, 250, 250]`,
- batch jednego obrazu: `[1, 3, 250, 250]`,
- prędkość pojazdu: `[1, 1]`,
- manewr trasy: `[1]`,
- logits polityki dla 10 akcji: `[1, 10]`,
- wartość stanu z critica: `[1, 1]`.

W tej implementacji obraz jest przetwarzany w formacie `CHW`, czyli:

```text
C = channels = 3
H = height
W = width
```

PyTorchowe konwolucje `nn.Conv2d` oczekują batcha w formacie `NCHW`:

```text
N = batch size
C = channels
H = height
W = width
```

Dlatego pojedynczy stan środowiska `[3, 250, 250]` jest przed forwardem
zamieniany na `[1, 3, 250, 250]` przez `unsqueeze(0)`.

### 4.2 CPU i CUDA

Tensor może być na CPU albo GPU:

- CPU: `torch.device('cpu')`
- GPU: `torch.device('cuda:0')`, `torch.device('cuda:1')`, itd.

Operacje muszą zwykle działać na tensorach na tym samym urządzeniu. Jeżeli model
jest na `cuda:0`, wejście też musi być na `cuda:0`. W `SharedActorCritic.forward`
wejście jest przenoszone do `self.device`:

```python
x = x.to(self.device, dtype=torch.float32)
```

To samo dzieje się z prędkością i manewrem.

### 4.3 Autograd

PyTorch buduje graf obliczeń podczas forward pass. Jeżeli policzymy loss i
wywołamy:

```python
total_loss.backward()
```

PyTorch obliczy gradienty parametrów modelu i zapisze je w `param.grad`.

W tej implementacji:

1. Worker liczy forward lokalnego modelu.
2. Worker zapisuje `value`, `log_prob` i `entropy` z każdego kroku rollouta.
3. Po zebraniu rollouta liczy `total_loss`.
4. Wywołuje `backward()` na lokalnym modelu.
5. Kopiuje lokalne gradienty do globalnego modelu na CPU.
6. Wywołuje `global_optimizer.step()`.

### 4.4 `detach()`

`detach()` odcina tensor od grafu autograd. Używa się go, kiedy jakaś wartość ma
być traktowana jako stała. W implementacji:

- returns są odłączane od grafu,
- value użyte do advantage dla policy też jest odłączane,
- gradienty są kopiowane jako `local_param.grad.detach()`.

Dzięki temu loss policy nie próbuje modyfikować sposobu liczenia targetu.

### 4.5 `share_memory_()`

`share_memory_()` przenosi storage tensora do pamięci współdzielonej między
procesami. Dla multiprocessing jest to kluczowe: wiele procesów może widzieć
ten sam tensor w pamięci.

W `GlobalNetwork`:

- `self.model.share_memory()` współdzieli parametry modelu,
- `SharedRMSprop.share_memory()` współdzieli stan optymalizatora,
- liczniki są `mp.Value`,
- tablice statystyk są `mp.Array`.

Gradienty `.grad` nie są współdzielone. To ważne i celowe: każdy worker ustawia
gradient globalnego modelu we własnym procesie, a współdzielone są parametry i
stan optymalizatora.

## 5. Minimalne podstawy: CARLA jako środowisko

CARLA działa jako symulator klient-serwer:

- serwer CARLA renderuje świat, pojazd, sensory i fizykę,
- klient Python łączy się z serwerem przez port, np. `2000`,
- klient wysyła sterowanie pojazdem i odbiera obrazy oraz dane z sensorów.

W tej implementacji każdy worker ma własny port:

```text
worker 0 -> port 2000
worker 1 -> port 2100
worker 2 -> port 2200
...
```

`CarlaEnv` ładuje mapę `Town03` i ustawia tryb synchroniczny:

```python
self.settings.synchronous_mode = True
self.settings.fixed_delta_seconds = 0.1
self.settings.max_substep_delta_time = 0.01
self.settings.max_substeps = 10
```

Tryb synchroniczny znaczy: symulator idzie do przodu dopiero po `world.tick()`.
To jest ważne przy RL, bo agent chce mieć kontrolę nad tym, kiedy stan świata
zmienia się po akcji.

Kamera działa przez kolejkę:

1. Sensor kamery wykonuje `listen(self.image_queue.put)`.
2. CARLA wrzuca klatki do kolejki.
3. Kod pobiera najnowszą klatkę przez `_get_latest_camera_image`.
4. Klatka jest konwertowana na tensor obrazu.

## 6. Minimalne podstawy: reinforcement learning i A3C

### 6.1 Podstawowe pojęcia RL

W reinforcement learning agent uczy się przez interakcję ze środowiskiem.

W tym kodzie:

| Pojęcie | Znaczenie w implementacji |
|---|---|
| `state` | Obraz z kamery po normalizacji, prędkość, manewr trasy. |
| `action` | Jedna z 10 dyskretnych komend pojazdu. |
| `reward` | Liczba mówiąca, czy krok był dobry. |
| `done` | Czy epizod się skończył. |
| `episode` | Jedna próba przejazdu od startu do końca, kolizji, zejścia z trasy albo limitu kroków. |
| `policy` | Rozkład prawdopodobieństwa po akcjach. |
| `value` | Oszacowanie, jak dobry jest aktualny stan. |

Agent nie dostaje gotowej instrukcji "skręć tutaj". Dostaje obserwację i reward,
a przez wiele prób uczy się, które akcje zwiększają oczekiwany zwrot.

### 6.2 Actor-Critic

Actor-Critic łączy dwa komponenty:

- actor: wybiera akcję, czyli modeluje `pi(a|s)`,
- critic: ocenia stan, czyli modeluje `V(s)`.

W tej implementacji actor i critic są w jednej sieci:

```text
obraz + prędkość + manewr
        |
   wspólny trunk
        |
  +-----+------+
  |            |
policy      value
logits      V(s)
```

`policy` zwraca logits dla 10 akcji. Logits to surowe liczby, które
`torch.distributions.Categorical(logits=logits)` zamienia wewnętrznie na
prawdopodobieństwa przez softmax.

`value` zwraca jedną liczbę: przewidywany przyszły zwrot ze stanu.

### 6.3 Return n-step

Worker nie czeka zawsze do końca epizodu. Zbiera rollout o długości
`rollout_length`, domyślnie 20 kroków. Potem liczy zwroty od końca:

```text
R_t = r_t + gamma * R_{t+1}
```

Jeżeli rollout kończy epizod, końcowe `R` startuje od zera.

Jeżeli rollout nie kończy epizodu, końcowe `R` startuje od wartości critica dla
ostatniego stanu:

```text
R_last = V(s_last)
```

To jest bootstrap.

### 6.4 Advantage

Advantage mówi, czy akcja była lepsza niż critic się spodziewał:

```text
A_t = R_t - V(s_t)
```

Jeżeli `A_t > 0`, akcja była lepsza od oczekiwań i policy powinno zwiększyć jej
prawdopodobieństwo. Jeżeli `A_t < 0`, akcja była gorsza i policy powinno je
zmniejszyć.

Domyślnie advantages są normalizowane w obrębie rollouta:

```text
A_norm = (A - mean(A)) / max(std(A), 1e-8)
```

To stabilizuje skalę policy loss.

### 6.5 Loss

Implementacja liczy trzy składniki:

```text
policy_loss = -mean(log_prob(action_t) * advantage_t)
value_loss  = value_loss_coef * smooth_l1_loss(V(s_t), R_t)
entropy     = mean(entropy(policy_t))

total_loss  = policy_loss + value_loss - entropy_coef * entropy
```

Minus przy entropii jest celowy: minimalizacja `total_loss` będzie zwiększać
entropię, czyli zachęcać politykę do eksploracji.

### 6.6 Entropy annealing

`entropy_coef` nie musi być stałe. Domyślnie:

```text
beta_start = 0.02
beta_end = 0.002
beta_anneal_frac = 0.6
steps = 10_000_000
```

W pierwszych 60% budżetu kroków współczynnik entropii maleje liniowo z 0.02 do
0.002. Na początku agent ma więcej eksplorować. Później ma bardziej wykorzystywać
to, czego się nauczył.

## 7. Mapa plików aktualnej implementacji

```text
new_hogwild_train.slurm
  |
  +-- startuje CARLA servers
  +-- uruchamia python -u new_hogwild_train_a3c_carla.py

new_hogwild_train_a3c_carla.py
  |
  +-- parsuje argumenty
  +-- ustawia seedy
  +-- tworzy GlobalNetwork
  +-- ładuje checkpoint przy --resume
  +-- startuje RunMonitor i W&B logger
  +-- wywołuje run_with_restart(...)

new_hogwild_run_a3c.py
  |
  +-- startuje workery A3CWorker
  +-- monitoruje, czy żyją
  +-- restartuje padnięte workery
  +-- przy szybkich crashach robi rollback globalnego modelu

new_hogwild_a3c.py
  |
  +-- SharedActorCritic
  +-- SharedRMSprop / SharedAdam
  +-- GlobalNetwork
  +-- A3CWorker
  +-- obliczanie lossów i update globalnego modelu

new_hogwild_carla_wrapper.py
  |
  +-- reset() i step() w formacie wygodnym dla A3C
  +-- normalizacja obserwacji
  +-- action repeat
  +-- reward shaping albo legacy reward
  +-- zapis klatek epizodu

carla_env.py
  |
  +-- połączenie z CARLA
  +-- mapa Town03
  +-- scenariusze, spawn, goal
  +-- route planner
  +-- sensory kamery/kolizji/lane invasion
  +-- niskopoziomowe sterowanie pojazdem
```

## 8. Uruchomienie przez SLURM

`new_hogwild_train.slurm` jest pełnym launcherem. Robi więcej niż samo
uruchomienie Pythona.

### 8.1 Zasoby joba

Nagłówek SLURM ustawia m.in.:

```text
partition: plgrid-gpu-a100
nodes: 1
ntasks-per-node: 1
cpus-per-task: 7
mem: 25G
time: 20:50:00
gpus: 1
job-name: new-hogwild-a3c-carla
signal: SIGUSR1@90
```

`SIGUSR1@90` oznacza, że SLURM wyśle sygnał 90 sekund przed końcem limitu czasu.
Skrypt ma trap na `SIGUSR1`, więc próbuje zamknąć trening łagodnie.

### 8.2 Argumenty launchera

Najważniejsze argumenty:

| Argument | Znaczenie |
|---|---|
| `-w`, `--workers` | Liczba workerów A3C i serwerów CARLA. |
| `--workers-per-gpu` | Ilu learner workerów przypisać do jednego GPU. |
| `--servers-per-gpu` | Ilu serwerów CARLA przypisać do jednego GPU. |
| `-s`, `--scenario` | Numer scenariusza CARLA. |
| `--resume DIR` | Wznów run z katalogu `DIR`. |
| `--outdir DIR` | Użyj konkretnego katalogu wyników. |
| `--steps NUM` | Nadpisz budżet kroków Pythona. |
| `--no-wandb` | Wyłącz W&B. |
| `--testing` | Tryb testowy: akcje deterministyczne, brak update'u. |
| `--no-carla` | Nie startuj CARLA, zakładaj że serwery już działają. |
| `--` | Wszystko po `--` trafia bezpośrednio do skryptu Pythona. |

Przykład:

```bash
sbatch --gpus=6 new_hogwild_train.slurm -w 6 --workers-per-gpu 1 --servers-per-gpu 1
```

### 8.3 Katalog wyników

Dla świeżego runa skrypt tworzy domyślnie:

```text
AV/A_to_B_GPU_34/runs/a3c_hogwild_<workers>w_<timestamp>_<jobid>
```

W środku zapisuje:

- `a3c_training.log` - stdout/stderr Pythona,
- `gpu_dmon.log` - monitoring `nvidia-smi dmon`,
- `carla_servers.log` - log supervisora CARLA,
- `args.txt` - argumenty Pythona,
- `logs/` - JSONL logger,
- `checkpoint.pth`,
- `checkpoint_step.txt`,
- `resume_state.json`,
- opcjonalnie `best_checkpoint.pth`,
- opcjonalnie `episodes/...` z zapisanymi klatkami.

### 8.4 Start serwerów CARLA

Skrypt uruchamia zewnętrzny:

```text
/net/tscratch/people/plgbartoszkawa/carla_athena_multiserver_v3.py
```

Uwaga: w repozytorium `AV/` jest `carla_athena_multiserver.py`, ale launcher
SLURM wskazuje na `carla_athena_multiserver_v3.py` poza katalogiem `AV/`.
Dokumentowany `new_hogwild_train.slurm` zależy więc od pliku zewnętrznego
względem `AV/`.

Po starcie CARLA skrypt czeka, aż wszystkie porty będą w stanie `LISTEN`.
Jeśli porty nie wstaną, wypisuje ostatnie linie logu CARLA i kończy job.

W analizowanym logu `AV/new-hogwild-a3c-carla-log-2652108.txt` widać realny
problem środowiskowy: skrypt próbował użyć venv
`/net/tscratch/people/plgbartoszkawa/venv`, którego nie było, a potem supervisor
CARLA padł przez brak modułu `psutil`. W `requirements.txt` `psutil` istnieje,
więc problem dotyczył aktywnego środowiska uruchomieniowego, nie samej listy
zależności.

## 9. Główne wejście Pythona: `new_hogwild_train_a3c_carla.py`

Ten plik jest entrypointem treningu. Nie wykonuje samego rollouta; organizuje
konfigurację, procesy i lifecycle runa.

### 9.1 Domyślne wartości

Najważniejsze domyślne wartości:

| Nazwa | Wartość | Znaczenie |
|---|---:|---|
| `DEFAULT_NUM_WORKERS` | `2` | Domyślna liczba workerów Pythona, jeśli nie nadpisze jej SLURM. |
| `DEFAULT_WORKERS_PER_GPU` | `2` | Domyślnie 2 workery na GPU. |
| `DEFAULT_START_PORT` | `2000` | Pierwszy port CARLA. |
| `DEFAULT_PORT_STEP` | `100` | Odstęp między portami. |
| `DEFAULT_SCENARIO` | `14` | Scenariusz jazdy. |
| `DEFAULT_CAMERA` | `semantic` | Typ kamery. |
| `DEFAULT_RES` | `250` | Rozdzielczość obrazu wejściowego. |
| `DEFAULT_OPTIMIZER` | `shared-rmsprop` | Domyślny optymalizator. |
| `DEFAULT_ROLLOUT_LENGTH` | `20` | Długość rollouta przed update'em. |
| `DEFAULT_GAMMA` | `0.99` | Discount factor. |
| `DEFAULT_LR` | `1e-4` | Learning rate startowy. |
| `DEFAULT_BETA_START` | `0.02` | Entropy coefficient na początku. |
| `DEFAULT_BETA_END` | `0.002` | Entropy coefficient na końcu annealingu. |
| `DEFAULT_BETA_ANNEAL_FRAC` | `0.6` | Przez jaką część treningu annealować entropię. |
| `DEFAULT_MAX_GRAD_NORM` | `5.0` | Gradient clipping. |
| `DEFAULT_NORMALIZE_ADVANTAGES` | `True` | Normalizacja advantage. |
| `DEFAULT_STEPS` | `10_000_000` | Budżet globalnych kroków. |
| `DEFAULT_SYNC_EVERY_N_UPDATES` | `1` | Synchronizacja lokalnego modelu po każdym update. |
| `DEFAULT_HOGWILD_LOCK_UPDATES` | `False` | Domyślnie update bez locka. |
| `DEFAULT_ACTION_REPEAT` | `2` | Jedna decyzja agenta = 2 ticki CARLA z tą samą akcją. |
| `DEFAULT_EPISODE_MAX_DECISIONS` | `100` | Wrapper kończy epizod po 100 decyzjach. |
| `DEFAULT_REWARD_MODE` | `legacy` | Domyślnie reward z `utils.reward_function`. |

Ważna różnica: `settings.py` też ma wartości takie jak `GAMMA`, `LR`,
`SCENARIO`, ale `new_hogwild_a3c.py` nie czyta `settings.py` bezpośrednio.
Entry point przekazuje konfigurację przez `SimpleNamespace`. `settings.py` nadal
jest używany pośrednio przez `carla_env.py`.

### 9.2 Argumenty CLI

`build_parser()` wystawia argumenty runa. Część wartości jest normalnie
nadpisywana przez SLURM:

- liczba workerów,
- porty,
- scenariusz,
- katalog output,
- resume,
- logging,
- W&B.

Rzadziej zmieniane parametry algorytmu też są dostępne:

- `--optimizer`,
- `--rollout-length`,
- `--gamma`,
- `--lr`,
- `--beta`,
- `--beta-start`,
- `--beta-end`,
- `--beta-anneal-frac`,
- `--value-loss-coef`,
- `--max-grad-norm`,
- `--no-normalize-advantages`.

`--beta` jest skrótem na stałą entropię: jeżeli podasz `--beta`, kod ustawi
`beta_start = beta_end = beta`.

### 9.3 Seedy

Proces główny ustawia:

```python
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
```

Każdy worker potem dodaje swój offset:

```python
seed = config.seed + worker_id * 1009
```

Dzięki temu workery nie mają identycznej sekwencji losowania, mimo że run ma
jeden globalny seed.

### 9.4 Przypisanie GPU

`_assign_worker_gpus(num_workers, workers_per_gpu, worker_gpu_start)`:

- jeśli nie ma GPU albo `workers_per_gpu <= 0`, zwraca `cpu`,
- jeśli jest jedno GPU, wszyscy dostają `cuda:0`,
- jeśli jest kilka GPU, tworzy listę typu:

```text
cuda:0, cuda:0, cuda:1, cuda:1, ...
```

liczba powtórzeń wynika z `workers_per_gpu`.

Przykład: 4 workery, 2 workery/GPU, 2 GPU:

```text
worker_gpus = ["cuda:0", "cuda:0", "cuda:1", "cuda:1"]
```

### 9.5 Start multiprocessing

Kod ustawia:

```python
mp.set_start_method('spawn', force=True)
```

`spawn` uruchamia każdy proces od świeżego interpretera Pythona. To jest bardziej
bezpieczne z CUDA niż `fork`, bo `fork` może skopiować niepoprawny stan runtime
CUDA.

Konsekwencja: obiekty przekazywane workerom muszą dać się spicklować, a tensory,
które mają być naprawdę wspólne, muszą mieć shared memory.

### 9.6 Tworzenie `GlobalNetwork`

Entry point tworzy:

```python
global_network = GlobalNetwork(
    config,
    state_shape=[args.res, args.res, 3],
    action_shape=args.n_actions,
    critic_shape=1,
)
```

`args.n_actions` jest równe `len(ac.ACTIONS_NAMES)`, czyli 10.

### 9.7 Resume

Jeżeli podano `--resume`, kod:

1. szuka najnowszego checkpointu przez `find_latest_checkpoint(args.resume)`,
2. ładuje go przez `global_network.load(checkpoint_path)`,
3. czyta `resume_state.json`,
4. dolicza poprzedni aktywny czas treningu,
5. ostrzega, jeśli ważne parametry zmieniły się względem zapisanego runa.

Ostrzeżenia są dla kluczy:

```text
steps, lr, beta_start, beta_end, beta_anneal_frac,
rollout_length, gamma, weight_decay, optimizer
```

### 9.8 W&B

Jeżeli `wandb` jest zainstalowany i nie ma `--no-wandb`, entry point tworzy
osobny proces `wandb_logger_process`. Workery wrzucają wybrane rekordy epizodów
do `mp.Queue`, a proces W&B je publikuje.

Ważne: W&B dostaje głównie rekordy epizodów z `_log_episode`, nie pełne logi
update'ów. Pełne logi update'ów są w JSONL na dysku.

## 10. Globalny model i optymalizator: `GlobalNetwork`

`GlobalNetwork` znajduje się w `new_hogwild_a3c.py`.

### 10.1 Co przechowuje

`GlobalNetwork` ma:

- `model` - `SharedActorCritic` na CPU,
- `optimizer` - `SharedRMSprop` albo `SharedAdam`,
- `update_lock` - opcjonalny lock na update,
- `stats_lock` - lock na statystyki epizodów,
- `save_lock` - lock na zapis/odczyt checkpointu,
- `global_step` - liczba globalnych kroków środowiska,
- `global_episode` - liczba globalnych epizodów,
- `total_updates` - liczba update'ów optymalizatora,
- `last_checkpoint_boundary` - ostatnia granica checkpointu,
- `best_reward` - najlepszy reward epizodu,
- `global_mean_reward` - średni reward z bufora ostatnich epizodów,
- `worker_mean_rewards` - średnie rewardy per worker,
- `recent_rewards` - bufor rewardów epizodów.

### 10.2 Dlaczego globalny model jest na CPU

PyTorch multiprocessing potrafi współdzielić tensory CPU przez shared memory.
Współdzielenie globalnego modelu na GPU między procesami jest trudniejsze,
bardziej kruche i zwykle wymaga innej architektury. Tutaj przyjęto prosty model:

```text
globalny model CPU shared memory
lokalny model worker cuda/cpu
gradienty kopiowane lokalny -> globalny CPU
optimizer.step() na globalnym CPU
```

Koszt: kopiowanie gradientów GPU -> CPU.

Zysk: prosty Hogwild z wieloma procesami i wspólnym stanem optymalizatora.

### 10.3 Liczniki

`global_step` jest zwiększany przed każdym krokiem środowiska workera:

```python
global_t = self.global_network.increment_global_step(1)
```

To oznacza, że przy 4 workerach licznik rośnie szybciej ściennie, ale nadal
reprezentuje realną liczbę decyzji agenta łącznie ze wszystkich workerów.

`global_episode` rośnie na początku każdego epizodu workera.

`total_updates` rośnie po każdym udanym `optimizer.step()`.

### 10.4 Learning rate decay

`set_lr_for_step(global_t)` robi liniowy decay LR od `config.lr` do zera:

```text
lr = ((steps - global_t - 1) / steps) * initial_lr
```

Jeżeli `config.steps <= 0`, LR pozostaje stały.

W praktyce LR jest ustawiany tuż przed update'em globalnego optymalizatora.
Bez locka kilka workerów może nadpisać LR prawie równocześnie, ale ponieważ
wszyscy bazują na rosnącym `global_t`, różnice są małe.

### 10.5 Save/load

Checkpoint zawiera:

- globalne liczniki,
- statystyki rewardów,
- `model.state_dict()`,
- `optimizer.state_dict()`.

Format jest nowy i wymaga klucza `model`. Kod jawnie odrzuca starszy format
checkpointów z osobnym `DiscreteActor + Critic`.

Zapis checkpointu jest zabezpieczony przed NaN:

```text
jeśli globalne parametry mają NaN -> nie zapisuj
```

## 11. Architektura sieci `SharedActorCritic`

`SharedActorCritic` jest wspólną siecią actora i critica.

### 11.1 Wejścia

Sieć przyjmuje:

```python
forward(self, x, speed=None, maneuver=None)
```

Gdzie:

| Argument | Kształt | Znaczenie |
|---|---|---|
| `x` | `[B, 3, H, W]` | Obraz z kamery, float32, wartości `[0,1]`. |
| `speed` | `[B, 1]` | Prędkość znormalizowana przez `/100`. |
| `maneuver` | `[B]` | Manewr trasy: 0 left, 1 straight, 2 right. |

Jeżeli `speed` jest `None`, sieć używa zera. Jeżeli `maneuver` jest `None`,
używa domyślnie `1`, czyli straight. Manewr jest clampowany do zakresu
`0..num_maneuvers-1` i one-hot encode'owany.

### 11.2 CNN

Część obrazowa:

```text
Conv2d(3 -> 32, kernel=5, stride=2, padding=2)
ReLU
Conv2d(32 -> 64, kernel=3, stride=2, padding=1)
ReLU
Conv2d(64 -> 128, kernel=3, stride=2, padding=1)
ReLU
Conv2d(128 -> 256, kernel=3, stride=2, padding=1)
ReLU
AdaptiveAvgPool2d((4, 4))
```

Dla obrazu `250x250` przybliżone rozmiary map cech:

```text
[B, 3, 250, 250]
 -> [B, 32, 125, 125]
 -> [B, 64, 63, 63]
 -> [B, 128, 32, 32]
 -> [B, 256, 16, 16]
 -> [B, 256, 4, 4]
 -> flatten [B, 4096]
```

`AdaptiveAvgPool2d((4,4))` sprawia, że trunk liniowy ma stały rozmiar wejścia
nawet gdy rozdzielczość obrazu się zmieni.

### 11.3 Gałąź prędkości

```text
Linear(1 -> 32)
ReLU
```

Prędkość pochodzi z `CarlaEnv.calculate_speed()` w km/h i wrapper dzieli ją
przez 100. Przykład: 20 km/h -> `0.2`.

### 11.4 Gałąź manewru

Manewr jest najpierw one-hot:

```text
0 -> [1, 0, 0]  left
1 -> [0, 1, 0]  straight
2 -> [0, 0, 1]  right
```

Potem:

```text
Linear(3 -> 32)
ReLU
```

### 11.5 Wspólny trunk

Cechy są konkatenowane:

```text
4096 image features + 32 speed features + 32 maneuver features = 4160
```

Potem:

```text
Linear(4160 -> 512)
ReLU
Linear(512 -> 256)
ReLU
```

### 11.6 Głowy policy i value

```text
policy: Linear(256 -> n_actions)
value:  Linear(256 -> 1)
```

Dla `n_actions = 10` wyjścia mają kształty:

```text
logits: [B, 10]
value:  [B, 1]
```

### 11.7 Brak BatchNorm i Dropout

W starszym `nets/a2c.py` są BatchNorm i Dropout. Nowy `SharedActorCritic` ich
nie używa. To ma sens dla A3C, bo worker wykonuje forward zwykle z batch size 1.
BatchNorm przy batch size 1 bywa niestabilny i zależny od trybu train/eval.
Dropout dodawałby dodatkowy szum do już asynchronicznego treningu.

## 12. Worker A3C: lokalny model, rollout, akcja, loss, gradienty

`A3CWorker` dziedziczy po `mp.Process`. Każdy worker jest osobnym procesem.

### 12.1 Inicjalizacja

Worker dostaje:

- `worker_id`,
- referencję do `global_network`,
- `config`,
- `port`,
- `device`, np. `cuda:0`,
- `run_output_dir`,
- `shutdown_event`,
- opcjonalną `log_queue`,
- `run_id`.

W `_init_networks()` tworzy lokalny model:

```python
self.model = SharedActorCritic(..., self.device).to(self.device)
self.sync_with_global()
```

Lokalny model nie jest współdzielony. To prywatna kopia workera.

### 12.2 Synchronizacja z globalnym modelem

`sync_with_global()` kopiuje wartości parametrów:

```python
local_param.data.copy_(global_param.data.to(self.device, non_blocking=True))
```

Po kopiowaniu na CUDA kod robi `torch.cuda.synchronize(self.device)`.

Synchronizacja dzieje się:

- na początku epizodu,
- po update, jeśli `local_updates % sync_every_n_updates == 0`,
- po wykryciu NaN gradientów.

Domyślnie `sync_every_n_updates = 1`, więc worker odświeża lokalne wagi po
każdym update.

### 12.3 Wybór akcji

`get_action(obs, speed, maneuver, testing=False)`:

1. Robi forward lokalnego modelu:

```python
logits, value = self.model(obs, speed, maneuver)
```

2. Tworzy rozkład:

```python
action_distribution = Categorical(logits=logits)
```

3. W trybie treningowym losuje:

```python
action = action_distribution.sample()
```

4. W trybie testowym wybiera argmax:

```python
action = action_distribution.probs.argmax(dim=-1)
```

5. Liczy:

```python
log_prob = action_distribution.log_prob(action)
entropy = action_distribution.entropy()
```

6. W treningu zapisuje `Transition(value, log_prob, entropy, action)`.

Ważne: `Transition` nie zapisuje całego stanu, tylko wartości potrzebne do
lossu. Stany nie są trzymane do replay buffer. A3C działa on-policy.

### 12.4 Główna pętla workera

W `run()` worker:

1. Ignoruje `SIGINT`, żeby proces główny zarządzał zatrzymaniem.
2. Ogranicza liczbę wątków BLAS/OpenMP do 1.
3. Ustawia seedy.
4. Ustawia CUDA device, jeśli dotyczy.
5. Tworzy lokalny model.
6. Tworzy `TrainingLogger`.
7. Tworzy `WorkerMonitor`.
8. Tworzy `CarlaA3CWrapper`.
9. W pętli:
   - czyści bufor rollouta,
   - zwiększa globalny numer epizodu,
   - synchronizuje model lokalny z globalnym,
   - resetuje środowisko,
   - wykonuje kroki aż do `done` albo shutdown,
   - co `rollout_length` albo na końcu epizodu liczy gradienty i update,
   - zapisuje checkpointy na granicach `save_frequency`,
   - loguje epizod.

### 12.5 Krok środowiska w workerze

W każdym kroku:

```python
global_t = increment_global_step(1)
state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
speed_tensor = torch.tensor([[speed]], dtype=torch.float32, device=device)
maneuver_tensor = torch.tensor([maneuver], device=device)
action, value_f, entropy_f = get_action(...)
next_state, next_speed, next_maneuver, reward, done, info = env.step(action)
```

`state` jest numpy array `[3,H,W]`, więc po `unsqueeze(0)` ma `[1,3,H,W]`.

Reward i komponenty rewardu są zapisywane:

```python
self.rewards.append(reward)
self.reward_components.append(info.get('reward_components', {}))
```

### 12.6 Obliczanie gradientów

`compute_and_apply_gradients(...)` wykonuje cały update.

Jeśli rollout jest pusty, zwraca `None`.

#### Krok 1: bootstrap

Jeżeli epizod się skończył:

```text
discounted_return = 0
```

Jeżeli nie:

```text
discounted_return = V(final_state)
```

To dzieje się w `torch.no_grad()`, bo bootstrap jest targetem, a nie częścią
grafu uczenia.

#### Krok 2: returns

Kod idzie po rewardach od końca:

```python
discounted_return = r + gamma * discounted_return
returns.insert(0, discounted_return)
```

`reward_scale = 0` oznacza "nie skaluj". Jeżeli `reward_scale` jest niezerowe,
reward jest dzielony przez tę wartość.

#### Krok 3: tensory rollouta

Kod robi:

```python
returns_tensor = [T, 1]
values_tensor = [T, 1]
log_probs_tensor = [T]
entropies_tensor = [T]
```

gdzie `T` to długość rollouta.

#### Krok 4: advantage

```python
advantages_tensor = returns_tensor - values_tensor.detach()
```

Do policy loss używa się ewentualnie znormalizowanej kopii advantage.

#### Krok 5: loss

```python
policy_loss = -(log_probs_tensor * policy_advantages.view(-1)).mean()
value_loss = value_loss_coef * smooth_l1_loss(values_tensor, returns_tensor)
entropy_mean = entropies_tensor.mean()
total_loss = policy_loss + value_loss - entropy_coef * entropy_mean
```

#### Krok 6: backward

```python
self.model.zero_grad()
total_loss.backward()
```

Gradienty pojawiają się w parametrach lokalnego modelu.

#### Krok 7: gradient clipping

Jeżeli `max_grad_norm > 0`:

```python
clip_grad_norm_(self.model.parameters(), max_grad_norm)
```

Domyślnie limit to `5.0`.

#### Krok 8: NaN guard

`has_nan_grads(self.model)` sprawdza, czy któryś gradient ma NaN.

Jeżeli tak:

- logger zapisuje event `nan_gradient`,
- update jest pomijany,
- worker synchronizuje się z globalnym modelem,
- rollout jest czyszczony.

To zabezpiecza globalny model przed bezpośrednim skażeniem NaN gradientem.

#### Krok 9: update globalny

Kod może użyć locka:

```python
update_ctx = update_lock if hogwild_lock_updates else nullcontext()
```

Domyślnie `hogwild_lock_updates = False`, więc locka nie ma.

W update:

1. ustaw LR dla aktualnego `global_t`,
2. skopiuj lokalne gradienty do globalnego modelu,
3. wykonaj `optimizer.step()`,
4. wyzeruj gradienty globalnego optymalizatora/modelu.

Po update:

- `total_updates` rośnie,
- logowany jest rekord update,
- rollout jest czyszczony.

## 13. Hogwild w tej implementacji

Hogwild oznacza update bez pełnej synchronizacji workerów.

### 13.1 Co jest wspólne

Współdzielone między procesami:

- storage parametrów globalnego modelu,
- storage stanu optymalizatora,
- liczniki `mp.Value`,
- tablice statystyk `mp.Array`.

Niewspółdzielone:

- lokalny model workera,
- lokalne rollouty,
- lokalne gradienty,
- `.grad` globalnego modelu jako atrybut procesu.

### 13.2 Jak gradient trafia do globalnego modelu

`transfer_local_gradients_to_global(...)` iteruje po parach parametrów:

```python
global_param.grad = local_param.grad.detach().to(cpu).clone()
```

Komentarz w kodzie podkreśla, że `.grad` jest celowo proces-lokalny. Gdyby
gradienty były współdzielone, workery mogłyby nadpisywać sobie gradienty przed
`optimizer.step()`.

### 13.3 Co może się ścigać

Bez `hogwild_lock_updates` dwa workery mogą jednocześnie:

- ustawiać LR optymalizatora,
- robić `optimizer.step()`,
- modyfikować parametry globalnego modelu,
- modyfikować shared state optymalizatora.

To jest świadoma decyzja. A3C akceptuje dodatkowy szum i brak idealnej kolejności
update'ów, bo dzięki temu workery nie czekają na siebie.

### 13.4 Kiedy można włączyć lock

`--hogwild-lock-updates` powoduje, że sekcja update'u jest objęta `mp.Lock`.
To zmniejsza ryzyko race condition i ułatwia debugowanie, ale może obniżyć
przepustowość, bo workery czekają w kolejce do update'u.

## 14. Wrapper CARLA: obserwacje, akcje, reward i koniec epizodu

`new_hogwild_carla_wrapper.py` ukrywa szczegóły `CarlaEnv` przed workerem.
Worker widzi tylko:

```python
reset() -> (state, speed, maneuver)
step(action) -> (next_state, next_speed, next_maneuver, reward, done, info)
reconnect()
is_server_alive()
```

### 14.1 Połączenie z CARLA

Wrapper tworzy `CarlaEnv` przez `_connect_with_retries()`:

- próbuje maksymalnie `max_connect_retries`,
- między próbami śpi `connect_retry_wait`,
- przy ostatniej porażce rzuca wyjątek.

`reconnect()` czyści referencje do `world` i `client`, czeka
`reconnect_wait`, potem tworzy nowe `CarlaEnv`.

### 14.2 Normalizacja obserwacji

`_state_to_chw_float(state)` akceptuje numpy array albo tensor. Robi:

1. jeżeli input ma batch `[1,...]`, usuwa batch,
2. jeżeli input jest `HWC`, transponuje do `CHW`,
3. wymusza `float32`,
4. sprawdza, że kształt to `[3,resY,resX]`,
5. sprawdza brak NaN/inf,
6. jeżeli max > 1, dzieli przez 255,
7. sprawdza zakres `[0,1]`,
8. zwraca tablicę contiguous.

To jest ważne, bo `CarlaEnv` zwraca obraz jako tensor PyTorch z batch dimension
`[1,3,H,W]` i wartościami zwykle 0..255. Wrapper zamienia go na numpy
`[3,H,W]` w zakresie 0..1.

### 14.3 Prędkość

`CarlaEnv.calculate_speed()` zwraca km/h. Wrapper dzieli przez 100:

```text
speed_f = speed_kmh / 100
```

W logach `speed_kmh` nadal jest w km/h.

### 14.4 Manewr

`CarlaEnv.plan_the_route()` wyciąga z route plannera decyzje:

```text
RoadOption.LEFT     -> 0
RoadOption.STRAIGHT -> 1
RoadOption.RIGHT    -> 2
```

Wrapper trzyma aktualny manewr w `_current_maneuver`. Przy wyjściu ze
skrzyżowania `_update_maneuver()` przechodzi do następnej decyzji z listy.

Sieć dostaje ten manewr jako dodatkowy sygnał wejściowy.

### 14.5 Action repeat

W `step(action)` wrapper wykonuje:

```python
for _ in range(self._action_repeat):
    self.env.step_apply_action(int(action))
    self.env.world.tick()
```

Domyślnie `action_repeat = 2`. To znaczy, że jedna decyzja agenta jest utrzymana
przez 2 ticki świata CARLA. Przy `fixed_delta_seconds = 0.1` jedna decyzja trwa
około 0.2 sekundy symulacji.

Po tych tickach wrapper wywołuje `self.env.step(...)`, które liczy reward,
done, route distance, prędkość i pobiera obraz.

### 14.6 Koniec epizodu w wrapperze

Wrapper może zakończyć epizod przez:

- `done` z `CarlaEnv`,
- limit `episode_max_decisions`, domyślnie 100.

To jest dodatkowy limit ponad `settings.STEP_COUNTER = 200` w `CarlaEnv`.
W praktyce dla `new_hogwild_*` limit wrappera 100 decyzji zwykle zadziała
wcześniej niż limit `CarlaEnv`.

## 15. Niskopoziomowe środowisko `CarlaEnv`

`carla_env.py` jest długim plikiem i pochodzi ze starszego projektu, ale
`new_hogwild_*` nadal go używa przez wrapper.

### 15.1 Inicjalizacja

`CarlaEnv.__init__`:

1. łączy się z `carla.Client("localhost", port)`,
2. ustawia timeout klienta na 120 s,
3. porównuje wersję klienta i serwera,
4. ładuje świat `Town03`,
5. ustawia synchronizację świata,
6. wybiera scenariusz,
7. tworzy spawn point i goal,
8. planuje trasę,
9. tworzy action space,
10. przygotowuje sensory i zmienne epizodu.

### 15.2 Scenariusze

Kod obsługuje scenariusze 1-16. Dla aktualnego domyślnego `scenario=14`
używana jest lista:

```python
MAP_POINTS_SC14 = [
    (28, 155), (49, 129), (83, 89), (77, 98), (54, 234)
]
```

Każda para oznacza indeks spawn pointu i indeks goal pointu na mapie CARLA.
Kod przechodzi cyklicznie po tej liście przez `goal_points_index`.

### 15.3 Planowanie trasy

`plan_the_route()`:

1. tworzy `GlobalRoutePlannerDAO`,
2. tworzy `GlobalRoutePlanner`,
3. wybiera goal zależnie od scenariusza,
4. wywołuje `trace_route(spawn_point, goal_location)`,
5. usuwa duplikaty waypointów,
6. wyciąga decyzje skrętu,
7. rysuje/wyznacza middle goals.

`car_decisions` to lista manewrów dla sieci. Na końcu kod dodaje `1`, czyli
straight, jako domyślny manewr po wykonaniu zaplanowanych skrętów.

### 15.4 Spawn pojazdu

`spawn_car()` wybiera blueprint `model3`, ustawia `role_name='ego'` i próbuje
`try_spawn_actor` do 10 razy. Jeżeli nie uda się zespawnować pojazdu, rzuca
`RuntimeError`.

### 15.5 Kamery

Kod obsługuje:

- RGB camera,
- semantic segmentation camera,
- depth camera, ale depth nie jest używana w `reset()`.

Dla `camera='semantic'` używa:

```text
sensor.camera.semantic_segmentation
CityScapesPalette
```

Obraz po przetworzeniu jest zapisywany w `self.front_camera` jako tensor:

```text
[1, 3, resY, resX]
```

### 15.6 Sensory kolizji i lane invasion

`add_collision_sensor()` zapisuje eventy do `collision_history_list`.

`add_line_invasion_sensor()` zapisuje eventy do `invasion_history_list`.

W `step()` lane invasion jest sprowadzone do `invasion_counter = 1`, jeśli
lista nie jest pusta. Potem lista jest czyszczona. To ogranicza wpływ wielu
eventów z jednej faktycznej inwazji pasa.

### 15.7 Reset

`CarlaEnv.reset(episode, save_image=False, reload_world=True)`:

1. robi `reload_world()` albo `reset_episode_state()`,
2. losuje scenariusz z listy,
3. tworzy scenariusz,
4. planuje trasę,
5. spawnuje samochód,
6. ustawia spectator,
7. dodaje kamerę,
8. dodaje sensory kolizji i lane invasion,
9. aplikuje akcję hamowania `3`,
10. wykonuje 15 ticków świata,
11. czyści kolejkę obrazów,
12. pobiera najnowszą klatkę,
13. przetwarza obraz,
14. zwraca `(front_camera, speed)`.

Wrapper przekazuje `reload_world=full_reload`, gdzie `full_reload` zależy od
`world_reload_interval`. Domyślnie `world_reload_interval=0`, więc pełny reload
świata nie jest robiony co epizod; używany jest lżejszy reset epizodu.

### 15.8 Step

`CarlaEnv.step(...)`:

1. liczy odległość od celu,
2. liczy odległość od zaplanowanej trasy,
3. liczy prędkość,
4. sprawdza middle point reward i terminal point,
5. liczy lane invasion counter,
6. wywołuje `utils.reward_function(...)`,
7. aktualizuje `prev_speed`,
8. kończy epizod przy kolizji/offroute/dotarciu/limicie kroków,
9. pobiera obraz z kamery,
10. przetwarza obraz,
11. zwraca:

```text
front_camera, reward, done, route_distance, speed_value, distance_from_goal
```

## 16. Akcje dyskretne

`ACTIONS.py` definiuje 10 akcji:

| Indeks | Nazwa | throttle | brake | steer |
|---:|---|---:|---:|---:|
| 0 | `forward` | 0.5 | 0 | 0 |
| 1 | `forward_left` | 0.5 | 0 | -0.5 |
| 2 | `forward_right` | 0.5 | 0 | 0.5 |
| 3 | `brake` | 0 | 1 | 0 |
| 4 | `brake_left` | 0 | 1 | -0.5 |
| 5 | `brake_right` | 0 | 1 | 0.5 |
| 6 | `forward_slight_left` | 0.5 | 0 | -0.2 |
| 7 | `forward_slight_right` | 0.5 | 0 | 0.2 |
| 8 | `brake_slight_left` | 0 | 1 | -0.2 |
| 9 | `brake_slight_right` | 0 | 1 | 0.2 |

`CarlaEnv.car_control_discrete(action)` mapuje indeks akcji na
`carla.VehicleControl`.

Wartości steer:

- ujemne: skręt w lewo,
- dodatnie: skręt w prawo,
- zero: prosto.

## 17. Reward: legacy i shaped

Wrapper ma dwa tryby:

```text
reward_mode = "legacy" albo "shaped"
```

Domyślnie w entry poincie jest `legacy`.

### 17.1 Legacy reward

W trybie `legacy` wrapper bierze reward zwrócony przez `CarlaEnv.step()` i
przepuszcza go bez zmiany.

`CarlaEnv.step()` liczy reward przez `utils.reward_function(...)`.

Ta funkcja:

1. kończy epizod przy kolizji lub `route_distance >= 10`,
2. dodaje reward/karę za lane invasion według `settings.REWARD_FROM_INV`,
3. liczy `speed_reward`:

```text
speed_reward = -1.2 + 4 * sin(speed / 10)
```

4. liczy reward za trzymanie trasy:

```text
if route_distance < 1.5:
    route_distance_reward = 1
    if on_junction and speed_reward > 0:
        route_distance_reward *= 4
else:
    route_distance_reward = -4 * sin(speed / 10)
```

5. sumuje:

```text
reward = terminal_state_reward
       + col_reward
       + speed_reward
       + route_distance_reward
       + inv_reward
       + mp_static_reward
```

W `settings.py` wartości statyczne rewardów są obecnie ustawione na 0:

```text
REWARD_FROM_TP = 0
REWARD_FROM_MP = 0
REWARD_FROM_COL = 0
REWARD_FROM_INV = 0
```

To znaczy, że główny sygnał w legacy mode pochodzi z prędkości i odległości od
trasy, a terminal/collision/invasion wpływają głównie przez `done`, nie przez
dodatnią/ujemną stałą rewardu.

### 17.2 Shaped reward

W trybie `shaped` wrapper liczy reward sam:

| Składnik | Znaczenie |
|---|---|
| `progress` | Poprawa odległości do celu względem poprzedniego kroku. |
| `target_speed` | Nagroda za jazdę blisko docelowej prędkości. |
| `route_penalty` | Kara proporcjonalna do odległości od trasy. |
| `time_penalty` | Stała kara za każdy krok. |
| `goal_bonus` | Bonus przy dotarciu do celu. |
| `collision_penalty` | Kara za nową kolizję. |
| `offroute_penalty` | Kara za przekroczenie progu zejścia z trasy. |
| `lane_invasion_penalty` | Kara za lane invasion. |

Domyślne współczynniki:

```text
progress_coef = 1.0
target_speed_coef = 1.0
route_penalty_coef = 0.1
time_penalty = 0.01
goal_bonus = 50.0
collision_penalty = 50.0
offroute_penalty = 25.0
lane_invasion_penalty = 5.0
target_speed_kmh = 20.0
offroute_threshold = 10.0
reward_clip = 50.0
```

Reward jest sumą komponentów i może być clipowany do `[-50, 50]`.

### 17.3 Logowanie komponentów rewardu

Wrapper dodaje `reward_components` do `info`. Worker zapisuje je w rolloutach, a
`summarize_reward_components()` sumuje i uśrednia komponenty przy update.

W logach update'ów pojawiają się pola typu:

```text
reward_legacy_sum
reward_legacy_mean
reward_progress_sum
reward_total_mean
...
```

zależnie od trybu rewardu.

## 18. Checkpointy, resume i rollback

### 18.1 Boundary checkpoint

`save_boundary_checkpoint(run_output_dir, global_t, worker_id)` zapisuje
checkpoint co `save_frequency` globalnych kroków. Domyślnie:

```text
save_frequency = 100000
```

Checkpoint top-level:

```text
<run_output_dir>/checkpoint.pth
<run_output_dir>/checkpoint_step.txt
```

Opcjonalnie, jeśli `save_worker_checkpoints=True`, zapisuje też:

```text
<run_output_dir>/checkpoints/worker_<id>/checkpoint.pth
<run_output_dir>/checkpoints/worker_<id>/checkpoint_step.txt
```

`last_checkpoint_boundary` zapobiega wielokrotnemu zapisowi tej samej granicy,
gdy kilka workerów trafi w okolice `global_t % save_frequency == 0`.

### 18.2 Best checkpoint

Po każdym epizodzie `GlobalNetwork.update_stats()` sprawdza, czy reward epizodu
jest nowym rekordem. Jeśli tak, worker zapisuje:

```text
best_checkpoint.pth
```

Zapis też jest blokowany, jeśli parametry globalne zawierają NaN.

### 18.3 Resume state

`resume_state.json` zawiera m.in.:

- `global_step`,
- `global_episode`,
- `total_updates`,
- `elapsed_training_s`,
- `last_session_elapsed_s`,
- timestampy,
- `training_args`.

To jest uzupełnienie checkpointu PyTorch. Checkpoint zawiera model i optimizer,
a `resume_state.json` pomaga odtworzyć księgowość czasu i argumentów.

### 18.4 Rollback po rapid crash

`new_hogwild_run_a3c.py` monitoruje padnięcia workerów. Dla każdego workera
trzyma:

- `restart_counts`,
- `last_crash_step`,
- `rapid_crash_count`.

Jeżeli worker pada wielokrotnie w krótkim oknie kroków:

```text
rapid_crash_count >= rapid_crash_threshold
```

domyślnie threshold to 3, a okno to 100 kroków, supervisor wywołuje
`rollback_global_network(...)`.

Rollback próbuje załadować najnowszy checkpoint bez NaN. Kolejność:

1. checkpoint konkretnego workera, jeśli istnieje,
2. top-level checkpoint,
3. checkpointy innych workerów od najnowszego.

Jeżeli checkpoint ma NaN w parametrach, jest pomijany.

## 19. Logowanie, monitoring i timing

### 19.1 `TrainingLogger`

Logger zapisuje JSONL, czyli jeden rekord JSON na linię.

Układ:

```text
<run_output_dir>/logs/
  metadata.json
  events.jsonl
  system.jsonl
  worker_0/
    episodes.jsonl
    updates.jsonl
    steps.jsonl
    timing.jsonl
    system.jsonl
  worker_1/
    ...
```

`steps.jsonl` istnieje tylko przy `--log-steps`.

### 19.2 Log epizodu

`episodes.jsonl` zawiera m.in.:

- `global_episode`,
- `global_t`,
- `total_reward`,
- `steps`,
- `mean_reward`,
- `reached_goal`,
- `duration_s`,
- `max_speed_kmh`,
- `min_route_dist`,
- `goal_dist`,
- `collisions`,
- `action_counts`,
- `local_mean_reward`,
- `global_mean_reward`,
- `reward_components`.

### 19.3 Log update'u

`updates.jsonl` zawiera m.in.:

- `update`,
- `global_t`,
- `trajectory_length`,
- `is_terminal`,
- `pi_loss`,
- `v_loss`,
- `total_loss`,
- `gradient_norm`,
- `lr`,
- `advantages_mean`,
- `advantages_std`,
- `val_mean`,
- `val_std`,
- `ent_mean`,
- `rew_mean`,
- `rew_sum`,
- komponenty rewardu.

Jeżeli `--log-update-arrays`, logger zapisuje też pełne tablice advantages,
values, rewards i entropies. To może mocno zwiększyć logi.

### 19.4 Eventy

`events.jsonl` zawiera zdarzenia:

- `training_start`,
- `training_end`,
- `worker_start`,
- `worker_restart`,
- `worker_give_up`,
- `rollback`,
- `checkpoint_save`,
- `nan_gradient`,
- `crash_recovery`,
- `camera_timeout`,
- `worker_crash`.

### 19.5 Monitoring systemu

`RunMonitor` zapisuje:

- CPU mean/max,
- liczba zajętych CPU,
- RAM,
- swap,
- liczba procesów,
- procesy CARLA,
- GPU utilization i VRAM, jeśli `pynvml` jest dostępne.

`WorkerMonitor` zapisuje per worker:

- CPU,
- RSS,
- VMS,
- liczba wątków,
- context switches.

Monitoring używa `psutil`; GPU używa opcjonalnie `pynvml`.

### 19.6 Timing

`TimingAccumulator` mierzy czas faz:

- `sync`,
- `env_reset`,
- `forward`,
- `env_step`,
- `loss_compute`,
- `backward`,
- `optim_update`,
- `checkpoint_save`.

Worker okresowo loguje średnie czasy operacji. To pozwala rozpoznać, czy bottleneck
jest w CARLA, forwardzie, backwardzie, kopiowaniu gradientów czy zapisie.

## 20. Dokładny przepływ danych od obrazu CARLA do update'u wag

Poniżej pełna ścieżka jednego kroku i jednego update'u.

### 20.1 Reset

```text
A3CWorker.run()
  -> env.reset()
    -> CarlaA3CWrapper.reset()
      -> CarlaEnv.reset(...)
        -> spawn vehicle
        -> add camera
        -> add collision/lane sensors
        -> tick world
        -> get latest camera frame
        -> process_semantic_img/process_rgb_img
      <- front_camera tensor [1,3,H,W], speed km/h
      -> _state_to_chw_float
      -> speed / 100
      -> current maneuver
  <- state numpy [3,H,W], speed float, maneuver int
```

### 20.2 Forward i akcja

```text
state numpy [3,H,W]
  -> torch.from_numpy(state).float()
  -> unsqueeze(0)
  -> [1,3,H,W] on cuda/cpu

speed float
  -> tensor [[speed]]
  -> [1,1]

maneuver int
  -> tensor [maneuver]
  -> [1]

SharedActorCritic.forward
  -> CNN image features [1,4096]
  -> speed features [1,32]
  -> maneuver one-hot [1,3]
  -> maneuver features [1,32]
  -> concat [1,4160]
  -> trunk [1,256]
  -> policy logits [1,10]
  -> value [1,1]

Categorical(logits)
  -> sample action albo argmax
  -> log_prob(action)
  -> entropy
```

### 20.3 Step w CARLA

```text
action index
  -> ACTION_CONTROL
  -> carla.VehicleControl(throttle, brake, steer)
  -> apply_control
  -> world.tick x action_repeat
  -> CarlaEnv.step
    -> calculate distance to goal
    -> calculate route distance
    -> calculate speed
    -> calculate reward
    -> get camera image
    -> process image
  -> wrapper normalizes next_state and reward
```

### 20.4 Rollout

Worker dopisuje:

```text
trajectory: value, log_prob, entropy, action
rewards: reward
reward_components: info["reward_components"]
```

Po `rollout_length` krokach albo `done=True` przechodzi do update'u.

### 20.5 Update

```text
final state
  -> V(final_state) jeśli nie done
  -> 0 jeśli done

rewards reversed
  -> returns [T,1]

advantages = returns - values.detach()
normalize advantages

loss = policy_loss + value_loss - entropy_coef * entropy
backward on local model
clip local gradients
check NaN
copy local grads to global CPU model
global optimizer.step()
global optimizer.zero_grad()
sync local model from global
```

To jest pełna pętla uczenia.

## 21. Konfiguracja i domyślne wartości

### 21.1 Konfiguracja runtime

Wartości z CLI trafiają do `config = SimpleNamespace(**vars(args))`. Workery nie
czytają argparse. Dostają gotowy obiekt config.

### 21.2 Ważne różnice między Python default i SLURM default

`new_hogwild_train_a3c_carla.py` ma `DEFAULT_NUM_WORKERS = 2`.

`new_hogwild_train.slurm` ma `NUM_WORKERS=1`.

Jeśli uruchamiasz przez SLURM bez `-w`, dostaniesz 1 workera, bo skrypt SLURM
jawnie przekaże `--num-workers 1`. Jeśli uruchamiasz Pythona bezpośrednio,
dostaniesz 2 workery.

### 21.3 Zależności

Root `requirements.txt` zawiera m.in.:

- `carla==0.9.15`,
- `torch==2.8.0`,
- `numpy==2.0.2`,
- `opencv-python==4.12.0.88`,
- `psutil==7.1.3`,
- `wandb==0.22.3`,
- CUDA paczki `nvidia-...cu12`.

`AV/Pipfile` jest starszy i mówi o Python 3.8 oraz `torch = "1.8.1"`. To jest
niespójne z root `requirements.txt` i z logiem klastra używającym modułów
Python 3.11/CUDA 12.4. Przy uruchamianiu ważniejsze jest faktycznie aktywne
środowisko niż sam plik z repozytorium.

## 22. Różnice względem starszych plików A2C/A3C

W `A_to_B_GPU_34/` są starsze pliki:

- `a2c_rgb.py`,
- `a2c_rgb_baseline.py`,
- `a2c_rgb_try2.py`,
- `a3c.py`,
- `a3c_improved.py`,
- `a3c_improved_0.py`,
- `a3c_improved_1.py`,
- `a3c_multigpu_profile.py`.

Najważniejsze różnice nowego `new_hogwild_*`:

| Obszar | Starsze pliki | `new_hogwild_*` |
|---|---|---|
| Organizacja | Dużo logiki w jednym pliku. | Moduły: entrypoint, worker, wrapper, logger, monitor, supervisor. |
| Model | Często osobny actor i critic z `nets/a2c.py`. | Jeden `SharedActorCritic` ze wspólnym trunkem. |
| BatchNorm/Dropout | Obecne w `nets/a2c.py`. | Brak w nowym modelu. |
| Optymalizator | Głównie `SharedAdam`. | Domyślnie `SharedRMSprop`, opcjonalnie `SharedAdam`. |
| Licznik kroków | Często aktualizowany epizodowo. | `global_step` rośnie co decyzję agenta. |
| Logging | CSV/stdout/W&B. | Strukturalne JSONL + W&B opcjonalnie. |
| Recovery | Ograniczone restarty. | Supervisor, restart, backoff, rollback. |
| Checkpoint | Stałe ścieżki modeli. | Katalog runa, checkpoint step, resume state, best checkpoint. |
| CARLA wrapper | Worker mieszał logikę CARLA i RL. | Osobny `CarlaA3CWrapper`. |
| NaN guard | Słabszy. | Skip update przy NaN gradientach, brak zapisu NaN checkpointów. |

Stare pliki są nadal wartościowe jako historia eksperymentów, ale aktualny
pipeline treningowy to `new_hogwild_train.slurm` -> `new_hogwild_train_a3c_carla.py`.

## 23. Znane ograniczenia i miejsca ryzyka

### 23.1 Środowisko uruchomieniowe

W logu przykładowego joba widać:

- brak aktywnego venv pod ścieżką z launchera,
- brak `psutil` w środowisku, mimo że jest w `requirements.txt`.

To oznacza, że przed treningiem trzeba upewnić się, że aktywne środowisko ma
zainstalowane zależności używane przez:

- trening,
- monitor,
- supervisor CARLA,
- W&B, jeśli włączone.

### 23.2 Zewnętrzny multiserver script

SLURM używa `carla_athena_multiserver_v3.py` poza repozytorium `AV/`. Jeżeli ten
plik zmieni się albo zniknie, launcher nie działa mimo poprawnego kodu
`new_hogwild_*`.

### 23.3 Race conditions

Domyślny Hogwild oznacza realne race conditions w optimizer step. To jest
zamierzone, ale jeżeli debugging wskazuje niestabilność, można uruchomić z:

```bash
-- --hogwild-lock-updates
```

czyli przekazać argument do Pythona przez `--` w skrypcie SLURM.

### 23.4 Checkpointy legacy

`GlobalNetwork.load()` nie obsługuje starszych checkpointów z oddzielnym
actorem i criticiem. Musi być checkpoint zawierający klucz `model` dla
`SharedActorCritic`.

### 23.5 `settings.py` nadal wpływa na `CarlaEnv`

Chociaż `new_hogwild_a3c.py` nie czyta `settings.py`, `carla_env.py` importuje
`settings` i używa m.in.:

- `STEP_COUNTER`,
- `SLEEP_BETWEEN_ACTIONS`,
- `REWARD_FROM_*`,
- `SERV_RESX`, `SERV_RESY`,
- `SPAWNING_TYPE`,
- `DRAW`,
- `ACTIONS`.

Zmiana `settings.py` może więc zmienić środowisko nawet bez zmiany argumentów
`new_hogwild_train_a3c_carla.py`.

### 23.6 Kształty wejścia

Sieć zakłada obraz 3-kanałowy. Wrapper wymusza `[3,resY,resX]`. Jeżeli kamera
lub preprocessing zacznie zwracać inny format, wrapper rzuci `ValueError`
zamiast po cichu trenować na złych danych. To jest dobre zachowanie diagnostyczne.

### 23.7 Epizody nie są zawsze porównywalne między wersjami

Nowy wrapper ma `episode_max_decisions=100`, a `CarlaEnv` ma `STEP_COUNTER=200`.
Starsze wersje mogły raportować inną liczbę epizodów/h, bo definicja końca
epizodu i action repeat były inne.

### 23.8 `spawn_single_pedestrian`

W `carla_env.py` jest funkcja `spawn_single_pedestrian`, która wygląda na
eksperymentalną i spawnuje kontroler pieszego więcej niż raz. Nie jest używana w
aktualnym `reset()`; jest zakomentowana. Nie wpływa na `new_hogwild_*`, dopóki
ktoś jej nie odkomentuje.

## 24. Jak czytać wyniki treningu

Najważniejsze pliki:

```text
logs/metadata.json
logs/events.jsonl
logs/worker_*/episodes.jsonl
logs/worker_*/updates.jsonl
logs/worker_*/timing.jsonl
logs/system.jsonl
logs/worker_*/system.jsonl
```

### 24.1 Czy agent się uczy

Patrz na:

- `global_mean_reward`,
- `local_mean_reward`,
- `total_reward`,
- `reached_goal`,
- `goal_dist`,
- `min_route_dist`,
- `action_counts`.

Jeżeli reward rośnie, ale `reached_goal=False` i `goal_dist` nie maleje, agent
może optymalizować reward poboczny, a nie dojazd.

### 24.2 Czy policy degeneruje

Patrz na:

- `ent_mean` w `updates.jsonl`,
- `action_counts`,
- `most_chosen_action_pct` w W&B log recordach.

Jeżeli jedna akcja dominuje bardzo wcześnie, entropy może być za niskie albo
reward zachęca do zbyt prostego zachowania.

### 24.3 Czy gradienty są stabilne

Patrz na:

- `gradient_norm`,
- eventy `nan_gradient`,
- `total_loss`,
- `pi_loss`,
- `v_loss`.

Rosnący `gradient_norm` i eventy NaN sugerują za wysoki LR, zły reward scale,
problemy z obserwacją albo niestabilny rollout.

### 24.4 Czy bottleneck jest w CARLA

Patrz na `timing.jsonl`:

- duże `env_step` lub `env_reset` -> bottleneck CARLA,
- duże `forward`/`backward` -> bottleneck model/GPU,
- duże `optim_update` -> koszt kopiowania gradientów CPU albo lock/race,
- duże `checkpoint_save` -> IO.

### 24.5 Czy serwery CARLA żyją

Patrz na:

- `carla_servers.log`,
- `logs/system.jsonl` sekcja `carla`,
- eventy `worker_restart`, `crash_recovery`, `camera_timeout`.

`camera_timeout` oznacza, że worker nie dostał obrazu w czasie. Kod pomija
epizod, czyści rollout i próbuje iść dalej.

## 25. Słownik pojęć

| Pojęcie | Wyjaśnienie |
|---|---|
| A3C | Asynchronous Advantage Actor-Critic, algorytm RL z wieloma workerami. |
| Actor | Część modelu wybierająca akcję. |
| Critic | Część modelu oceniająca stan przez `V(s)`. |
| Advantage | `R - V(s)`, informacja czy akcja była lepsza od oczekiwań. |
| Rollout | Krótka sekwencja kroków zebrana przez workera przed update'em. |
| Bootstrap | Użycie `V(s_last)` jako końcówki returnu, gdy rollout nie kończy epizodu. |
| Entropy | Miara losowości policy. Wyższa entropy = większa eksploracja. |
| Logits | Surowe wyjścia sieci dla akcji, przed softmaxem. |
| Categorical | Rozkład dyskretny PyTorch używany do losowania akcji. |
| Hogwild | Update wielu workerów bez blokowania na wspólnych parametrach. |
| Shared memory | Pamięć widoczna dla wielu procesów. |
| `.grad` | Atrybut parametru z gradientem po `backward()`. |
| CUDA | Platforma obliczeń GPU NVIDIA. |
| NCHW | Format tensora obrazu `[batch, channels, height, width]`. |
| CHW | Format pojedynczego obrazu `[channels, height, width]`. |
| CARLA tick | Jeden synchroniczny krok symulatora. |
| Action repeat | Powtórzenie tej samej akcji przez kilka ticków. |
| `done` | Flaga końca epizodu. |
| Checkpoint | Zapis modelu, optymalizatora i liczników. |
| Resume | Wznowienie treningu z checkpointu. |
| Rollback | Cofnięcie globalnego modelu do ostatniego poprawnego checkpointu. |

## Krótki pseudokod całej implementacji

```text
main:
    args = parse_cli()
    config = args + defaults
    set seeds
    assign worker GPUs
    create output dir
    set multiprocessing start method = spawn
    create shutdown_event
    create GlobalNetwork on CPU shared memory
    if resume:
        load checkpoint
    write metadata
    start RunMonitor
    start optional W&B process
    run_with_restart(global_network, config)
    on shutdown:
        stop W&B
        stop monitor
        write resume_state
        write training_end event

run_with_restart:
    for each worker:
        start A3CWorker(worker_id, port, device)
    while not shutdown:
        sleep worker_check_interval
        for each worker:
            if dead:
                join
                count restart
                maybe rollback
                wait backoff
                restart worker
    stop all workers

A3CWorker.run:
    set thread limits and seeds
    create local model on device
    sync local model from global
    create logger and monitor
    create CarlaA3CWrapper
    while not shutdown:
        clear rollout
        global_episode += 1
        sync local model
        state, speed, maneuver = env.reset()
        while not done:
            global_step += 1
            action = sample policy(state, speed, maneuver)
            next_state, reward, done, info = env.step(action)
            store value/log_prob/entropy/reward
            if rollout_length reached or done:
                compute n-step returns
                compute advantage
                compute policy/value/entropy loss
                backward local model
                clip gradients
                if NaN: skip update and resync
                else:
                    copy gradients to global CPU model
                    global optimizer.step()
                    global optimizer.zero_grad()
                    total_updates += 1
                    log update
                clear rollout
                sync local model if configured
            maybe save checkpoint
        log episode
```

## Najkrótsze podsumowanie działania

`new_hogwild_*` to modułowy trening A3C dla CARLA. Każdy worker ma własny
symulator CARLA i lokalną kopię sieci na GPU/CPU. Globalna sieć i optymalizator
są na CPU w pamięci współdzielonej. Worker zbiera krótki rollout, liczy lokalny
loss actor-critic, kopiuje gradienty do globalnej sieci i wykonuje asynchroniczny
krok optymalizatora. Wrapper dba o normalizację obrazu, prędkość, manewr trasy,
powtarzanie akcji, reward i limity epizodu. Supervisor restartuje padnięte
workery i potrafi cofnąć model do checkpointu. Całość jest uruchamiana przez
SLURM, który startuje serwery CARLA, czeka na porty, uruchamia trening i sprząta
procesy przy końcu joba.
