# Porównanie `new_hogwild_*` z `a3c_improved_1.py`

Stan dokumentu: 2026-05-08.

Ten plik opisuje aktualną implementację A3C w plikach `new_hogwild_*` w katalogu
`AV/A_to_B_GPU_34` oraz porównuje ją z bazową implementacją
`a3c_improved_1.py`. Opis obejmuje także zmiany w `carla_env.py` i skrypcie
`new_hogwild_train.slurm`.

## Krótkie podsumowanie

`a3c_improved_1.py` było jednoplikową implementacją A3C: globalny actor i critic
na CPU w pamięci współdzielonej, lokalne modele workerów na GPU, gradienty
kopiowane GPU -> CPU i optymalizacja w stylu Hogwild. Ta część koncepcyjnie
została zachowana.

`new_hogwild_*` rozdziela ten kod na moduły, dodaje supervisor restartów,
checkpointy, resume, JSONL logging, monitoring systemu/GPU, mniej kosztowny reset
CARLA, reward shaping, aktywne ograniczanie epizodów i czytelniejszy launcher
SLURM. Najważniejsze zmiany nie są kosmetyczne: nowa wersja ma wyraźnie lepszą
obserwowalność, krótszy hot path epizodu i mniejsze ryzyko utraty pracy po
awarii.

Bardzo duża liczba epizodów w nowej implementacji wynika z dwóch grup zmian.
Pierwsza grupa to faktyczne przyspieszenie pętli: brak pełnego `reload_world()`
co epizod, mniej logowania stdout, czystsza obsługa kolejki obrazów i brak
ciągłego `cProfile` w workerze. Druga grupa to zmiana definicji epizodu:
wrapper domyślnie kończy epizod po `DEFAULT_EPISODE_MAX_DECISIONS = 100`, a przy
`DEFAULT_ACTION_REPEAT = 2` jedna decyzja agenta wykonuje dwa niskopoziomowe
kroki CARLA. W starej ścieżce limit środowiska wynosił `STEP_COUNTER = 200`
decyzji workera. Dlatego epizody/h są lepszą metryką przepustowości, ale nie są
w pełni porównywalne 1:1 ze starym `a3c_improved_1.py` bez uwzględnienia limitu
długości epizodu.

## Mapa plików

| Plik | Rola w aktualnej implementacji |
|---|---|
| `new_hogwild_train_a3c_carla.py` | Główne wejście Pythona: konfiguracja, argparse, seed, GPU assignment, W&B, monitoring, resume, start supervisora. |
| `new_hogwild_a3c.py` | Rdzeń A3C: modele, `SharedAdam`, `SharedRMSprop`, `GlobalNetwork`, `A3CWorker`, loss, update, checkpointy. |
| `new_hogwild_run_a3c.py` | Supervisor workerów: start, restart, backoff, limit restartów, rollback do ostatniego poprawnego checkpointu. |
| `new_hogwild_carla_wrapper.py` | Warstwa między A3C i `CarlaEnv`: reset, retry, health-check, action repeat, reward shaping, zapis klatek. |
| `new_hogwild_training_logger.py` | Strukturalne logi JSONL: epizody, update'y, timing, eventy, metadane. |
| `new_hogwild_system_monitor.py` | Monitoring CPU, RAM, procesów CARLA, workerów i GPU przez `psutil`/`pynvml`. |
| `new_hogwild_timing_utils.py` | Lekki profiler faz pętli treningowej. |
| `new_hogwild_prepare_output_dir.py` | Tworzy katalog runa i zapisuje argumenty uruchomienia. |
| `new_hogwild_train.slurm` | Pełny launcher: start serwerów CARLA, czekanie na porty, start treningu, log tail, cleanup. |
| `carla_env.py` | Bazowe środowisko CARLA, poprawione tak, żeby wrapper mógł je używać bez kosztownego reloadu co epizod. |

## Co było w `a3c_improved_1.py`

`a3c_improved_1.py` łączyło w jednym pliku konfigurację, modele, workerów,
supervisor, logging, W&B i pętlę treningową. Wiele wartości było ustawionych jako
globalne stałe albo pobierane z `settings.py`, na przykład `NUM_WORKERS`,
`WORKER_GPUS`, `MODEL_LOAD_PATH`, `MODEL_SAVE_PATH`, `T_MAX`, `LR`,
`ENTROPY_COEF`, `SAVE_INTERVAL`, `CARLA_TIMEOUT_WAIT`.

Globalna sieć była na CPU i używała oddzielnych modeli:

- `DeepDiscreteActor`
- `DeepCritic`
- dwa osobne optymalizatory `SharedAdam`
- liczniki `global_episode`, `total_updates`, `global_steps`, `best_reward`

Worker uruchamiał lokalne kopie actor/critic na GPU. Co update wykonywał:

- synchronizację lokalnych wag z globalnymi,
- zbieranie trajektorii przez `T_MAX` kroków lub do końca epizodu,
- obliczenie n-step returns,
- osobny `actor_loss` i `critic_loss`,
- dwa wywołania `backward()`,
- kopiowanie gradientów do modeli globalnych na CPU,
- `optimizer.step()` na globalnych optymalizatorach.

To była dobra baza dla A3C, ale miała ograniczenia:

- reset CARLA robił pełny `reload_world()` przy każdym epizodzie,
- logging był głównie tekstowy/CSV, trudny do automatycznej analizy,
- checkpointy miały stałe ścieżki i były zależne od `MODEL_SAVE_PATH`,
- resume było ograniczone do modelu i kilku liczników,
- brakowało kontroli jakości checkpointów przy NaN,
- awarie workerów były restartowane, ale bez rollbacku,
- global step był aktualizowany po epizodzie, nie krok po kroku,
- lista argumentów i ustawień była rozproszona między kodem, `settings.py` i SLURM.

## Główne decyzje w `new_hogwild_*`

### 1. Zachowany rdzeń Hogwild, ale uporządkowany

Nowa implementacja nadal używa CPU shared-memory jako globalnego modelu. Workery
nadal działają na GPU, liczą forward/backward lokalnie i kopiują gradienty do
globalnego modelu na CPU.

Zmiany:

- `GlobalNetwork` znajduje się w `new_hogwild_a3c.py`.
- Liczniki są jednoznaczne: `global_step`, `global_episode`, `total_updates`.
- `global_step` rośnie w każdym kroku środowiska, a nie dopiero na końcu epizodu.
- Budżet treningu `--steps` jest dzięki temu rzeczywistym budżetem kroków.
- `last_checkpoint_boundary` zabezpiecza przed zapisem wielu checkpointów na tej
  samej granicy kroku, gdy wiele workerów trafi równocześnie w `save_frequency`.

Powód:

Skalowanie A3C zależy od tego, czy każdy worker może iść do przodu bez blokowania
innych. Jednocześnie potrzebny jest dokładny globalny licznik, bo na nim opiera
się learning-rate decay, checkpointing, resume i porównywanie runów.

### 2. Model domyślny: wspólny actor-critic

`a3c_improved_1.py` używało oddzielnego actora i critica z `nets.a2c`.

`new_hogwild_a3c.py` domyślnie używa `SharedActorCritic`, czyli jednego modelu ze
wspólnym trunkem wizualnym i dwiema głowami:

- `policy`
- `value`

Stary układ jest nadal dostępny przez `--model-arch legacy`.

Powód:

Wspólny trunk zmniejsza ilość pracy na forward/backward i ogranicza rozjazd
reprezentacji między policy i value. Model nie używa BatchNorm/Dropout, więc jest
bezpieczniejszy dla batch size 1 typowego dla workerów A3C.

### 3. Optymalizator: domyślnie `SharedRMSprop`

`a3c_improved_1.py` używało `SharedAdam`.

`new_hogwild_a3c.py` ma:

- `SharedRMSprop` jako domyślny optymalizator,
- `SharedAdam` jako opcję przez `--optimizer shared-adam`.

Powód:

RMSprop jest klasycznym wyborem dla A3C i ma prostszy stan współdzielony. Adam
pozostał dostępny, bo może być przydatny do porównań albo resume starszych
eksperymentów.

### 4. Stabilniejszy loss i update

W `a3c_improved_1.py` entropy było odejmowane od actor loss, gradient clipping był
zakomentowany, a actor i critic robiły osobne backward.

W `new_hogwild_a3c.py`:

- loss jest liczony jako `policy_loss + value_loss - entropy_coef * entropy`,
- value loss używa `smooth_l1_loss`,
- `value_loss_coef` jest jawne,
- advantages mogą być normalizowane,
- entropy coefficient może być annealowany od `DEFAULT_BETA_START` do
  `DEFAULT_BETA_END`,
- gradient clipping jest aktywny przez `DEFAULT_MAX_GRAD_NORM`,
- NaN w gradientach powoduje pominięcie update'u i resync workera z globalnego
  modelu,
- NaN w globalnych parametrach blokuje zapis checkpointu.

Powód:

Te zmiany zmniejszają ryzyko eksplozji gradientów i psucia checkpointów. Dają też
więcej danych diagnostycznych: w logach update'u są straty, gradient norm,
learning rate, entropy, statystyki advantages i reward components.

### 5. Prawdziwy step-based trening

Stara wersja zwiększała `global_steps` w `update_stats()` na końcu epizodu.

Nowa wersja zwiększa `global_step` w `A3CWorker.run()` przed każdym krokiem
środowiska. Dzięki temu:

- `--steps` zatrzymuje trening po realnej liczbie kroków,
- learning-rate decay reaguje na faktyczny postęp,
- checkpointy są zapisywane po granicach kroków,
- runy 1-worker i multi-worker można porównywać po aktywnych krokach, a nie tylko
  po epizodach.

Powód:

Epizody mają zmienną długość, szczególnie po dodaniu limitu decyzji i reward
shapingu. Step-based licznik jest bardziej stabilnym punktem odniesienia.

## Przepływ workera: wcześniej i teraz

### Worker w `a3c_improved_1.py`

Stary worker wykonywał prawie całą logikę samodzielnie:

1. Ustawiał limity wątków CPU i uruchamiał `cProfile`.
2. Tworzył lokalnego actora i critica na przypisanym GPU.
3. Łączył się bezpośrednio z `CarlaEnv`.
4. Na początku epizodu zwiększał `global_episode`, synchronizował lokalne modele
   z globalnymi i wywoływał `env.reset(...)`.
5. Normalizował obraz przez `/ 255.0` i prędkość przez `/ 100.0` bez osobnej
   walidacji formatu obserwacji.
6. W każdym kroku sam sprawdzał manewr na skrzyżowaniu przez `env.planner`.
7. Wybierał akcję z lokalnego actora.
8. Wykonywał `env.step_apply_action(action)`, czyścił `image_queue`, robił dwa
   `world.tick()` i dopiero potem wywoływał `env.step(...)`.
9. Gromadził rewardy i trajektorię.
10. Po `T_MAX` krokach albo końcu epizodu liczył oddzielny actor loss i critic
    loss, robił dwa `backward()` i kopiował gradienty do globalnych modeli.
11. Na końcu epizodu aktualizował `global_steps` o długość epizodu, logował CSV i
    co `SAVE_INTERVAL` lokalnych epizodów zapisywał model do stałej ścieżki.
12. Profilowanie `cProfile` było włączone wewnątrz workera i okresowo zrzucało
    pliki `profile_worker_<id>.out`.

Ten worker działał, ale mieszał algorytm A3C, szczegóły CARLA, zapis danych,
profilowanie i recovery w jednym procesie.

### Worker w `new_hogwild_*`

Nowy worker ma węższą odpowiedzialność:

1. Ustawia limity wątków CPU, seed per worker i przypisuje urządzenie CUDA.
2. Tworzy lokalny model `SharedActorCritic` albo legacy actor/critic.
3. Otwiera strukturalne logi JSONL i `WorkerMonitor`.
4. Tworzy `CarlaA3CWrapper`, który przejmuje reset, ticki, kolejkę obrazów,
   manewry, reward shaping i zapis klatek.
5. Na początku epizodu zwiększa `global_episode`, synchronizuje wagi i wywołuje
   `env.reset()` wrappera.
6. W każdym kroku zwiększa `global_step`, więc budżet `--steps`, LR schedule i
   checkpointy są krokowe, a nie epizodowe.
7. Forward lokalnego modelu zwraca akcję, value i entropy.
8. Wrapper wykonuje akcję `action_repeat` razy, robi ticki CARLA, pobiera świeżą
   obserwację, przelicza reward i zwraca `info` z metrykami środowiska.
9. Po `t_max` krokach albo końcu epizodu worker liczy jeden łączny loss,
   wykonuje jeden `backward()`, klipuje gradienty, sprawdza NaN i dopiero wtedy
   przenosi gradienty do globalnego modelu.
10. Po update worker synchronizuje się z globalnym modelem co
    `sync_every_n_updates`.
11. Na końcu epizodu loguje JSONL, aktualizuje rolling mean reward, zapisuje
    `best_checkpoint.pth` przy nowym najlepszym wyniku i wykonuje GC zgodnie z
    `gc_interval`.

Najważniejsza różnica praktyczna: stary worker sam obsługiwał CARLA i przez to
każda zmiana środowiska dotykała pętli A3C. Nowy worker traktuje CARLA jako
wrapper o prostym API: `reset()` i `step(action)`.

Druga ważna różnica praktyczna dotyczy długości epizodu. W starej pętli jedna
decyzja workera oznaczała jedno `step_apply_action()` i limit środowiska
`STEP_COUNTER = 200`. W nowej pętli jedna decyzja A3C oznacza domyślnie dwa
`step_apply_action()` przez `action_repeat = 2`, a wrapper dodatkowo kończy
epizod po 100 decyzjach. To powoduje, że agent szybciej zamyka epizody i częściej
dostaje terminalne update'y. Jest to celowe, ale przy porównywaniu runów trzeba
patrzeć również na kroki, czas `env_step` i długość epizodu, nie tylko na liczbę
epizodów.

## Gradienty i Hogwild

W obu implementacjach ogólna idea jest taka sama: globalny model jest na CPU w
pamięci współdzielonej, workery mają lokalne kopie modelu na GPU, liczą gradienty
lokalnie i przenoszą je do globalnego modelu.

W `a3c_improved_1.py` współdzielone były:

- parametry globalnego actora i critica,
- stan optymalizatorów `SharedAdam`,
- bufory `.grad` globalnych parametrów.

`transfer_grads_to_shared()` kopiowało lokalne gradienty bezpośrednio do tych
współdzielonych buforów `.grad`, a potem worker wywoływał `optimizer.step()`. To
było proste i szybkie, ale przy równoległych workerach mogło dojść do wyścigu:
worker A zaczynał wpisywać gradient, worker B nadpisywał ten sam bufor, a worker A
mógł wykonać krok optymalizatora na gradiencie częściowo należącym do innego
workera.

W `new_hogwild_a3c.py` współdzielone są parametry i stan optymalizatora, ale
gradient danego workera jest klonowany i przypisywany procesowo lokalnie podczas
`transfer_grads()`. Worker nie udostępnia jednego stałego bufora `.grad`, do
którego równocześnie piszą inne workery. Same `optimizer.step()` nadal są domyślnie
Hogwild, czyli bez globalnego locka wokół aktualizacji wag.

Dla diagnostyki istnieje `--hogwild-lock-updates`. Ta opcja serializuje update'y
workerów przez lock, ale normalny tryb pozostaje asynchroniczny.

Powód:

Nowa wersja ogranicza najbardziej ryzykowny wyścig na buforach gradientów, ale
zachowuje główną cechę Hogwild: wielu workerów może aktualizować wspólny model bez
stałego blokowania całej pętli treningowej.

## Zmiany w konfiguracji i argumentach

### Aktualny stan

`new_hogwild_train_a3c_carla.py` trzyma domyślne wartości jako jawne stałe
modułowe. Są pogrupowane według odpowiedzialności:

- `# Run shape`
- `# Algorithm`
- `# Optimizer internals`
- `# Recovery`
- `# Logging and monitoring`
- `# Checkpointing`
- `# CARLA step/reset behavior`
- `# Reward shaping`
- `# Fixed implementation choices`

### Co zostało w CLI

W CLI zostały argumenty, które realnie mają sens jako parametry runa:

- liczba workerów i mapowanie na GPU,
- porty CARLA,
- scenariusz,
- seed,
- model/optimizer jako opcje eksperymentalne,
- `t-max`, `gamma`, `lr`, entropy schedule,
- checkpoint/resume,
- monitoring/logging,
- `action-repeat`, `episode-max-decisions`, `world-reload-interval`,
- `reward-mode`.

### Co jest stałą konfiguracją implementacji

Parametry, które opisują bieżący wariant algorytmu i zwykle nie są zmieniane w
każdym uruchomieniu, są stałymi w Pythonie:

- `mp_density`,
- parametry wewnętrzne RMSprop/Adam,
- współczynniki reward shapingu,
- `action_type`,
- domyślne wartości recovery i monitoringu.

Powód:

Typowa komenda SLURM ma opisywać głównie kształt runa: liczbę workerów, GPU,
scenariusz, porty, resume i monitoring. Stałe algorytmu są w jednym miejscu na
górze pliku Pythona, więc łatwo sprawdzić, jaki wariant A3C jest aktualnie
uruchamiany.

## Zmiany w SLURM

`new_hogwild_train.slurm` jest teraz launcherem całej ścieżki:

1. parsuje mały zestaw argumentów,
2. tworzy katalog runa,
3. uruchamia `carla_athena_multiserver_v3.py`,
4. czeka aż porty CARLA będą w stanie `LISTEN`,
5. uruchamia `new_hogwild_train_a3c_carla.py`,
6. zapisuje `a3c_training.log`, `carla_servers.log`, `gpu_dmon.log`,
7. przy wyjściu zamyka trening, tail loga, serwery CARLA i `nvidia-smi dmon`.

Najczęściej wystarczają argumenty:

```bash
sbatch --ntasks-per-node=7 --cpus-per-task=7 --mem=200G --time=5:00:00 --gpus=6 \
  --job-name=a3c-carla-6w_hogwild_gpu \
  /net/tscratch/people/plgbartoszkawa/AV/A_to_B_GPU_34/new_hogwild_train.slurm \
  -w 6 -s 14 --workers-per-gpu 1 --servers-per-gpu 1
```

Rzadkie argumenty Pythona można przekazać po `--`, na przykład:

```bash
sbatch --gpus=6 new_hogwild_train.slurm -w 6 --workers-per-gpu 1 --servers-per-gpu 1 -- --lr 5e-5
```

Powód:

Stary styl wymagał wielu argumentów w komendzie i mieszał parametry infrastruktury
z parametrami algorytmu. Teraz SLURM odpowiada głównie za run shape i CARLA, a
Python trzyma stałe algorytmu.

## Zmiany w katalogu runa

`new_hogwild_prepare_output_dir.py` zapisuje:

- `args.txt` dla nowego runa,
- `args_resume.txt` przy resume.

Pozostałe artefakty runa zapisują konkretne komponenty pipeline'u:

- `a3c_training.log` zapisuje stdout/stderr treningu,
- `carla_servers.log` zapisuje log supervisora serwerów CARLA,
- `gpu_dmon.log` zapisuje monitoring `nvidia-smi dmon`,
- `logs/` zawiera JSONL z epizodami, update'ami, timingiem, eventami i systemem,
- checkpointy i `resume_state.json` zapisują stan potrzebny do wznowienia.

Powód:

Katalog runa ma zawierać rzeczy potrzebne do analizy i wznowienia treningu, a nie
być kopią całego stanu repozytorium. Dzięki temu łatwiej porównywać runy i
automatycznie parsować wyniki.

## CARLA wrapper i reset środowiska

Największa praktyczna różnica względem `a3c_improved_1.py` jest w obsłudze
CARLA, a nie w samym równaniu A3C. Stara wersja używała `CarlaEnv` bezpośrednio
w workerze. Worker sam resetował `state_observer`, normalizował obraz i speed,
czyścił `image_queue`, wykonywał akcję, robił ticki świata, obsługiwał manewry
na skrzyżowaniach i decydował o zapisie obrazów.

Nowa wersja przenosi tę logikę do `new_hogwild_carla_wrapper.py`. `carla_env.py`
pozostaje niskopoziomowym środowiskiem CARLA: świat, sensory, aktorzy, trasa,
kamera, kolejki i legacy reward. Wrapper decyduje, jak A3C ma z tego środowiska
korzystać:

- łączy się z `CarlaEnv` przez retry i potrafi zrobić `reconnect`,
- resetuje epizod bez pełnego reloadu świata, chyba że `world_reload_interval`
  wymusi reload,
- waliduje obserwację i zamienia format na `[3, H, W]`,
- skaluje obraz do `[0, 1]` i speed do `speed / 100`,
- wykonuje `action_repeat`,
- kończy epizod po `episode_max_decisions`,
- liczy shaped reward i zapisuje jego komponenty,
- zapisuje klatki tylko dla wskazanych epizodów,
- zbiera statystyki epizodu potrzebne do JSONL i W&B.

Powód podziału jest praktyczny. Gdyby całą logikę A3C dopisać do `carla_env.py`,
środowisko musiałoby znać `global_episode`, `run_id`, `action_repeat`,
`save_episodes`, shaped reward, format wejścia PyTorch, retry policy i statystyki
workerów. To związałoby ogólne środowisko CARLA z jednym konkretnym pipeline'em.
Obecny układ zostawia `CarlaEnv` jako warstwę CARLA, a wrapper jako warstwę
treningu.

## Dlaczego nowa wersja robi tak dużo epizodów

Wysoka liczba epizodów/h w nowym pipeline nie wynika tylko z szybszego
forward/backward. Najważniejsze przyczyny są po stronie resetu, długości epizodu
i hot path środowiska.

1. Brak pełnego `reload_world()` co epizod.
   W starej ścieżce `env.reset(...)` wołał domyślne `reload_world=True`, więc
   każdy epizod niszczył świat, ładował go ponownie, odświeżał mapę, blueprinty i
   ustawienia synchroniczne. W nowej ścieżce wrapper liczy `full_reload` z
   `world_reload_interval`; przy domyślnym `0` wywołuje
   `reset_episode_state()`, które czyści aktorów, sensory, kolejkę obrazów i
   liczniki, ale nie robi `client.reload_world()`. To usuwa największy stały
   koszt początku epizodu.

2. Epizod jest krótszy jako jednostka treningowa.
   Stare środowisko miało `STEP_COUNTER = 200`, a stary worker wykonywał jedno
   `step_apply_action()` na jedną decyzję agenta. Nowy wrapper ma domyślnie
   `episode_max_decisions = 100` i `action_repeat = 2`. Jedna decyzja A3C
   wykonuje więc dwa `step_apply_action() + world.tick()`, a epizod kończy się po
   100 decyzjach lub wcześniej. Dlatego liczba epizodów może mocno wzrosnąć nawet
   dla jednego workera. To jest poprawne dla tej implementacji, ale epizod/h nie
   oznacza dokładnie tego samego co w `a3c_improved_1.py`.

   Dokładna różnica wygląda tak:

   | Element epizodu | `a3c_improved_1.py` | `new_hogwild_*` |
   |---|---|---|
   | Jednostka, dla której policy wybiera akcję | 1 decyzja workera | 1 decyzja A3C / 1 `global_step` |
   | Wywołania `step_apply_action()` na jedną decyzję | 1 | 2 przy `action_repeat=2` |
   | `world.tick()` na jedną decyzję | 2 | 2, po jednym ticku na każde powtórzenie akcji |
   | Obserwacja i reward | po każdej decyzji workera | po całym `action_repeat`, czyli po dwóch powtórzeniach tej samej akcji |
   | Główny licznik końca epizodu | `CarlaEnv.step_counter >= STEP_COUNTER`, domyślnie 200 | `CarlaA3CWrapper.step_count >= episode_max_decisions`, domyślnie 100, oraz nadal wewnętrzny limit `CarlaEnv.step_counter >= 200` |
   | Maksymalna liczba decyzji policy w epizodzie | około 200 | około 100 |

   To znaczy, że nowy epizod zwykle zawiera mniej decyzji policy, mniej zapisanych
   rewardów i mniej kroków A3C niż stary epizod. Jednocześnie pojedyncza decyzja
   w nowej wersji reprezentuje powtórzoną akcję w CARLA. Dlatego dwa runy z tą
   samą liczbą epizodów nie muszą oznaczać tej samej liczby decyzji agenta,
   update'ów ani tej samej długości symulowanego przejazdu. Do uczciwego
   porównania trzeba zestawiać także `global_step`, `total_updates`, średnią
   długość epizodu, `action_repeat` i czasy `env_reset`/`env_step`.

3. Mniej pracy w samym workerze.
   Stary worker miał w pętli szczegóły CARLA, ręczne czyszczenie kolejki, ręczną
   obsługę manewrów i stale włączony `cProfile`. Nowy worker wywołuje prosty
   `env.step(action)`, a szczegóły robi wrapper. Profilowanie jest zastąpione
   lekkim `TimingAccumulator`, który zapisuje czasy faz bez kosztu pełnego
   profilera Pythonowego w każdym kroku.

4. Mniej blokowania przez logi i dysk.
   `CarlaEnv(verbose=False)` jest domyślne, więc wiele komunikatów środowiska nie
   trafia do stdout. Klatki są zapisywane tylko dla `--save-episodes` albo
   `--save-episode-interval`, a katalog zapisu jest cache'owany per epizod.
   Dzięki temu normalny hot path nie płaci za diagnostyczny zapis obrazów.

5. Stabilniejsza obsługa kamery.
   `_get_latest_camera_image()` pobiera najnowszą klatkę i wyrzuca starsze, a
   wrapper dodatkowo czyści kolejkę przed wykonaniem powtarzanej akcji. To
   zmniejsza ryzyko pracy na zaległych obrazach i ogranicza narastanie backlogu
   w kolejce kamery.

Wniosek: nowa implementacja rzeczywiście jest szybsza, ale część wzrostu liczby
epizodów pochodzi z tego, że epizod jest krótszy i tańszy. Przy porównaniach
należy patrzeć razem na `episodes/h`, `steps/s`, średnią długość epizodu,
`env_reset`, `env_step` i `action_repeat`.

## Zmiany w `carla_env.py`

`carla_env.py` zostało dostosowane tak, żeby wrapper mógł pracować szybciej i
czyściej.

Najważniejsze zmiany:

- `CarlaEnv.__init__` ma parametr `verbose=False`, więc standardowe treningi nie
  zalewają stdout logami środowiska.
- `self.verbose` jest ustawiane na początku konstruktora, przed pierwszym
  wywołaniem `plan_the_route()`. To naprawia regresję z runa
  `a3c_hogwild_6w_20260505_114615_2566209`, gdzie wszystkie workery kończyły się
  przed pierwszym epizodem błędem `AttributeError: 'CarlaEnv' object has no
  attribute 'verbose'`.
- `reset()` przyjmuje `reload_world=True` dla kompatybilności, ale wrapper
  domyślnie podaje `False`. Pełny reload można przywrócić okresowo przez
  `world_reload_interval`.
- `reset_episode_state()` niszczy aktorów i sensory, czyści kolejkę obrazów,
  historię kolizji/lane invasion, middle-point rewards, liczniki i stan prędkości
  bez przeładowywania świata.
- `_get_latest_camera_image()` pobiera najnowszą klatkę z kolejki i wyrzuca
  starsze, zamiast pozwalać workerowi pracować na zaległych obrazach.
- `reset()` i `step()` zwracają prędkość jako zwykły `float`, nie tensor zależny
  od CUDA. Wrapper dopiero potem skaluje speed i tworzy tensor na urządzeniu
  workera.
- `step()` zapisuje `last_invasion_counter`, żeby wrapper mógł policzyć penalty
  za lane invasion bez czytania wewnętrznej listy środowiska.
- Obraz z CARLA jest zapisywany w `state_observer.image`, żeby wrapper mógł
  wykonać `save_to_disk()` tylko dla wskazanych epizodów.
- Semantyczne i RGB obserwacje są przygotowywane tak, żeby wrapper mógł wymusić
  poprawny format `[3, H, W]`.

Powód:

Pełny reload świata był jednym z największych kosztów czasu i nie jest potrzebny
po każdym krótkim epizodzie. Oddzielenie `reload_world()` od zwykłego resetu
epizodu zwiększa liczbę epizodów na godzinę, ale zostawia możliwość okresowego
reloadu, gdyby CARLA zaczęła kumulować zły stan. Zwracanie prostych typów,
trzymanie najnowszej klatki i wyciszenie logów zmniejszają ilość pracy w pętli
A3C.

## Reward

W `a3c_improved_1.py` worker brał nagrodę prawie bezpośrednio z
`CarlaEnv.step(...)` i dopisywał ją do `self.rewards`. Ta nagroda pochodziła z
legacy `reward_function(...)` używanej wewnątrz `carla_env.py`.

W aktualnym Hogwild finalna funkcja nagrody dla treningu jest w
`new_hogwild_carla_wrapper.py`, w metodzie `_shape_reward(...)`. Przepływ jest
taki:

1. Worker wybiera akcję i wywołuje `CarlaA3CWrapper.step(action)`.
2. Wrapper wykonuje akcję w CARLA, robi ticki i woła bazowe `CarlaEnv.step(...)`.
3. `CarlaEnv.step(...)` zwraca legacy reward oraz informacje środowiskowe:
   `route_distance`, `speed_value`, `distance_from_goal`, `done`.
4. Wrapper pobiera też liczbę kolizji i `last_invasion_counter`.
5. `_shape_reward(...)` zamienia te dane na finalny `reward_f`.
6. To `reward_f` trafia do `self.rewards` workera A3C i jest używane w n-step
   returns: `R = reward + gamma * R`.

Ważny szczegół: `CarlaEnv.step(...)` nadal liczy legacy reward nawet w trybie
`shaped`, bo wrapper potrzebuje z niego stanu `done` oraz wartości diagnostycznej
`legacy_reward`. Do uczenia trafia jednak `reward_f` z wrappera. Limit
`episode_max_decisions` jest nakładany już po obliczeniu shaped rewardu w
wrapperze, więc epizod może zakończyć się przez limit decyzji bez `goal_bonus`,
jeżeli bazowe środowisko nie uznało jeszcze celu za osiągnięty.

Domyślny tryb to `reward-mode shaped`. Tryb legacy można wymusić przez:

```bash
--reward-mode legacy
```

Wtedy `_shape_reward(...)` zwraca po prostu starą nagrodę z `CarlaEnv`.

W trybie `shaped` reward jest sumą komponentów:

- `progress`: dodatni, gdy auto zmniejsza dystans do celu; ujemny, gdy oddala się
  od celu. Pojedynczy krok jest obcięty do zakresu `[-5, 5]`.
- `target_speed`: nagradza jazdę blisko prędkości docelowej `20 km/h`; zbyt wolna
  albo zbyt szybka jazda daje słabszy wynik.
- `route_penalty`: mała ciągła kara za odległość od trasy.
- `time_penalty`: stała kara `-0.01` za każdy krok decyzyjny, żeby agent nie
  opłacał się przez stanie w miejscu.
- `goal_bonus`: duży bonus `+50` za zakończenie epizodu blisko celu.
- `collision_penalty`: kara `-50` za nową kolizję.
- `offroute_penalty`: kara `-25`, gdy odległość od trasy przekracza próg
  `10.0`.
- `lane_invasion_penalty`: kara `-5` za lane invasion w danym kroku.

Suma komponentów jest obcinana przez `reward_clip` do zakresu `[-50, 50]`.
Wrapper zapisuje też słownik `reward_components`, dzięki czemu w logach można
zobaczyć, czy agent dostaje nagrodę za postęp, czy głównie kary za off-route,
kolizje albo lane invasion.

W Hogwild każdy worker liczy reward lokalnie dla własnego procesu CARLA. Reward
nie jest współdzielony między workerami. Współdzielony jest dopiero efekt uczenia:
lokalny reward wpływa na lokalne returns, advantage i gradient, a gradient
aktualizuje globalny model.

Powód:

Stara nagroda dawała mniej gęsty sygnał uczenia. Shaped reward daje agentowi
informację w każdym kroku: czy zbliża się do celu, czy jedzie rozsądną prędkością,
czy trzyma się trasy, czy uderzył w coś i czy naruszył pas. Dzięki temu każdy
worker częściej produkuje użyteczny gradient. Tryb `legacy` pozostał jako punkt
porównawczy.

## Logging i monitoring

`a3c_improved_1.py` używało:

- loggera tekstowego,
- `log.csv`,
- W&B subprocess,
- profili `profile_worker_<id>.out`.

`new_hogwild_*` zapisuje strukturalne logi:

- `logs/metadata.json`,
- `logs/events.jsonl`,
- `logs/system.jsonl`,
- `logs/worker_<id>/episodes.jsonl`,
- `logs/worker_<id>/updates.jsonl`,
- `logs/worker_<id>/timing.jsonl`,
- opcjonalnie `logs/worker_<id>/steps.jsonl`,
- `logs/worker_<id>/system.jsonl`.

Dodatkowo:

- W&B ma stabilne `wandb_run_id.txt`, żeby resume trafiało do tego samego runa,
- `RunMonitor` loguje system, CARLA i GPU,
- `WorkerMonitor` loguje zasoby procesu workera,
- `TimingAccumulator` pokazuje czas w `env_reset`, `env_step`, `forward`,
  `backward`, `optim_update`, `checkpoint_save`.

Powód:

Przy analizie skalowania tekstowy log i CSV nie wystarczają. JSONL pozwala liczyć
epizody/h, kroki/s, czas resetu, czas środowiska, czas update'u i zużycie GPU bez
parsowania niestabilnych komunikatów stdout.

## Checkpointing i resume

Stara wersja zapisywała model do stałej ścieżki `MODEL_SAVE_PATH` co
`SAVE_INTERVAL` lokalnych epizodów workera.

Nowa wersja zapisuje checkpointy do katalogu runa:

- `checkpoint.pth`,
- `checkpoint_step.txt`,
- `best_checkpoint.pth`,
- opcjonalnie `checkpoints/worker_<id>/checkpoint.pth`,
- `resume_state.json`.

Checkpoint zawiera:

- model lub actor/critic,
- optymalizator,
- `global_step`,
- `global_episode`,
- `total_updates`,
- `last_checkpoint_boundary`,
- `best_reward`,
- `global_mean_reward`,
- średnie workerów,
- ostatnie rewardy.

Resume:

- wyszukuje najnowszy checkpoint,
- ładuje model i optymalizator,
- odtwarza liczniki,
- czyta `resume_state.json`,
- ostrzega, jeżeli ważne argumenty zmieniły się względem zapisanego runa,
- liczy aktywny czas treningu kumulatywnie między sesjami.

Powód:

Run na klastrze może zostać przerwany przez limit czasu albo awarię CARLA.
Checkpoint i resume muszą odtwarzać nie tylko wagi, ale też kontekst treningu.

## Supervisor i odporność na awarie

`a3c_improved_1.py` restartowało martwe workery, ale bez limitu restartów i bez
rollbacku modelu.

`new_hogwild_run_a3c.py` dodaje:

- licznik restartów per worker,
- `max_restarts_per_worker`,
- wykrywanie szybkich powtarzających się crashy,
- rollback do ostatniego checkpointu bez NaN,
- backoff restartu CARLA/workera,
- eventy `worker_start`, `worker_restart`, `rollback`, `worker_give_up`,
- graceful shutdown przez `shutdown_event`.

Powód:

Jeżeli worker zaczyna crashować natychmiast po restarcie, przyczyną może być
zepsuty globalny model albo checkpoint. Rollback daje szansę wrócić do ostatniego
dobrego stanu zamiast tracić cały run.

## Zapis obrazów

W starej wersji zapis obrazów był częściowo zakomentowany i zależny od
`StateObserver`.

Nowy wrapper zapisuje klatki przez `carla.Image.save_to_disk()` dla:

- konkretnych epizodów z `--save-episodes`,
- albo epizodów okresowych z `--save-episode-interval`.

Ścieżka:

```text
AV/A_to_B_GPU_34/episodes/<run_id>/<global_episode>-<port>/<step>.jpeg
```

Powód:

Zapis obrazów jest potrzebny diagnostycznie, ale nie może spowalniać każdego
epizodu. Dlatego jest domyślnie wyłączony i działa tylko dla wskazanych epizodów.

## Porównanie zachowania 1:1

| Obszar | `a3c_improved_1.py` | `new_hogwild_*` | Dlaczego zmieniono |
|---|---|---|---|
| Struktura | Jeden duży plik. | Moduły per odpowiedzialność. | Łatwiejszy debug i rozwój. |
| Konfiguracja | `settings.py` + globalne stałe + stałe ścieżki. | CLI dla run shape, uppercase defaults w entrypoincie. | Krótsze komendy i mniej rozproszone wartości. |
| Model | Oddzielny actor i critic. | Domyślnie wspólny actor-critic, legacy opcjonalne. | Mniej pracy na krok i spójna reprezentacja. |
| Optymalizator | `SharedAdam`. | Domyślnie `SharedRMSprop`, Adam opcjonalnie. | Klasyczny A3C i prostszy shared state. |
| Global step | Aktualizowany po epizodzie. | Aktualizowany co krok środowiska. | Dokładne budżety, LR decay i checkpointy. |
| Reset CARLA | Pełny `client.reload_world()` co epizod. | Domyślnie `reset_episode_state()` bez reloadu świata. | Największa redukcja stałego kosztu epizodu. |
| Długość epizodu | Limit środowiska `STEP_COUNTER = 200` decyzji workera. | `episode_max_decisions = 100` decyzji A3C oraz `action_repeat = 2`. | Więcej epizodów/h, ale epizod nie jest 1:1 tą samą jednostką co wcześniej. |
| Reward | Legacy reward z `CarlaEnv`. | Domyślnie shaped reward, legacy dostępne. | Gęstszy sygnał uczenia. |
| Gradient clipping | Zakomentowany. | Aktywny. | Stabilność. |
| Bufory gradientów | Współdzielone `.grad` w modelu globalnym. | Sklonowane, procesowo lokalne `.grad`; współdzielone są parametry i stan optymalizatora. | Mniej ryzyka wyścigów między workerami. |
| Tick/action flow | Jedna akcja w workerze, ręczne czyszczenie kolejki, dwa ticki, `env.step`. | Wrapper robi `action_repeat` razy `apply_action + tick`, potem pobiera najnowszą obserwację przez `env.step`. | Mniej logiki CARLA w workerze i jawny związek między decyzją A3C a tickami CARLA. |
| NaN guard | Brak. | Skip update i NaN-safe checkpoint. | Ochrona runa przed zatruciem modelu. |
| Checkpointy | Stała ścieżka poza katalogiem runa. | Checkpointy w katalogu runa. | Łatwiejsze resume i porównanie eksperymentów. |
| Resume | Ograniczone. | Model, optymalizator, liczniki, W&B id, aktywny czas. | Mniejsze ryzyko utraty pracy. |
| Logging/profiling | CSV/tekst/W&B oraz stale aktywny `cProfile` w workerze. | JSONL + W&B + monitoring + lekki timing per faza. | Mniejszy koszt hot path i lepsza analiza skalowania. |
| Supervisor | Restart workera. | Restart, limit, backoff, rollback. | Odporność na awarie CARLA/modelu. |
| Run dir | Brak spójnego zestawu artefaktów. | Logi, checkpointy, args, resume state, monitoring. | Reprodukowalność i audyt. |

## Co zostało świadomie kompatybilne

Zachowano:

- PyTorch i `torch.multiprocessing`,
- globalny model na CPU,
- lokalne workery na GPU,
- kopiowanie gradientów GPU -> CPU,
- `Categorical(logits=...)`,
- n-step A3C,
- Huber loss dla value,
- możliwość pracy w trybie testing,
- możliwość użycia starego actor/critic przez `--model-arch legacy`,
- możliwość starego rewardu przez `--reward-mode legacy`.

Powód:

Zmiany miały poprawić przepustowość, skalowanie, stabilność i obsługę runów, ale
nie zastępować całego algorytmu inną metodą RL.

## Aktualny typowy sposób uruchamiania

Dla 6 workerów na 6 GPU:

```bash
sbatch \
  --ntasks-per-node=7 \
  --cpus-per-task=7 \
  --mem=200G \
  --time=5:00:00 \
  --gpus=6 \
  --job-name=a3c-carla-6w_hogwild_gpu \
  /net/tscratch/people/plgbartoszkawa/AV/A_to_B_GPU_34/new_hogwild_train.slurm \
  -w 6 \
  -s 14 \
  --workers-per-gpu 1 \
  --servers-per-gpu 1
```

Dla resume:

```bash
sbatch --gpus=6 \
  /net/tscratch/people/plgbartoszkawa/AV/A_to_B_GPU_34/new_hogwild_train.slurm \
  -w 6 -s 14 --workers-per-gpu 1 --servers-per-gpu 1 \
  -r /net/tscratch/people/plgbartoszkawa/AV/A_to_B_GPU_34/runs/<run_dir>
```

## Ograniczenia i rzeczy do dalszej obserwacji

Run `a3c_hogwild_6w_20260505_114615_2566209` nie wykonał treningu: każdy worker
crashował w konstruktorze `CarlaEnv`, zanim powstał pierwszy epizod albo update.
Przyczyną była zła kolejność inicjalizacji `self.verbose`. Ten błąd został
naprawiony w `carla_env.py`, ale po tej poprawce nadal trzeba wykonać świeży
benchmark CARLA, żeby ocenić realne skalowanie.

Warto nadal obserwować:

- epizody/h dla 1, 6 i 12 workerów,
- średnią długość epizodu, bo nowy limit 100 decyzji i `action_repeat=2` zmieniają
  znaczenie samej liczby epizodów,
- `env_reset` i `env_step` w `timing.jsonl`,
- obciążenie GPU w `gpu_dmon.log` i `logs/system.jsonl`,
- liczbę restartów workerów w `events.jsonl`,
- udział czasu CARLA względem forward/backward,
- czy `--workers-per-gpu` i `--servers-per-gpu` są dobrane do konkretnego węzła.

Kilka zachowań, które warto mieć jasno opisane przy analizie wyników:

- `global_step` w nowej wersji liczy decyzje A3C, nie pojedyncze ticki CARLA.
  Przy `action_repeat=2` jeden global step oznacza dwa `step_apply_action()`.
- `episode_max_decisions` kończy epizod na poziomie wrappera. To przyspiesza
  zamykanie epizodów i update terminalny, ale może zakończyć epizod bez bonusu za
  cel, jeśli bazowe `CarlaEnv.step()` nie zwróciło jeszcze `done=True`.
- `world_reload_interval=0` oznacza brak okresowego pełnego reloadu świata po
  inicjalizacji środowiska. To jest szybkie, ale jeśli w długich runach CARLA
  zacznie kumulować zły stan, należy testować dodatnią wartość tego parametru.
- W trybie `shaped` legacy reward nadal jest liczony i logowany jako
  `legacy_reward`, ale gradienty uczą się z nagrody z `_shape_reward(...)`.

Najważniejszy praktyczny wniosek: aktualna wersja jest zaprojektowana tak, żeby
zachować rdzeń A3C z `a3c_improved_1.py`, ale mieć krótszą pętlę CARLA, gęstszy
sygnał rewardu, czytelne logi i dane potrzebne do dalszej analizy skalowania.
