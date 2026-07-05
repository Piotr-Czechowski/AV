# Raport: przejście z A2C do A3C/Hogwild w CARLA

Data analizy: 2026-06-17  
Zakres: implementacja A2C w `AV/A_to_B_GPU_34/a2c_rgb_try2.py`, implementacja A3C w `AV/A3C/new_hogwild_*`, wspólne pliki środowiska CARLA, akcje, sieci pomocnicze, skrypty uruchomieniowe i powiązany supervisor serwerów CARLA.

## 1. Najkrótsze podsumowanie

Przejście z A2C do A3C nie polega tylko na zmianie wzoru straty. To jest zmiana całej architektury treningu: z jednego procesu, jednej instancji CARLA i osobnych sieci actor/critic na wiele workerów, wiele portów CARLA, globalny model w pamięci współdzielonej CPU, lokalne kopie modelu na GPU/CPU, asynchroniczne aktualizacje Hogwild, osobny supervisor workerów, checkpointy po krokach, logi JSONL, monitoring zasobów i możliwość resume.

Wspólne pozostały kluczowe elementy domenowe: `CarlaEnv`, `ACTIONS.py`, `settings.py`, `utils.py`, `state_observer.py`, `carla_navigation/` oraz stary `nets/a2c.py` są takie same w porównywanych katalogach. To oznacza, że zmiana zachowania nie wynika z innego podstawowego świata CARLA ani innej listy akcji, tylko z nowego sposobu opakowania środowiska, zarządzania procesami i aktualizacji modelu.

Największe zmiany:

- A2C: jeden agent, jeden port `2000`, jeden rollout, synchroniczna aktualizacja po 5 decyzjach.
- A3C: N workerów, porty `2000 + 100 * worker_id`, lokalne rollouty domyślnie po 20 decyzjach, globalny model aktualizowany asynchronicznie.
- A2C: osobny `DiscreteActor` i osobny `Critic`, dwa optimizery Adam.
- A3C: jeden `SharedActorCritic` ze wspólnym trunk CNN i dwiema głowami: policy oraz value.
- A2C: CARLA server jest założony jako coś zewnętrznego; po awarii proces czeka i zakłada, że serwer wróci.
- A3C: SLURM ma uruchamiać supervisor wielu serwerów CARLA, sprawdzać porty, prowadzić dashboard i sprzątać procesy.
- A2C: reset CARLA domyślnie przeładowuje świat w każdym epizodzie.
- A3C: wrapper domyślnie nie przeładowuje całego świata co epizod, tylko niszczy/spawnuje aktorów; pełny reload jest konfigurowalny przez `world_reload_interval`.

## 2. Pliki i odpowiedzialności

### A2C

| Plik | Rola |
|---|---|
| `AV/A_to_B_GPU_34/a2c_rgb_try2.py` | Główna pętla A2C: agent, polityka, rollout, strata, zapis/odczyt, restart procesu po crashu. |
| `AV/A_to_B_GPU_34/carla_env.py` | Bazowe środowisko CARLA: scenariusze, route planner, sensory, reset, step, reward. |
| `AV/A_to_B_GPU_34/nets/a2c.py` | Stare klasy `DiscreteActor` i `Critic`, używane przez A2C. |
| `AV/A_to_B_GPU_34/ACTIONS.py` | 10 dyskretnych akcji sterowania. |
| `AV/A_to_B_GPU_34/settings.py` | Stałe dla A2C: port, kamera, gamma, lr, scenario, logging. |
| `AV/A_to_B_GPU_34/utils.py` | Funkcja nagrody `reward_function` i kolorowe printy. |
| `AV/A_to_B_GPU_34/state_observer.py` | Zapisywanie klatek i nakładek diagnostycznych dla wybranych epizodów. |

### A3C/Hogwild

| Plik | Rola |
|---|---|
| `AV/A3C/new_hogwild_train_a3c_carla.py` | Entry point treningu A3C: CLI, defaulty, globalna konfiguracja, W&B subprocess, resume, monitoring, start supervisora workerów. |
| `AV/A3C/new_hogwild_a3c.py` | Rdzeń algorytmu: `SharedActorCritic`, `GlobalNetwork`, `A3CWorker`, shared optimizer, gradient transfer, checkpointing. |
| `AV/A3C/new_hogwild_carla_wrapper.py` | Adapter między `CarlaEnv` a workerem A3C: normalizacja obserwacji, action repeat, reward shaping, reconnect, statystyki epizodu. |
| `AV/A3C/new_hogwild_run_a3c.py` | Supervisor workerów uczących: restart, limit restartów, rapid crash detection, rollback globalnego modelu. |
| `AV/A3C/new_hogwild_training_logger.py` | JSONL logger: epizody, update'y, kroki, timing, eventy, metadata. |
| `AV/A3C/new_hogwild_system_monitor.py` | Monitoring CPU/RAM/GPU/CARLA/workerów do JSONL. |
| `AV/A3C/new_hogwild_timing_utils.py` | Pomiar czasu faz pętli: sync, reset, forward, env_step, backward, optimizer. |
| `AV/A3C/new_hogwild_prepare_output_dir.py` | Tworzenie katalogu runu i zapis argumentów. |
| `AV/A3C/new_hogwild_train.slurm` | Pipeline HPC: start serwerów CARLA, czekanie na porty, start treningu, tail logów, cleanup. |
| `AV/A3C/new_hogwild_train_paths.json` | Faktyczne ścieżki dla SLURM: venv, katalog projektu, skrypt multiserwerów, W&B. |
| `/net/tscratch/people/plgbartoszkawa/carla_athena_multiserver_v3.py` | Aktywny supervisor wielu serwerów CARLA wskazany przez `new_hogwild_train_paths.json`. |
| `/net/tscratch/people/plgbartoszkawa/carla_athena/dashboard.py` i `AV/A3C/templates/dashboard.html` | Dashboard monitorujący porty, GPU, zasoby i logi serwerów CARLA. |

## 3. Co zostało takie samo

Te elementy są identyczne lub logicznie wspólne:

- `carla_env.py`: ten sam bazowy świat, scenariusze, route planner, sensory i reward legacy.
- `ACTIONS.py`: 10 dyskretnych akcji:
  `forward`, `forward_left`, `forward_right`, `brake`, `brake_left`, `brake_right`, `forward_slight_left`, `forward_slight_right`, `brake_slight_left`, `brake_slight_right`.
- `settings.py`: nadal definiuje domyślny port `2000`, kamerę `semantic`, `GAMMA=0.99`, `LR=1e-4`, scenario `[14]`.
- `utils.reward_function`: legacy reward oparty m.in. o kolizję/offroute, sinusoidalny reward prędkości z pikiem ok. 20 km/h i karę/nagrodę za dystans do trasy.
- `carla_navigation/`: route planning i wykrywanie manewrów są wspólne.
- `state_observer.py`: nadal istnieje, ale A3C wrapper używa go dużo oszczędniej.

Wniosek: nowy A3C nie zmienia podstawowej definicji zadania jazdy. Zmienia sposób równoległego zbierania doświadczenia, aktualizacji modelu, komunikacji z CARLA, odporności na awarie i obserwowalności treningu.

## 4. Architektura algorytmu

### A2C: jeden agent, osobny actor i critic

W `a2c_rgb_try2.py` klasa `DeepActorCriticAgent` tworzy:

- jedną instancję `CarlaEnv` na porcie z `settings.PORT`,
- osobny `DeepDiscreteActor`,
- osobny `DeepCritic`,
- dwa optimizery Adam,
- lokalne listy `trajectory` i `rewards`.

Model A2C korzysta ze starego `nets/a2c.py`. `DiscreteActor` i `Critic` mają bardzo podobną, zdublowaną architekturę CNN: warstwy Conv2d z BatchNorm, ReLU, dodatkowy Conv2d do 512 kanałów, `AdaptiveAvgPool2d((4, 4))`, osobne gałęzie dla prędkości i manewru, a potem MLP. Actor kończy się liczbą akcji, critic jedną wartością.

Ważny detal: w A2C actor i critic liczą swoje własne cechy obrazu. To podwaja koszt ekstrakcji wizualnej i pozwala actorowi/criticowi rozjeżdżać reprezentacje.

### A3C: globalny `SharedActorCritic`

A3C w `new_hogwild_a3c.py` wprowadza pojedynczy model:

- wspólny CNN trunk,
- gałąź `speed_fc`,
- gałąź `maneuver_fc`,
- wspólny MLP trunk,
- głowa `policy`,
- głowa `value`.

Globalny model żyje na CPU i jest przeniesiony do pamięci współdzielonej przez `share_memory()`. Każdy worker ma lokalną kopię modelu na przypisanym urządzeniu (`cuda:n` albo CPU). Worker synchronizuje lokalne parametry z globalnymi, zbiera rollout, liczy gradient lokalnie, kopiuje gradienty do modelu globalnego i wykonuje krok optimizera globalnego.

To jest prawdziwy Hogwild: parametry i stan optimizera są współdzielone, gradienty są proces-lokalne, a update domyślnie nie jest blokowany lockiem (`hogwild_lock_updates=False`). Lock istnieje jako opcja diagnostyczna.

## 5. Różnice w przepływie treningu

### A2C: przepływ epizodu

1. Proces `handle_crash()` inicjalizuje W&B, tworzy `DeepActorCriticAgent`, próbuje wczytać checkpoint z twardo wpisanej ścieżki.
2. Pętla epizodów działa do `episode >= 42030`.
3. Każdy epizod resetuje `state_observer` i wywołuje `environment.reset(...)`.
4. Obraz jest dzielony przez `255.0`, prędkość przez `100.0`.
5. Manewr jest pobierany z `environment.car_decisions` i ręcznie aktualizowany po opuszczeniu skrzyżowania.
6. Agent próbkuje akcję z `Categorical(logits=actor(...))`.
7. Akcja jest aplikowana dwa razy:
   - `step_apply_action(action)`,
   - `world.tick()`,
   - ponownie `step_apply_action(action)`,
   - ponownie `world.tick()`.
8. `environment.step(...)` zwraca nowy stan, reward, done, route distance, speed i dystans do celu.
9. Reward jest dopisywany do rollouta.
10. Update następuje co 5 kroków albo na końcu epizodu.
11. Po epizodzie model jest zapisywany zawsze, do tej samej bazowej ścieżki.
12. Jeżeli proces padnie bez wpisu do `results_queue`, zewnętrzna pętla usuwa `core.*`, czeka `CARLA_SERVER_START_PERIOD` i uruchamia proces od nowa.

### A3C: przepływ runu

1. `new_hogwild_train.slurm` przygotowuje środowisko HPC, venv, zmienne CUDA/PyTorch i katalog output.
2. Ten sam skrypt ma uruchomić N serwerów CARLA przez `MULTISERVER_SCRIPT`, poczekać aż porty będą w stanie LISTEN i dopiero wtedy uruchomić trening.
3. `new_hogwild_train_a3c_carla.py` parsuje CLI, ustawia seed, przydziela workery do GPU, tworzy `GlobalNetwork`, logger, monitor zasobów i opcjonalny subprocess W&B.
4. `run_with_restart()` startuje N procesów `A3CWorker`.
5. Każdy worker:
   - ustawia seed `seed + worker_id * 1009`,
   - ogranicza wątki BLAS/OpenMP do 1,
   - inicjalizuje lokalny model,
   - tworzy `CarlaA3CWrapper` na swoim porcie,
   - resetuje środowisko,
   - zbiera rollout.
6. Każdy krok workera zwiększa globalny licznik kroków.
7. Update następuje po `rollout_length=20` decyzjach albo na końcu epizodu.
8. Worker liczy n-step return z bootstrapem z critic, normalizuje advantage, liczy policy loss, value loss i entropy loss.
9. Gradienty są clipowane, sprawdzane pod kątem NaN, kopiowane do globalnego modelu i aplikowane shared optimizerem.
10. Worker synchronizuje lokalny model z globalnym po każdym update lub według `sync_every_n_updates`.
11. Supervisor workerów restartuje martwe workery, wykrywa szybkie serie crashy i może zrobić rollback do ostatniego poprawnego checkpointu.
12. Run zapisuje JSONL logi, timing, system metrics, checkpointy i resume state.

## 6. Parametry i hiperparametry

| Obszar | A2C | A3C/Hogwild | Zmiana znaczenia |
|---|---:|---:|---|
| Liczba learnerów | 1 proces uczący | `num_workers`, domyślnie Python: 2; SLURM default: 1; można podać `-w` | Przejście do równoległego zbierania doświadczeń. |
| Liczba serwerów CARLA | 1 zewnętrzny serwer | Docelowo 1 serwer na worker | Środowisko skaluje się razem z workerami. |
| Port CARLA | `settings.PORT = 2000` | `start_port + port_step * worker_id`, domyślnie `2000 + 100*i` | Każdy worker ma własny port. |
| Scenario | `settings.SCENARIO = [14]` | `--scenario`, default `[14]` | Przeniesione z ustawień globalnych do CLI. |
| Kamera | `semantic` | `semantic` | Bez zmiany defaultu. |
| Rozdzielczość env | `250 x 250` | `--res 250` | Bez zmiany wartości, ale A3C robi walidację kształtu w wrapperze. |
| Akcje | 10 dyskretnych | 10 dyskretnych | Bez zmiany przestrzeni akcji. |
| Gamma | `0.99` | `0.99` | Bez zmiany. |
| Learning rate | `1e-4` stałe | `1e-4` z liniowym decay do 0 | A3C wygasza LR wraz z globalnymi krokami. |
| Optimizer | 2x Adam | `shared-rmsprop` default, `shared-adam` opcjonalnie | Stan optimizera jest współdzielony między procesami. |
| Weight decay | `1e-2` | `0.0` | A3C usuwa regularizację L2 z defaultu. |
| Długość rollouta | 5 decyzji | 20 decyzji | A3C rzadziej aktualizuje model na worker, ale ma wiele workerów. |
| Entropia | `USE_ENTROPY=True`, bez jawnego współczynnika | beta schedule `0.02 -> 0.002` przez 60% kroków | A3C kontroluje eksplorację stabilniej. |
| Normalizacja advantage | brak | `normalize_advantages=True` | A3C zmniejsza wariancję gradientu polityki. |
| Clip gradientu | brak | `max_grad_norm=5.0` | A3C ma ochronę przed wybuchami gradientu. |
| Value loss coef | brak jawnego mnożnika | `value_loss_coef=1.0` | A3C jawnie parametryzuje wagę value loss. |
| Reward scale | brak | `reward_scale=0.0`, gdzie 0 oznacza brak skalowania | A3C ma opcję skalowania bez zmiany defaultu. |
| Limit treningu | epizody do `42030` | global steps `10_000_000` | A3C steruje treningiem przez kroki globalne. |
| Limit epizodu | `CarlaEnv.STEP_COUNTER=200` plus cutoff `sum(rewards) <= -6` | `episode_max_decisions=100` plus env step counter; przy `action_repeat=2` daje porównywalnie 200 niskopoziomowych akcji | A3C ma jawny limit decyzji workera i usuwa cutoff po sumie rollout reward. |
| Action repeat | ręcznie 2x w pętli A2C | `action_repeat=2` w wrapperze | Ta sama idea, ale w A3C opakowana parametrem. |
| Reload świata | domyślnie każdy reset | domyślnie brak pełnego reloadu; opcja `world_reload_interval` | A3C zmniejsza koszt resetu, ale bardziej polega na czyszczeniu aktorów. |
| Checkpoint | co epizod, jedna ścieżka | co `save_frequency=100000` kroków, best checkpoint, resume state | A3C jest odporniejsze na długie runy i resume. |
| Logging | W&B bezpośrednio z procesu uczącego | JSONL per worker + W&B subprocess + system monitor | A3C odciąża pętlę treningu i daje post-mortem. |

## 7. Różnice w stracie i aktualizacji gradientów

### A2C

A2C liczy n-step target:

`G_t = r_t + gamma * r_{t+1} + ... + gamma^n * V(s_{t+n})`

Potem:

- `td_err = td_target - critic_prediction`,
- actor loss: `-log_prob(action) * td_err`,
- critic loss: `smooth_l1_loss(critic_prediction, td_target)`,
- actor optimizer robi backward i step,
- critic optimizer robi osobny backward i step.

Szczegóły implementacyjne warte odnotowania:

- Advantage w actor loss nie jest jawnie odcinany przez `.detach()`. Ponieważ `critic_prediction` jest tensorem z graphu critika, actor backward może wyliczać też gradienty dla critika, choć potem są czyszczone przed critic backward. To nie musi aktualizować critika w actor optimizerze, ale jest niepotrzebnym sprzężeniem grafów i kosztem.
- Entropia jest dodawana jako `- self.action_distribution.entropy().mean()` bez współczynnika beta. W praktyce współczynnik jest równy 1.0, co jest bardzo duże w porównaniu z typowymi wartościami A3C.
- `self.action_distribution` oznacza ostatnią dystrybucję, a nie batch entropii dla każdego kroku rollouta.
- Gradient clipping i NaN guard nie występują.

### A3C

A3C liczy:

- return z bootstrapem z lokalnego critika,
- `advantages = returns - values.detach()`,
- opcjonalną normalizację advantages,
- `policy_loss = -(log_probs * advantages).mean()`,
- `value_loss = value_loss_coef * smooth_l1_loss(values, returns)`,
- `total_loss = policy_loss + value_loss - entropy_coef * entropy_mean`.

Następnie:

- lokalny model robi `backward()`,
- gradienty są clipowane,
- warstwy z NaN gradientami powodują skip update'u i resync z globalnego modelu,
- gradienty są kopiowane do globalnego modelu CPU,
- shared optimizer robi update globalnych parametrów,
- globalny LR jest ustawiany według liniowego harmonogramu.

To jest bardziej stabilne i bardziej typowe dla A3C: worker liczy gradient lokalnie, globalny model jest jedynym źródłem prawdy, a worker okresowo synchronizuje kopię.

## 8. Integracja ze środowiskiem CARLA

### A2C: bezpośrednie użycie `CarlaEnv`

A2C operuje bezpośrednio na `CarlaEnv`:

- resetuje `state_observer`,
- wywołuje `environment.reset(save_image=..., episode=...)`,
- ręcznie normalizuje obraz i prędkość,
- ręcznie śledzi manewr przez `planner.on_junction(...)`,
- ręcznie czyści `image_queue`,
- ręcznie wykonuje action repeat przez dwa `step_apply_action()` i dwa `world.tick()`,
- wywołuje `environment.step(...)`.

Kod treningowy zna dużo detali środowiska: kolejkę obrazów, planner, vehicle, `state_observer`, `car_decisions`, `world.tick()`. Granica między agentem i środowiskiem jest cienka.

### A3C: `CarlaA3CWrapper`

A3C wprowadza warstwę adaptera:

`reset() -> (state, speed, maneuver)`  
`step(action) -> (next_state, next_speed, next_maneuver, reward, done, info)`  
`reconnect()`  
`is_server_alive()`

Wrapper przejmuje:

- konwersję obserwacji do `np.float32 [3,H,W]` w zakresie `[0,1]`,
- konwersję speed do float i dzielenie przez `100.0`,
- walidację kształtu obrazu,
- action repeat,
- zapis klatek,
- aktualizację manewru,
- statystyki epizodu,
- reward shaping lub legacy reward,
- timeout/reconnect,
- limit decyzji epizodu.

To jest duża poprawa architektoniczna: worker A3C nie musi znać szczegółów sensorów i ticków CARLA.

### Reset świata

To jedna z najbardziej praktycznych zmian:

- A2C wywołuje `CarlaEnv.reset(...)` bez argumentu `reload_world`, więc używa domyślnego `reload_world=True`. Każdy epizod niszczy aktorów i przeładowuje świat.
- A3C wrapper wylicza `full_reload = world_reload_interval > 0 and episode % world_reload_interval == 0`. Default `world_reload_interval=0`, więc `CarlaEnv.reset(..., reload_world=False)` resetuje stan epizodu i aktorów bez pełnego reloadu świata.

Efekt:

- A3C powinno mieć szybsze resety.
- A3C jest mniej kosztowne dla wielu równoległych serwerów.
- Ryzyko: jeżeli `reset_episode_state()` nie czyści czegoś, co pełny `reload_world()` czyścił, błędy mogą akumulować się między epizodami.

## 9. Reward: legacy i shaped

Bazowy `CarlaEnv` nadal liczy legacy reward przez `utils.reward_function`. W A3C wrapper może:

- przepuścić legacy reward bez zmian (`reward_mode='legacy'`, obecny default w `new_hogwild_train_a3c_carla.py`),
- albo policzyć shaped reward (`reward_mode='shaped'`).

Shaped reward w wrapperze rozkłada nagrodę na komponenty:

- progress do celu,
- bliskość target speed,
- kara za dystans od trasy,
- kara czasu,
- bonus celu,
- kara kolizji,
- kara offroute,
- kara lane invasion,
- clipping.

Ważne: default w kodzie A3C to obecnie `legacy`, mimo że wrapper ma domyślne `reward_mode='shaped'`. Entry point nadpisuje to na `legacy`. Dlatego w typowym uruchomieniu A3C reward powinien być zgodny z A2C, a shaped reward jest opcją eksperymentalną.

## 10. Uruchamianie serwerów CARLA

### A2C

A2C nie startuje serwera CARLA w analizowanym pliku. `CarlaEnv` ma funkcję `start_carla_server(...)`, ale jej użycie jest zakomentowane. `a2c_rgb_try2.py` zakłada, że serwer działa na `localhost:settings.PORT`, czyli zwykle `2000`.

Mechanizm awarii:

- proces treningowy działa wewnątrz pętli `while 1`,
- jeśli proces padnie i nie zapisze wyniku w `results_queue`, wrapper uznaje crash,
- usuwa `core.*`,
- czeka `CARLA_SERVER_START_PERIOD`, default 30 s,
- uruchamia trening jeszcze raz.

Ten kod nie restartuje samego CARLA. On tylko czeka z założeniem, że serwer zostanie zrestartowany z zewnątrz.

### A3C

A3C ma pełniejszy pipeline:

1. `new_hogwild_train.slurm` startuje supervisor CARLA przez `MULTISERVER_SCRIPT`.
2. Przekazuje docelowo:
   - liczbę serwerów,
   - liczbę serwerów na GPU,
   - GPU start,
   - start port,
   - port step.
3. Czeka aż porty `START_PORT + i * PORT_STEP` będą `LISTEN` według `lsof`.
4. Dopiero potem uruchamia `new_hogwild_train_a3c_carla.py`.
5. Przy wyjściu robi cleanup procesu treningowego, supervisora CARLA, `nvidia-smi dmon`, taila logów i procesów `CarlaUE4`.

Aktywny supervisor `/net/tscratch/people/plgbartoszkawa/carla_athena_multiserver_v3.py`:

- uruchamia CARLA w Apptainerze z `--nv`,
- używa obrazu `/net/tscratch/people/plgbartoszkawa/carla_0.9.15.sif`,
- odpala `/home/carla/CarlaUE4.sh -RenderOffScreen -nosound --carla-server`,
- dodaje `-carla-rpc-port=<port>` i `-graphicsadapter=<gpu_id>`,
- monitoruje porty przez `lsof`,
- zapisuje `server_logs/server_run_...`,
- prowadzi dashboard Flask na porcie 5000,
- restartuje pojedynczą instancję CARLA po wyjściu.

### Istotna niespójność w obecnym stanie plików

W `AV/A3C/new_hogwild_train_paths.json` aktywny `MULTISERVER_SCRIPT` to:

`/net/tscratch/people/plgbartoszkawa/carla_athena_multiserver_v3.py`

Ten plik ma stałe:

- `NUM_SERVERS = 8`,
- `SERVERS_PER_GPU = 2`,
- `START_PORT = 2000`,
- `PORT_STEP = 100`.

Nie widać w nim parsera argumentów CLI, a `new_hogwild_train.slurm` próbuje przekazać mu `--num-servers`, `--servers-per-gpu`, `--start-port`, `--port-step`. To oznacza, że aktywny supervisor może ignorować wartości ze SLURM.

Dowód z istniejącego runu: katalog `AV/A3C/runs/a3c_hogwild_4w_20260615_201914_2675449` sugeruje 4 workery, ale `carla_servers.log` zaczyna od `Starting 8 CARLA servers`. W tym samym runie są tylko `gpu_dmon.log` i `carla_servers.log`, bez właściwego `a3c_training.log` oraz bez `logs/metadata.json`, więc wygląda to na run zatrzymany przed startem treningu albo na nieudany start serwerów.

To nie przekreśla implementacji A3C, ale jest ważnym ryzykiem operacyjnym: liczba workerów i liczba serwerów CARLA mogą się rozjechać.

## 11. Checkpointing, resume i odporność na awarie

### A2C

A2C:

- wczytuje z jednej twardo wpisanej ścieżki,
- zapisuje po każdym epizodzie do jednej twardo wpisanej ścieżki,
- zapis obejmuje actor, critic, optimizery i kilka metryk,
- nie ma checkpointów po krokach,
- nie ma resume state z czasem aktywnego treningu,
- nie ma rollbacku po NaN,
- nie ma rozróżnienia checkpointu najlepszego i ostatniego.

### A3C

A3C:

- zapisuje globalny model i globalny optimizer,
- zapisuje global step, global episode, total updates, reward buffers, worker means,
- zapisuje `checkpoint.pth` i `checkpoint_step.txt`,
- zapisuje `resume_state.json`,
- zapisuje `best_checkpoint.pth` przy nowym najlepszym rewardzie,
- może zapisywać per-worker checkpointy,
- odmawia zapisu, jeśli parametry globalne mają NaN,
- przy rapid crashach próbuje rollback do ostatniego checkpointu bez NaN,
- przy resume ostrzega, jeśli kluczowe hiperparametry zmieniły się względem zapisanych argumentów.

To jest duża zmiana jakościowa: A3C jest projektowane pod długie HPC runy, przerwania sygnałami, restart i post-mortem.

## 12. Logging i obserwowalność

### A2C

A2C loguje:

- W&B bezpośrednio z procesu treningowego,
- `actor_loss`, `critic_loss`,
- histogramy gradientów actor/critic,
- `step_reward`,
- metryki epizodu: steps, duration, reward, episode, mean reward, max speed, distance from goal.

Logika zapisu klatek:

- lista epizodów do zapisu jest twardo wpisana,
- zapisywane są raw frame, panel z wartościami i obraz łączony,
- ścieżki są zakodowane jako `A_to_B_GPU_34/images/...`.

### A3C

A3C dodaje:

- `logs/metadata.json`,
- `logs/events.jsonl`,
- `logs/system.jsonl`,
- `logs/worker_<id>/episodes.jsonl`,
- `logs/worker_<id>/updates.jsonl`,
- `logs/worker_<id>/steps.jsonl` opcjonalnie,
- `logs/worker_<id>/timing.jsonl`,
- `logs/worker_<id>/system.jsonl`,
- W&B w osobnym procesie przez kolejkę,
- timing faz pętli,
- monitor CPU/RAM/GPU/CARLA/workerów,
- `gpu_dmon.log` z SLURM,
- logi supervisora CARLA i dashboard.

Efekt: A3C jest znacznie łatwiejsze do diagnozowania, zwłaszcza gdy problemem jest nie sam model, tylko timeout kamery, niedziałający port CARLA, brak pamięci GPU albo crash workera.

## 13. Integracja z GPU i multiprocessing

### A2C

A2C ustawia `device = settings.SHOULD_USE_CUDA` i używa jednego urządzenia. `torch.multiprocessing` jest użyte głównie jako wrapper procesu restartowalnego, nie jako prawdziwa równoległość algorytmiczna.

### A3C

A3C:

- globalny model trzyma na CPU,
- lokalne modele workerów trzyma na przypisanym urządzeniu,
- przypisuje workery do GPU przez `workers_per_gpu` i `worker_gpu_start`,
- ogranicza każdy worker do jednego wątku CPU/BLAS,
- używa `spawn`,
- używa shared memory dla parametrów i stanu optimizera,
- pozwala na wielu workerów na jednym GPU albo rozłożenie na wiele GPU.

To zmienia wykorzystanie sprzętu: GPU nie jest już tylko miejscem jednego modelu, ale zasobem dla wielu lokalnych rolloutów, a CPU przechowuje wspólny model i optimizer.

## 14. Najważniejsze skutki przejścia A2C -> A3C

### Plusy

- Więcej doświadczenia zbieranego równolegle.
- Lepsze wykorzystanie wielu CARLA serverów i GPU.
- Mniej blokowania treningu przez pojedynczy epizod.
- Lepsza odporność na crash workera.
- Lepszy logging, resume i diagnostyka.
- Stabilniejsze straty: detach advantage, entropy schedule, gradient clipping, NaN guard.
- Czystsza separacja: worker nie musi znać szczegółów `CarlaEnv`.
- Możliwość reward shaping bez ruszania bazowego `CarlaEnv`.

### Koszty i ryzyka

- Większa złożoność operacyjna: wiele procesów, portów, loggerów, katalogów.
- Ryzyko rozjazdu liczby workerów i liczby serwerów CARLA.
- Aktywny multiserver może ignorować argumenty SLURM, bo ma hardcoded `NUM_SERVERS=8`.
- Asynchroniczny Hogwild bez locka może wprowadzać niedeterministyczność update'ów.
- Brak pełnego reloadu świata co epizod może ujawniać błędy czyszczenia aktorów.
- Checkpointy A2C legacy nie są kompatybilne z A3C `SharedActorCritic`; A3C loader jawnie odrzuca format legacy bez klucza `model`.
- W A3C default Python `num_workers=2`, default SLURM `NUM_WORKERS=1`, aktywny multiserver `NUM_SERVERS=8`; to trzeba ujednolicić przed produkcyjnymi runami.

## 15. Mapa zmiany mentalnej

### A2C

`agent -> CarlaEnv -> rollout 5 kroków -> actor update -> critic update -> save episode -> następny epizod`

To jest sekwencyjne. Jeden epizod i jedna instancja środowiska decydują o tempie treningu.

### A3C

`SLURM -> N serwerów CARLA -> GlobalNetwork CPU -> N workerów -> lokalne rollouty -> asynchroniczne gradienty -> globalny optimizer -> checkpoint/logging/restart`

To jest system rozproszony w obrębie jednego joba HPC. Algorytm RL stał się częścią większej infrastruktury treningowej.

## 16. Co warto poprawić przed kolejnym długim treningiem

1. Ujednolicić liczbę workerów i serwerów.
   - `new_hogwild_train.slurm` przekazuje `--num-servers`, ale aktywny `carla_athena_multiserver_v3.py` ma hardcoded `NUM_SERVERS=8`.
   - Albo dodać argparse do supervisora, albo wskazać wersję, która już obsługuje CLI.

2. Ujednolicić ścieżkę multiservera.
   - `.example` wskazuje `/net/tscratch/people/plgbartoszkawa/carla_athena/carla_athena_multiserver_v3_hogwild.py`.
   - Faktyczny JSON wskazuje `/net/tscratch/people/plgbartoszkawa/carla_athena_multiserver_v3.py`.
   - W katalogu `carla_athena/` jest też `carla_athena_multiserver_v3.py`, ale nie jest identyczny z aktywnym plikiem.

3. Jawnie zdecydować reward mode.
   - Obecny entry point A3C ustawia `DEFAULT_REWARD_MODE='legacy'`.
   - Wrapper ma shaped reward gotowy, ale nie jest defaultem z entry pointu.

4. Zdecydować, czy brak pełnego reloadu świata jest celowy.
   - Default `world_reload_interval=0` przyspiesza trening.
   - Dla diagnostyki można uruchomić kilka runów z okresowym reloadem, np. co 10 lub 50 epizodów.

5. Przenieść twarde wartości A2C do konfiguracji, jeśli A2C ma dalej służyć jako baseline.
   - Ścieżki checkpointów,
   - lista epizodów do zapisu,
   - limit 42030,
   - cutoff `sum(agent.rewards) <= -6`.

6. W A2C odciąć advantage w actor loss i dodać jawny entropy coefficient, jeśli porównanie ma być metodycznie czystsze.

7. W A3C zapisać w raportach runu także konfigurację supervisora CARLA, nie tylko konfigurację learnerów.

## 17. Wnioski końcowe

Przejście z A2C do A3C jest przejściem od eksperymentalnego, jednowątkowego treningu do pełnego pipeline'u HPC. Algorytmicznie najważniejsze są: wiele workerów, globalna sieć w shared memory, lokalne rollouty, shared optimizer, entropy schedule, gradient clipping i resume. Inżynieryjnie najważniejsze są: wrapper CARLA, start wielu serwerów, monitor portów, JSONL logging, restart workerów i checkpointy po globalnych krokach.

Największą rzeczą do naprawy przed traktowaniem A3C jako stabilnego następcy A2C jest spójność uruchamiania CARLA: liczba workerów w treningu, liczba portów oczekiwanych przez SLURM i liczba serwerów uruchamianych przez aktywny multiserver muszą być tym samym kontraktem.

