# Analiza systemu logowania w implementacji A3C (`AV/A3C`)

Dokument powstał na podstawie ręcznej, dokładnej analizy kodu w katalogu `AV/A3C` (a nie automatycznego streszczenia). Plik `logging-analisys.md` dołączony do zgłoszenia został zweryfikowany linia po linii — w sekcji [4. Błędy i nieścisłości w `logging-analisys.md`](#4-błędy-i-nieścisłości-w-logging-analisysmd) wypisano wszystkie znalezione rozbieżności. **Najważniejsze odkrycie: do W&B trafiają wyłącznie zagregowane metryki epizodu — dane per-step, per-update (straty, gradienty) i timing NIGDY nie są wysyłane do W&B**, wbrew temu, co sugerował automatycznie wygenerowany plik.

Pliki źródłowe analizowane w tym dokumencie (wszystkie w `AV/A3C/`):

- `new_hogwild_training_logger.py` — klasa `TrainingLogger` (zapis lokalny JSONL)
- `new_hogwild_train_a3c_carla.py` — punkt wejścia, proces W&B, config, `metadata.json`, `resume_state.json`
- `new_hogwild_run_a3c.py` — nadzorca workerów, rollback, własny zapis zdarzeń
- `new_hogwild_a3c.py` — `GlobalNetwork` (checkpointy), `A3CWorker` (pętla treningowa, wywołania loggera, kolejka do W&B)
- `new_hogwild_system_monitor.py` — `RunMonitor`, `WorkerMonitor`
- `new_hogwild_timing_utils.py` — `TimingAccumulator`
- `new_hogwild_prepare_output_dir.py` — zapis argumentów uruchomienia
- `new_hogwild_carla_wrapper.py` — budowa `info` dict (komponenty nagrody, prędkość, dystans do celu)
- `carla_env.py`, `utils.py` — starszy kod (`ColoredPrint`), używany tylko do wypisywania na konsolę

---

## 1. Co jest logowane do W&B i gdzie w kodzie

### 1.1. Hiperparametry (raz, na starcie)

`wandb.init(..., config=_config_to_dict(config))` — `new_hogwild_train_a3c_carla.py:140-149`, wewnątrz `wandb_logger_process()`. Loguje **cały** `SimpleNamespace` argumentów CLI (wszystkie `--flagi`, w tym stałe algorytmiczne dodane w `_apply_config_defaults()`).

### 1.2. Metryki treningowe — TYLKO poziom epizodu

Cały mechanizm transportu do W&B opiera się o `mp.Queue` (`log_queue`), tworzoną w `new_hogwild_train_a3c_carla.py:503-504` i konsumowaną przez proces `wandb_logger_process()` (`new_hogwild_train_a3c_carla.py:135-176`).

**Jedyne miejsce w całym kodzie, które coś wkłada do tej kolejki, to `A3CWorker._log_episode()`** w `new_hogwild_a3c.py:746-776`. Dzieje się to raz na zakończony epizod, dla każdego workera osobno. Budowany jest ręczny słownik (niezależny od tego, co trafia do `episodes.jsonl`!):

| Klucz w W&B (po prefiksowaniu) | Źródło |
|---|---|
| `episode` | `global_episode` (bez prefiksu workera — patrz niżej) |
| `global_step` | `global_t` (bez prefiksu workera) |
| `worker/<id>/reward` | `episode_total_reward` |
| `worker/<id>/local_mean_reward` | średnia krocząca z ostatnich 100 epizodów danego workera |
| `worker/<id>/global_mean_reward` | średnia krocząca po wszystkich workerach (`GlobalNetwork.update_stats`) |
| `worker/<id>/episode_length` | liczba kroków w epizodzie |
| `worker/<id>/total_steps` | licznik kroków danego workera od startu procesu |
| `worker/<id>/max_speed_kmh` | maks. prędkość w epizodzie (jeśli dostępna) |
| `worker/<id>/min_route_dist` | min. odległość od trasy (jeśli dostępna) |
| `worker/<id>/distance_from_target` | dystans do celu na koniec epizodu (jeśli dostępny) |
| `worker/<id>/reward_component_<nazwa>` | suma każdego składnika nagrody w epizodzie (jeśli `reward_mode=shaped`) |
| `worker/<id>/most_chosen_action_pct` | udział najczęściej wybieranej akcji |
| `worker/<id>/action_<i>` | liczba wystąpień akcji `i` w epizodzie |

Logika prefiksowania: w `wandb_logger_process()` (`new_hogwild_train_a3c_carla.py:158-165`) każdy klucz oprócz `episode` i `global_step` dostaje prefiks `worker/<id>/`.

**Ważne:** `pi_loss`, `v_loss`, `total_loss`, `gradient_norm`, `lr`, `entropy_coef`, statystyki `advantages`/`values`/`entropies`, dane per-step (`action`, `value`, `entropy`, `reward`, `done`, `speed_kmh`...) oraz `timing.jsonl` **nie istnieją w W&B** — metody `TrainingLogger.log_step()`, `log_update()`, `log_timing()`, `log_event()` (`new_hogwild_training_logger.py:85-223`) w ogóle nie mają dostępu do `log_queue` i tylko piszą do plików lokalnych.

---

## 2. Co jest logowane lokalnie i gdzie w kodzie

Wszystko poniżej ląduje w `run_output_dir` (katalog runa), niezależnie od tego, czy W&B jest włączony.

### 2.1. Pliki JSONL (`TrainingLogger`, `new_hogwild_training_logger.py`)

| Plik | Metoda / wywołanie | Zawartość | Częstotliwość |
|---|---|---|---|
| `logs/metadata.json` | `write_metadata()` (L235-249), wywołane w `train_a3c_carla.py:538-544` | `start_time`, `args` (pełny config), `model`, `n_params`, `n_workers`, `model_params` | raz, na starcie |
| `logs/worker_<id>/steps.jsonl` | `log_step()` (L85-111), wywołane w `a3c.py:933-950` | `global_t`, `local_t`, `global_episode`, `step_in_ep`, `action`, `value`, `entropy`, `reward`, `done`, `speed_kmh`, `route_dist`, `goal_dist`, `maneuver`, **`reward_components`** (przekazywane jako `extra`) | co krok, tylko z `--log-steps` (domyślnie wyłączone) |
| `logs/worker_<id>/episodes.jsonl` | `log_episode()` (L113-141), wywołane z `A3CWorker._log_episode()` (`a3c.py:702-733`) | `global_episode`, `global_t`, `total_reward`, `steps`, **`mean_reward`** (=`total_reward/steps`), `reached_goal`, `duration_s`, `max_speed_kmh`, `min_route_dist`, `goal_dist`, `collisions`, `action_counts`, `port`, `local_mean_reward`, `global_mean_reward`, `is_new_best`, `reward_components` (pełny słownik sum) | co epizod |
| `logs/worker_<id>/updates.jsonl` | `log_update()` (L143-191), wywołane w `a3c.py:666-682` | `update`, `global_t`, `trajectory_length`, `is_terminal`, `pi_loss`, `v_loss`, `total_loss`, `gradient_norm`, `lr`, **`entropy_coef`**, `advantages_mean/std`, `val_mean/std`, `ent_mean`, `rew_mean`, `rew_sum`, **`reward_<component>_sum`/`_mean`** (z `summarize_reward_components()`, `a3c.py:241-250`), opcjonalnie surowe tablice `advantages/values/rewards/entropies` z `--log-update-arrays` | co optymalizator (`rollout_length` kroków lub koniec epizodu) |
| `logs/worker_<id>/timing.jsonl` | `log_timing()` (L193-201), wywołane w `a3c.py:986-988` (co `diag_log_interval` update'ów) i `a3c.py:1024-1026` (co `diag_log_wall_s` sekund) | `window_updates`, `ops`: `{sync, env_reset, forward, env_step, loss_compute, backward, optim_update, checkpoint_save}` → `{avg_ms, count, total_s}` | okresowo |
| `logs/events.jsonl` | `log_event()` (L203-205) **oraz** niezależna funkcja `_append_event()` w `new_hogwild_run_a3c.py:152-161` (patrz [4.7](#4-błędy-i-nieścisłości-w-logging-analisysmd)) | `training_start`/`training_end` (`train_a3c_carla.py:547-551, 625-630`), `checkpoint_save`/`model_save`/`model_load` (`training_logger.py:207-215`), `crash_recovery` (`a3c.py:1044-1046`), `nan_gradient` (`a3c.py:630-633`), `camera_timeout` (`a3c.py:1053-1056`), `worker_crash` (`a3c.py:1065-1068`), `worker_start/worker_restart/worker_give_up/rollback` (`run_a3c.py:183-246`) | zdarzeniowo |

### 2.2. Monitoring systemowy (`new_hogwild_system_monitor.py`)

| Plik | Klasa | Zawartość | Uruchamiane w |
|---|---|---|---|
| `logs/system.jsonl` | `RunMonitor` (L205-233) | `system` (CPU %, pamięć, swap, liczba procesów), `carla` (CPU/RSS/liczba żywych serwerów CARLA), `gpus` (util %, pamięć) jeśli `pynvml` dostępny | `main()`, `train_a3c_carla.py:554-560`, co `--monitor-interval` (domyślnie 10 s) |
| `logs/worker_<id>/system.jsonl` | `WorkerMonitor` (L236-254) | `cpu_percent`, `rss_gb`, `vms_gb`, `num_threads`, `ctx_voluntary`, `ctx_involuntary` | `A3CWorker.run()`, `a3c.py:824-827` |

Te pliki **nigdy** nie trafiają do W&B.

### 2.3. Checkpointy i stan wznowienia

| Plik | Tworzony przez | Zawartość |
|---|---|---|
| `checkpoint.pth` + `checkpoint_step.txt` | `GlobalNetwork.save_boundary_checkpoint()`, `a3c.py:365-429`, wywoływane przez `A3CWorker._save_checkpoint()` co `--save-frequency` kroków | pełny `state_dict` modelu/optymalizatora + liczniki globalne |
| `checkpoints/worker_<id>/checkpoint.pth` | tamże, tylko z `--save-worker-checkpoints` | to samo, ale per worker |
| `best_checkpoint.pth` | `GlobalNetwork.save()`, `a3c.py:355-363`, wywoływane z `_log_episode()` przy nowym rekordzie (`a3c.py:739-744`) | jak wyżej |
| `resume_state.json` | **dwa różne miejsca**: częściowo w `save_boundary_checkpoint()` (`a3c.py:402-426`) i w pełni w `_write_resume_state()` (`train_a3c_carla.py:408-431`, wywoływane raz na koniec sesji) | `global_step`, `global_episode`, `total_updates`, czasy sesji, `training_args` |
| `args.txt` / `args_resume.txt` | `prepare_output_dir()`, `new_hogwild_prepare_output_dir.py:8-33` | pełny zrzut argumentów CLI — **duplikat pola `args` w `metadata.json`** |
| `wandb_run_id.txt` | `main()`, `train_a3c_carla.py:565-575` | identyfikator runu W&B (do wznowienia) |

### 2.4. Konsola / log joba SLURM

Wszystkie `print(..., flush=True)` w `new_hogwild_a3c.py`, `new_hogwild_run_a3c.py`, `new_hogwild_train_a3c_carla.py`, `new_hogwild_timing_utils.py.log_and_reset()` oraz `ColoredPrint` z `utils.py` (używane w `carla_env.py`) trafiają do pliku zdefiniowanego w `new_hogwild_train.slurm:11-12` (`--output`/`--error`, ten sam plik `new-hogwild-a3c-carla-log-%J.txt`). Przykładowe prefiksy: `[RUN]`, `[RESUME]`, `[SAVE]`, `[BEST]`, `[NaN]`, `[W{id}]`, `[RESTART]`, `[ROLLBACK...]`, `[SIGNAL]`, `[BENCHMARK]`, `[TIMING]`. To jest de facto **czwarte miejsce logowania**, pomijane przez `logging-analisys.md`, a niosące istotne informacje operacyjne (crashe, restarty, throughput) niedostępne nigdzie indziej w tak zwięzłej formie.

Uwaga poboczna: `utils.py` definiuje też `ColoredPrint.store()`, który miałby zapisywać do `logfile.log`, ale metoda ta nigdzie nie jest wywoływana — to martwy kod z poprzedniej implementacji.

---

## 3. Podsumowanie tabelaryczne (poprawione)

| Dane | Lokalnie | W&B |
|---|---|---|
| Hiperparametry | `args.txt`, `metadata.json` (duplikat) | `wandb.init(config=...)` |
| Per-krok (`action`, `value`, `entropy`, `reward`, `speed_kmh`...) | `steps.jsonl` (opcjonalnie) | **brak** |
| Epizod (`total_reward`, `steps`, `duration_s`, komponenty nagrody...) | `episodes.jsonl` | **tak** (podzbiór pól, inne nazewnictwo) |
| Aktualizacja gradientu (`pi_loss`, `v_loss`, `gradient_norm`, `lr`...) | `updates.jsonl` | **brak** |
| Timing faz pętli | `timing.jsonl` | **brak** |
| Zasoby systemowe / GPU / CARLA | `system.jsonl` (x2) | **brak** |
| Checkpointy | pliki `.pth` | **brak** |
| Zdarzenia (restart, rollback, NaN, crash) | `events.jsonl` | **brak** |
| Diagnostyka operacyjna (print) | log joba SLURM | **brak** |

---

## 4. Błędy i nieścisłości w `logging-analisys.md`

1. **Największy błąd:** tabele w sekcjach „Queue → W&B” i „Podsumowanie” sugerują, że `log_step`, `log_update`, `log_timing` wypychają dane do `log_queue` i trafiają do W&B. W rzeczywistości `TrainingLogger` w ogóle nie ma referencji do kolejki — jedynym miejscem tworzącym rekord dla W&B jest ręcznie pisany kod w `A3CWorker._log_episode()` (`a3c.py:746-776`), i to tylko dla danych epizodu.
2. Nie wspomniano o polu `mean_reward` (`total_reward/steps`) w `episodes.jsonl` (`training_logger.py:123`) — mylone/pomijane obok podobnie nazwanych `local_mean_reward`/`global_mean_reward`.
3. Nie wspomniano o polu `entropy_coef` dopisywanym do `updates.jsonl` (`a3c.py:676`).
4. Nie wspomniano o agregatach komponentów nagrody (`reward_<nazwa>_sum`, `reward_<nazwa>_mean`) dopisywanych do `updates.jsonl` przez `summarize_reward_components()` (`a3c.py:241-250, 663-664, 681`).
5. Nie wspomniano, że `steps.jsonl` (gdy włączone) zawiera też `reward_components` (`a3c.py:948-949`).
6. Nie wspomniano, że zapis do `events.jsonl` odbywa się przez **dwie niezależne implementacje**: `TrainingLogger.log_event()` i osobną funkcję `_append_event()` w `new_hogwild_run_a3c.py:152-161` (ten sam plik, inny kod zapisujący).
7. `resume_state.json` opisano jako plik tworzony raz na koniec treningu — w rzeczywistości jest też częściowo nadpisywany przy każdym checkpointcie granicznym (`GlobalNetwork.save_boundary_checkpoint`, `a3c.py:402-426`), z innym (mniejszym) zestawem pól niż wersja z końca sesji.
8. Pominięto plik `args.txt`/`args_resume.txt` (`new_hogwild_prepare_output_dir.py`), który duplikuje `args` z `metadata.json`.
9. Pominięto log joba SLURM (stdout/stderr) jako osobną, czwartą kategorię logowania z istotną treścią diagnostyczną.
10. Sekcja „Checkpointing & Resume State” nie zaznacza, że pliki `checkpoint.pth`/`checkpoints/worker_*` mogą też pochodzić z `rollback_global_network()` (`run_a3c.py:57-132`), które odczytuje (nie zapisuje) checkpointy w procesie nadzorcy, niezależnie od workerów.

Poza powyższymi punktami reszta opisu w `logging-analisys.md` (nazwy plików, ogólna struktura katalogów, `RunMonitor`/`WorkerMonitor`) jest zgodna z kodem.

---

## 5. Propozycja refaktoru logowania

Zgodnie z wytycznymi: usuwamy tylko to, co na pewno jest zbędne, dodajemy tylko to, co ma wyraźne uzasadnienie i dużą wartość, a sposób logowania zmieniamy tylko tam, gdzie jest to mocno zalecane.

### 5.1. Do usunięcia (niska wartość / zbędne)

| Co | Gdzie | Uzasadnienie |
|---|---|---|
| Zawsze otwierane, ale nieużywane pliki `worker_-1/episodes.jsonl`, `worker_-1/updates.jsonl`, `worker_-1/timing.jsonl` | `TrainingLogger.__init__`, `new_hogwild_training_logger.py:63-65`, instancjonowane jako logger zdarzeń w `train_a3c_carla.py:546, 624` | Logger `worker_id=-1` służy wyłącznie do `log_event()`, a mimo to `__init__` bezwarunkowo tworzy 3 dodatkowe, wiecznie puste pliki JSONL w każdym runie. Otwieranie plików leniwie (przy pierwszym zapisie) usunie ten szum bez utraty żadnych danych. |
| Duplikat pliku z hiperparametrami — `args.txt`/`args_resume.txt` | `new_hogwild_prepare_output_dir.py` | Te same dane (pełny config) są już w `logs/metadata.json` razem z `model`/`n_params`. Utrzymywanie dwóch źródeł prawdy grozi rozjazdem przy przyszłych zmianach. Rekomendacja: zostawić `metadata.json` jako jedyne źródło, usunąć zapis `args.txt` (albo odwrotnie, jeśli istnieją zewnętrzne skrypty parsujące akurat `args.txt` — do potwierdzenia przed usunięciem). |

Świadomie **nie** proponujemy usuwania: pola `worker` w każdym rekordzie per-worker (przydatne przy łączeniu logów wielu workerów offline), surowych tablic z `--log-update-arrays` (i tak domyślnie wyłączone, więc nie generują kosztu w standardowych runach) ani żadnych pól z `episodes.jsonl`/`updates.jsonl` — każde z nich ma realne zastosowanie diagnostyczne przy debugowaniu treningu na CARLA.

### 5.2. Do dodania (wysoka wartość, jasne uzasadnienie)

| Co dodać | Gdzie | Uzasadnienie |
|---|---|---|
| Zagregowane metryki treningu (`pi_loss`, `v_loss`, `total_loss`, `gradient_norm`, `lr`, `entropy_coef`) wysyłane do W&B co N aktualizacji (np. co `diag_log_interval`) | `A3CWorker.compute_and_apply_gradients()` / `_log_episode()`, `a3c.py` | To największa luka w obecnym systemie: jedyne zdalne narzędzie monitorujące (W&B) nie pokazuje w ogóle krzywej strat ani normy gradientu. Bez tego, aby ocenić czy trening się zbiega, trzeba ręcznie ściągać `updates.jsonl` z klastra HPC. Dodanie tego jednym zapytaniem do kolejki (analogicznym do już istniejącego mechanizmu w `_log_episode`) daje ogromną wartość praktyczną przy niewielkim koszcie (throttlowane co N update'ów, nie co krok). |
| Globalna metryka `best_reward` / `global_mean_reward` (bez prefiksu `worker/<id>/`) | `_log_episode()`, `a3c.py:735-744` | Obecnie postęp widać tylko jako osobne serie per-worker w W&B, co utrudnia szybką ocenę ogólnego trendu treningu na jednym wykresie. Dane te już są liczone (`GlobalNetwork.update_stats`), brakuje tylko wysyłki bez prefiksu workera. |
| Okresowe podsumowanie zdrowia systemu (np. szczytowe RSS/GPU util z ostatniego okna) wysyłane do W&B raz na kilka minut | `RunMonitor`, `new_hogwild_system_monitor.py` + nowe połączenie z `log_queue` | Częsty scenariusz na klastrze SLURM to zabicie zadania przez OOM-killer bez czytelnego komunikatu. Obecnie fakt ten widać dopiero po ściągnięciu `system.jsonl` z HPC. Wysyłanie zagregowanego podsumowania (a nie surowych próbek co 10 s) do W&B pozwoli zauważyć rosnące zużycie pamięci zdalnie, zanim dojdzie do crasha. |
| Licznik odrzuconych rekordów kolejki W&B (`dropped_wandb_records`) zapisywany do `events.jsonl` przy zamknięciu workera | `A3CWorker._log_episode()`, `a3c.py:774` (obecnie `except Exception: pass` przy pełnej kolejce `maxsize=1000`) | Bez tego nie ma żadnego śladu, czy dane widoczne w W&B są kompletne, czy część epizodów „zgubiła się” przy przepełnionej kolejce — szczególnie istotne przy dużej liczbie workerów. |

Świadomie **nie** proponujemy dodawania per-step danych do W&B (zbyt duży wolumen przy wielogodzinnych runach na HPC, `steps.jsonl` lokalnie w zupełności wystarcza do szczegółowego debugowania).

### 5.3. Zmiany w sposobie logowania (tylko mocno zalecane)

| Zmiana | Uzasadnienie |
|---|---|
| Skonsolidować dwie niezależne implementacje zapisu `events.jsonl` (`TrainingLogger.log_event()` w `training_logger.py` i `_append_event()` w `new_hogwild_run_a3c.py:152-161`) w jedną, współdzieloną funkcję/klasę | Obecnie oba miejsca piszą do tego samego pliku, ale osobnym kodem (różna obsługa błędów, różne domyślne pola). To źródło przyszłych niespójności formatu — konsolidacja jest tania i eliminuje realne ryzyko. |
| Ujednolicić zapis `resume_state.json` w jednej funkcji pomocniczej używanej zarówno przy checkpoincie granicznym (`GlobalNetwork.save_boundary_checkpoint`), jak i na końcu sesji (`_write_resume_state`) | Obecnie te dwa miejsca zapisują częściowo różne zestawy pól do tego samego pliku, co może prowadzić do utraty pól (np. `elapsed_training_s`) przy określonej kolejności zdarzeń (crash tuż po checkpoincie). Ujednolicenie eliminuje tę klasę błędów bez zmiany zawartości pliku. |
| Przenieść logikę budowania rekordu dla W&B z ręcznego kodu w `A3CWorker._log_episode()` do wspólnej metody w `TrainingLogger` (opcjonalny parametr `log_queue` przekazywany do konstruktora) | Obecnie `TrainingLogger.log_episode()` i logika wysyłki do W&B w `_log_episode()` to dwa niezależne, ręcznie synchronizowane słowniki z częściowo innymi nazwami pól (`goal_dist` vs `distance_from_target`, `local_mean_reward` identyczne). Scalenie w jednym miejscu eliminuje ryzyko, że przyszła zmiana pola w jednym miejscu „zapomni” o drugim, i ułatwia ewentualne przyszłe dodanie wysyłki `log_update`/`log_timing` do W&B (patrz 5.2) bez duplikowania wzorca. |

Wszystkie pozostałe elementy istniejącego potoku logowania (struktura katalogów per-worker, format JSONL, częstotliwości zapisu domyślne, monitoring systemowy, checkpointy) rekomendujemy pozostawić bez zmian — działają poprawnie i dostarczają dane o wyraźnej wartości diagnostycznej.
