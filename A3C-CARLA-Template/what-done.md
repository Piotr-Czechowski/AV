# Handoff: A3C-CARLA-Template (sesja przebudowy na uniwersalny szablon)

Dokument dla następnego agenta. Język użytkownika: polski. Imię: Bartosz. Nie commituj bez prośby. Nie edytuj pliku planu Cursor z tej sesji.

Źródło prawdy kodu: katalog `A3C-CARLA-Template/`. Referencja algorytmu: działający `A3C/` (`new_hogwild_*.py`). Leftover docs w starym template i `impr.md` / `A3C_CODE_WALKTHROUGH.md` / `A3C_LOGGING_REFACTOR_REVIEW.md` **nie** są źródłem prawdy i **nie** wchodzą do template.

Poprzedni chat: [template A3C CARLA](f45a5127-d8c1-4737-8ffb-0c089f2c0cd9).

---

## Cel tej sesji

Z `A3C-CARLA-Template` zrobić **uniwersalny plug-and-play Hogwild A3C** pod CARLA 0.9.15, skopiowany z działającego `A3C/`, bez zmiany algorytmu.

Świadomie **nie** wdrażać z `impr.md`: GAE, V-trace, PPO, IMPALA, nowe sieci, zmiana flow A3C.

Użytkownik sam wybiera jak stawia CARLĘ. HPC/Apptainer to jedna opcja, nie jedyna.

---

## Twarde decyzje (nie odwracać bez pytania)

1. **Dwie warstwy launch.** `train_a3c.py` **nigdy** nie startuje CARLI. Łączy się do już słuchających portów RPC. `carla_multiserver_launcher.py` jest opcjonalny (supervise + restart).
2. **Trzy runtime launchera:** `native` | `docker` | `apptainer`. **Nie ma** `--runtime external`. Serwery już stoją → launcher się nie odpala.
3. **Docker:** `--network host` + `--entrypoint BINARY` + `--gpus device=<id>`. Host network trzyma RPC i streaming (RPC+1) na hoście. Recipe na Linux; Docker Desktop na macOS nie dzieli sieci tak samo.
4. **GPU / `-graphicsadapter`:**
   - apptainer: 1-based (`cuda_index + 1`) — historyczny quirk HPC,
   - docker: adapter `0` (w kontenerze jedna GPU); host GPU z `CUDA_VISIBLE_DEVICES` przez `docker_gpu_device()`,
   - native: 0-based `cuda_index`.
5. **Default launchera:** `--runtime` = `CARLA_RUNTIME` z env albo `native`. Slurm i tak ustawia `CARLA_RUNTIME=apptainer`.
6. **Config:** `.env` (maszyna/sekrety) → `settings.py` (env/experiment defaults) → `rl_configuration.py` (akcje + legacy reward) → CLI `train_a3c.py` (shape runu / algorytm). CLI wygrywa. `load_dotenv` **nie** nadpisuje zmiennych już w środowisku.
7. **Model:** `models/shared_actor_critic.py` (`SharedActorCritic` 1:1 z A3C). Katalog `nets/` usunięty. **Brak** `models/__init__.py` (import torch przy `import models` psuł smoke bez torch).
8. **Blueprint env:** Town03, scenario 14, 10 akcji dyskretnych, kamera semantic 250×250. Dual cap: wrapper `--episode-max-decisions` default 100 vs `settings.STEP_COUNTER` = 200 w `CarlaEnv`.
9. **Punkty scenariuszy:** te same co w A3C. Warianty zakomentowane **wewnątrz** list (`MAP_POINTS_SC14` / `SC15`), nie nad listą.
10. **`utils.py` zostaje** — `ColoredPrint` używany w `carla_env.py`.
11. **Python:** 3.10–3.11. Deps: `carla==0.9.15`, `numpy`, `torch`, `opencv-python`, `networkx`. Extra: `wandb`, `resources` (`psutil`). Bez freeze CUDA i bez jupyter/pillow w `pyproject.toml`.
12. **Testy nie są deliverablem.** Smoke na macOS (bez CARLI) OK. Finalna weryfikacja = żywy `sbatch` na HPC — **nie wykonana z tej maszyny**.
13. **Docs dopiero po implementacji.** Zrobione: `README.md`, `docs/configuration.md`, `docs/launch.md`. `docs/running.md` usunięty. `docs/logging.md` został (istniejący, nie był celem przebudowy).

---

## Architektura launch (stan obecny)

```
[opcjonalnie] carla_multiserver_launcher.py
    native | docker --network host | apptainer
    1 wątek supervise na serwer: start / crash-restart / hang-detect (lsof RPC)
        ↓ porty LISTEN (2000, 2100, …)
train_a3c.py  --carla-host  --start-port  --port-step  --num-workers
    a3c_core.A3CWorker  →  CarlaA3CWrapper(host, port)
                        →  CarlaEnv  →  carla.Client(host, port)
```

Worker `i` → port `start_port + i * port_step`.

### `host` / `--carla-host`

To **hostname TCP** do RPC, nie „start CARLA”.

Łańcuch: `.env` `CARLA_HOST` → `settings.CARLA_HOST` → `--carla-host` (`train_a3c.py`, default `settings.CARLA_HOST`) → `config.carla_host` → `a3c_core` (`getattr(..., 'localhost')`) → `CarlaA3CWrapper(host=...)` → `CarlaEnv` → `carla.Client(self.host, port)`.

`localhost` jest poprawne gdy serwer i trening są na tej samej maszynie: native, apptainer na węźle, docker `--network host`. Inny host tylko gdy CARLA słucha gdzie indziej.

---

## Co powstało / zmieniło się (pliki)

### Nowe

| Plik | Rola |
|------|------|
| `settings.py` | `load_dotenv()` + `CARLA_HOST` / `PORT` / `MAP_NAME` / `SCENARIO` / `STEP_COUNTER`… |
| `carla_multiserver_launcher.py` | Supervise N serwerów, 3 runtime |
| `models/shared_actor_critic.py` | `SharedActorCritic` przeniesiony 1:1 |
| `env.example` | Szablon `.env` |
| `examples/local/run_servers.sh` | Native launcher, foreground |
| `examples/docker/run_servers.sh` | Docker launcher, foreground |
| `examples/run_train.sh` | Wait na porty + `train_a3c.py` (wspólny dla local/docker) |
| `examples/hpc/train.slurm` | Jeden job: launcher + wait + trening + cleanup |
| `examples/hpc/requirements-gpu.txt` | `carla`, `numpy`, `opencv-python`, `networkx` (torch CUDA osobno) |
| `tests/test_template_smoke.py` | argv launchera, CUDA_VISIBLE_DEVICES, spójność akcji, skip modelu bez torch |
| `docs/configuration.md`, `docs/launch.md` | Warstwy config / jak odpalać |

`a3c_core.py`, `carla_wrapper.py`, `train_a3c.py`, `timing_utils.py`, `training_logger.py`, `run_a3c.py` to kod Hogwild przeniesiony/dostosowany (algorytm nietknięty). Wrapper i core dostały `host` / `map_name`.

### Zmodyfikowane w tej sesji (istotne)

- `carla_env.py` — host/map z konstruktora (fallback `settings.*`), zero import-time globals do klienta; punkty scenariuszy z komentarzami w listach; `utils.ColoredPrint`.
- `carla_wrapper.py` — `host` / `map_name` do `CarlaEnv`. Później: usunięte martwe `is_server_alive()`, `_episode_total_reward`, `_save_failures` (nigdzie nie czytane; recovery i tak przez `reconnect()` w workerze; sumę nagrody liczy worker).
- `a3c_core.py` — import `SharedActorCritic` z `models/`; przekazuje `carla_host` / `map_name`.
- `train_a3c.py` — `--carla-host`, `--map-name`, defaulty z `settings`.
- `rl_configuration.py` — akcje + legacy reward (rozdzielone od settings).
- `state_observer.py` — usunięty nieużywany PIL.
- `pyproject.toml` — odchudzone deps, Python `>=3.10,<3.12`.
- `.gitignore` (repo AV) — m.in. `runs/`.

### Usunięte z template (celowo)

- `run_settings.py`
- `carla_athena_multiserver_v3.py`
- `nets/a2c.py` i katalog `nets/`
- root `train.slurm` (Athena-hardcoded) — zastąpiony `examples/hpc/train.slurm`
- `docs/running.md`
- `env_dotenv.py` (loader wchłonięty do `settings.py`)
- `models/__init__.py`
- stare `new_hogwild_*` w template (źródło zostało w `A3C/`)

Poza template (root AV, nie ruszać jako część szablonu): `impr.md`, `A3C_CODE_WALKTHROUGH.md`, `A3C_LOGGING_REFACTOR_REVIEW.md`.

---

## Przykłady: HPC vs local/docker — **nie ten sam orkiestrator**

Logiczny porządek ten sam: start serwerów → porty LISTEN → A3C. Sklejanie inne.

### HPC — jeden skrypt (`examples/hpc/train.slurm`)

`sbatch` z katalogu template. Opcjonalny `.env`. Default runtime **apptainer**. Launcher w tle (`nohup`) → wait 60×10 s ≈ 10 min → `train_a3c.py` w tle + tail logu. EXIT / SIGTERM / SIGUSR1 (~90 s przed limitem): SIGTERM treningu, launchera, `nvidia-smi dmon`, `pkill CarlaUE4`. Koniec treningu gasi też CARLĘ.

Linia nazw slurm: `A3C/new_hogwild_train.slurm` → kopia `A3C-CARLA-Template/train.slurm` (Athena, skasowana) → teraz `examples/hpc/train.slurm` (`CHANGE_ME` partition/account, bez obowiązkowego `new_hogwild_train_paths.sh`).

`--no-carla` = serwery już stoją. `--server-gpu-start` **jest używane** przez nowy launcher (stary Athena launcher to ignorował). `LOG_RESOURCES=""` zainicjowane (stary `set -u` mógł wywalić skrypt).

### Local / Docker — dwa terminale

- Terminal A: `examples/local/run_servers.sh` albo `examples/docker/run_servers.sh` — `exec` launchera, zostaje na foreground. **Nie** czeka na porty, **nie** startuje A3C.
- Terminal B: `examples/run_train.sh` — wait 60×5 s ≈ 5 min (`lsof`), potem `exec train_a3c.py`. `--no-wait` pomija poll.

Koniec treningu **nie** gasi serwerów. Ctrl+C w A gasi launcher (docker `--rm`). Brak wspólnego trapu, brak `nvidia-smi dmon`, cieńsze CLI niż slurm. Default outdir `runs/demo` vs HPC `runs/a3c_<Nw>_<czas>_<jobid>`.

Local vs docker między sobą: prawie identyczne; różni się `--runtime` i wymagany env (`CARLA_PATH` vs `CARLA_CONTAINER_IMAGE`).

---

## Launcher — krok po kroku (`carla_multiserver_launcher.py`)

Nie trenuje. Tylko N procesów CARLA + watchdog.

1. Import `settings` ładuje `.env`; `main()` woła `load_dotenv()` jeszcze raz (no-op gdy już załadowane).
2. CLI: `--runtime`, `--num-servers`, porty, `--image`, `--binary`, `--carla-path`, `--servers-per-gpu`, `--server-gpu-start`, `--outdir`.
3. `resolve_binary`: native + `CARLA_PATH` składa ścieżkę do `CarlaUE4.sh`.
4. apptainer/docker bez image → exit. Native bez pliku binarki → exit.
5. Logi: `<outdir>/server_logs/servers.log` + stdout. Raw UE4: `carla_server_<i>.log`.
6. GPU: `CUDA_VISIBLE_DEVICES` albo `nvidia-smi -L`.
7. Wątek `supervise(i)`: port = `start_port + i * port_step`, GPU = `i // servers_per_gpu + server_gpu_start` (clamp).
8. `build_cmd` jak wyżej (adapter indexing!).
9. Watchdog: `Popen` w nowej sesji (osobna grupa). Proces padł → sleep 5 s → restart. Co 30 s `lsof` RPC; 3 strajki (~90 s) bez LISTEN → SIGTERM grupy → restart.
10. SIGINT/SIGTERM: `STOP`, wątki gaszą procesy.

Hang-check działa tylko gdy launcher naprawdę pilnuje procesów.

---

## Testy / weryfikacja

```bash
cd A3C-CARLA-Template
python -m unittest tests.test_template_smoke
```

Ostatni znany wynik: 8 testów, 1 skip bez torch na macOS. Uruchamiać z katalogu template (nie z `AV/`), inaczej importy spadają.

Po slimie wrappera (20.09.2026): `python -m py_compile carla_wrapper.py` OK; AST: brak `is_server_alive` / `_episode_total_reward` / `_save_failures`; smoke nadal 8 testów, 1 skip.

**Nie zrobione:** żywy `sbatch` na klastrze. macOS nie odpala CARLI.

---

## Follow-up po szablonie (komentarze + częściowe odchudzenie)

- Komentarze i docs w template ujednolicone (EN, krótkie). Docs opisują zachowanie (wybrana mapa/scenariusz), nie default Town03 jako niezmiennik. `carla_navigation/` nietknięte.
- Odchudzenie **tylko** `carla_wrapper.py`:
  - `is_server_alive()` — nigdy nie wołane; worker wykrywa timeout i woła `reconnect()`.
  - `_episode_total_reward` — worker sam sumuje reward z `step()` do loggera.
  - `_save_failures` — inkrement przy IO, nigdy nie logowane; JPEG dump nadal łyka wyjątki (`pass`).
- Reszta planu odchudzenia (`carla_env`, `StateObserver`, `reward_function`, `utils`, `training_logger`) **nie ruszana** — może się przydać.

---

## Świadomie niedokończone / następne tematy

Z `AV/do-zrob.txt` (root, nie template):

- ujednolicenie komentarzy — **zrobione**
- odchudzenie — **częściowo** (tylko martwe API wrappera); reszta plików celowo zostawiona
- finalne prod-grade zmiany
- przejrzenie `carla_env` i uniwersalności
- na koniec docs i README

Inne luki:

- Brak live HPC.
- Local/docker **nie** są jednym skryptem „jak slurm”; jeśli użytkownik chce jeden plik start→check→train→cleanup na desktopie, tego nie ma.
- `docs/logging.md` nie był celem przebudowy.
- Git: zmiany niecommitowane (użytkownik nie prosił).
- Caveman lite było włączone w czacie (`/caveman lite`); off tylko na „stop caveman” / „normal mode”.

---

## Pułapki przy dalszej edycji

- Nie wciągać `impr.md` do algorytmu.
- Nie startować CARLI z `train_a3c.py`.
- Nie dodawać `models/__init__.py` który importuje torch.
- Komentarze wariantów SC14/SC15 **zostają w listach**.
- Docker: nie wracać do podwójnego ENTRYPOINT (image + binary jako argv po tagu). Używać `--entrypoint`.
- Native adapter nie jest 1-based. Apptainer tak.
- `a3c_core.py` nie importuje `settings.py`; host/map idą przez config z `train_a3c.py`.
- Smoke: `cd A3C-CARLA-Template` przed unittest.
- `host='localhost'` w sygnaturze wrappera to fallback konstruktora; ścieżka treningu zawsze podaje `config.carla_host`.
