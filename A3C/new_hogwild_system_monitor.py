"""Background system-resource loggers. Ported from async-rl.

RunMonitor    - one per run; system-wide CPU/memory/swap + CARLA process stats.
                Writes <run_output_dir>/logs/system.jsonl.
WorkerMonitor - one per worker; per-process stats.
                Writes <run_output_dir>/logs/worker_<id>/system.jsonl.

Optional GPU stats: if pynvml is available (bundled with recent NVIDIA
drivers), we also sample GPU utilization and memory per device.
"""

import json
import os
import threading
from datetime import datetime

import psutil

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
    _NVML_DEVICE_COUNT = pynvml.nvmlDeviceGetCount()
except Exception:
    _NVML_AVAILABLE = False
    _NVML_DEVICE_COUNT = 0


def _ts():
    return datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]


def _find_carla_pids():
    pids = []
    for p in psutil.process_iter(['name', 'cmdline']):
        try:
            name = p.info.get('name', '') or ''
            cmdline = p.info.get('cmdline', []) or []
            cmd_str = ' '.join(cmdline)
            if 'CarlaUE4' in name or 'CarlaUE4' in cmd_str:
                pids.append(p.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return pids


def _sample_gpus():
    if not _NVML_AVAILABLE:
        return []
    out = []
    for i in range(_NVML_DEVICE_COUNT):
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            out.append({
                'index': i,
                'util_percent': int(util.gpu),
                'mem_used_gb': round(mem.used / 1e9, 3),
                'mem_total_gb': round(mem.total / 1e9, 3),
            })
        except Exception:
            pass
    return out


def _sample_system():
    log_record = {}
    cpu_pct = psutil.cpu_percent(interval=None, percpu=True)
    log_record['cpu_percent_mean'] = round(sum(cpu_pct) / len(cpu_pct), 1)
    log_record['cpu_percent_max'] = round(max(cpu_pct), 1)
    log_record['cpu_count_busy'] = sum(1 for c in cpu_pct if c > 50.0)
    log_record['cpu_count_total'] = len(cpu_pct)

    freq = psutil.cpu_freq()
    if freq is not None:
        log_record['cpu_freq_mhz'] = round(freq.current, 1)

    vmem = psutil.virtual_memory()
    log_record['mem_used_gb'] = round(vmem.used / 1e9, 2)
    log_record['mem_available_gb'] = round(vmem.available / 1e9, 2)
    log_record['mem_percent'] = vmem.percent

    swap = psutil.swap_memory()
    log_record['swap_used_gb'] = round(swap.used / 1e9, 2)

    total_procs = 0
    total_rss = 0.0
    for p in psutil.process_iter(['memory_info']):
        try:
            mi = p.info.get('memory_info')
            if mi:
                total_rss += mi.rss
            total_procs += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    log_record['total_procs'] = total_procs
    log_record['total_rss_gb'] = round(total_rss / 1e9, 2)
    return log_record


def _sample_carla(carla_pids):
    servers = []
    total_cpu = 0.0
    total_rss = 0.0
    alive = 0
    for pid in carla_pids:
        try:
            p = psutil.Process(pid)
            cpu = p.cpu_percent(interval=None)
            rss = p.memory_info().rss
            total_cpu += cpu
            total_rss += rss
            alive += 1
            servers.append({
                'pid': pid,
                'cpu_percent': round(cpu, 1),
                'rss_gb': round(rss / 1e9, 3),
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return {
        'total_cpu_percent': round(total_cpu, 1),
        'total_rss_gb': round(total_rss / 1e9, 3),
        'alive': alive,
        'servers': servers,
    }


def _sample_worker(pid):
    log_record = {}
    try:
        proc = psutil.Process(pid)
        with proc.oneshot():
            log_record['cpu_percent'] = proc.cpu_percent(interval=None)
            mem_info = proc.memory_info()
            log_record['rss_gb'] = round(mem_info.rss / 1e9, 3)
            log_record['vms_gb'] = round(mem_info.vms / 1e9, 3)
            log_record['num_threads'] = proc.num_threads()
            ctx = proc.num_ctx_switches()
            log_record['ctx_voluntary'] = ctx.voluntary
            log_record['ctx_involuntary'] = ctx.involuntary
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return log_record


class _BaseMonitor:
    def __init__(self, log_path, interval):
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread = None

        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._file = open(log_path, 'a', buffering=1)

        psutil.cpu_percent(interval=None, percpu=True)

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 2)
        try:
            self._file.flush()
            self._file.close()
        except Exception:
            pass

    def _sample(self):
        raise NotImplementedError

    def _loop(self):
        sample_count = 0
        self._on_loop_start()

        while not self._stop_event.is_set():
            try:
                log_record = self._sample()
                log_record['ts'] = _ts()
                self._file.write(json.dumps(log_record) + '\n')
            except Exception as e:
                try:
                    self._file.write(json.dumps({
                        'ts': _ts(), 'error': str(e),
                    }) + '\n')
                except Exception:
                    pass

            sample_count += 1
            if self._stop_event.wait(timeout=self.interval):
                break
            self._on_tick(sample_count)

    def _on_loop_start(self):
        pass

    def _on_tick(self, sample_count):
        pass


class RunMonitor(_BaseMonitor):
    def __init__(self, run_output_dir, interval=10.0, track_carla=True,
                 track_gpu=True):
        log_path = os.path.join(run_output_dir, 'logs', 'system.jsonl')
        super().__init__(log_path, interval)
        self.track_carla = track_carla
        self.track_gpu = track_gpu
        self._carla_pids = []

    def _on_loop_start(self):
        if self.track_carla:
            self._carla_pids = _find_carla_pids()
            for pid in self._carla_pids:
                try:
                    psutil.Process(pid).cpu_percent(interval=None)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

    def _on_tick(self, sample_count):
        if self.track_carla and sample_count % 10 == 0:
            self._carla_pids = _find_carla_pids()

    def _sample(self):
        log_record = {'system': _sample_system()}
        if self.track_carla:
            log_record['carla'] = _sample_carla(self._carla_pids)
        if self.track_gpu and _NVML_AVAILABLE:
            log_record['gpus'] = _sample_gpus()
        return log_record


class WorkerMonitor(_BaseMonitor):
    def __init__(self, run_output_dir, worker_id, interval=10.0):
        log_path = os.path.join(run_output_dir, 'logs',
                                'worker_{}'.format(worker_id),
                                'system.jsonl')
        super().__init__(log_path, interval)
        self.worker_id = worker_id
        self._worker_pid = os.getpid()

    def _on_loop_start(self):
        try:
            psutil.Process(self._worker_pid).cpu_percent(interval=None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    def _sample(self):
        log_record = _sample_worker(self._worker_pid)
        log_record['worker'] = self.worker_id
        return log_record
