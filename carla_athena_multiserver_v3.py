"""
Run multiple CARLA servers, each inside its own Apptainer container.
Enhanced with health checking and real-time dashboard.
Uses lsof to reliably check port status.
"""
#rm -f server.log && nohup $SCRATCH/venvs/carla_env/bin/python carla_athena_multiserver_v3.py > multiserver_v3.log 2>&1 &
#
import multiprocessing as mp
import os
import sys
import signal
import subprocess
from time import sleep, time
from typing import List, Dict, Optional, Set
from datetime import datetime
import threading
import logging
import psutil
import torch
import json
from pathlib import Path
import re
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num-servers", type=int, default=1)
    parser.add_argument("--servers-per-gpu", type=int, default=2)
    parser.add_argument("--server-gpu-start", type=int, default=0)
    parser.add_argument("--start-port", type=int, default=2000)
    parser.add_argument("--port-step", type=int, default=100)
    parser.add_argument("--enable-dashboard", action="store_true")
    parser.add_argument("--outdir", type=str, default='.')

    return parser.parse_args()


ARGS = parse_args()


# MAIN SETTINGS
NUM_SERVERS = ARGS.num_servers
SERVERS_PER_GPU = ARGS.servers_per_gpu
START_PORT = ARGS.start_port
PORT_STEP = ARGS.port_step
ENABLE_DASHBOARD = ARGS.enable_dashboard


# TO BE CHANGED FOR EACH SYSTEM
CONTAINER_IMAGE: str = (
    "/net/tscratch/people/plgkatsiaryna/carla-scratch/carla_0.9.15.sif"
)
CARLA_BINARY: List[str] = [
    "/home/carla/CarlaUE4.sh",
    "-RenderOffScreen",
    "-nosound",
    "--carla-server",
]
DELAY: int = 0.1  # seconds
BASE_CMD: List[str] = ["apptainer", "exec", "--nv", CONTAINER_IMAGE]

# Health check settings
HEALTH_CHECK_INTERVAL: int = 30  # seconds
LSOF_CHECK_TIMEOUT: int = 5      # seconds

# Map instance index → GPU ID (1–4)
NUM_GPUS = torch.cuda.device_count()
GPU_MAP = {i: i // SERVERS_PER_GPU + 1 for i in range(NUM_GPUS * SERVERS_PER_GPU)}

# Global variables for logging and status
LOG_DIR = None
SERVER_PROCESSES: Dict[int, subprocess.Popen] = {}
SERVER_STATUS: Dict[int, Dict] = {}  # Shared status for dashboard
STOP_MONITORING = threading.Event()
STATUS_LOCK = threading.Lock()


def setup_logging() -> str:
    """Create log directory structure and return path."""
    base_dir = ARGS.outdir + "/server_logs"
    os.makedirs(base_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(base_dir, f"server_run_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup main server logger
    server_logger = logging.getLogger('servers')
    server_logger.setLevel(logging.INFO)
    server_handler = logging.FileHandler(os.path.join(log_dir, 'servers.log'))
    server_handler.setFormatter(
        logging.Formatter('%(asctime)s - [%(name)s] - %(levelname)s - %(message)s')
    )
    server_logger.addHandler(server_handler)
    
    # Setup health check logger
    health_logger = logging.getLogger('health')
    health_logger.setLevel(logging.INFO)
    health_handler = logging.FileHandler(os.path.join(log_dir, 'health.log'))
    health_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    health_logger.addHandler(health_handler)
    
    return log_dir


def initialize_server_status():
    """Initialize status dictionary for all servers."""
    with STATUS_LOCK:
        for idx in range(NUM_SERVERS):
            port = START_PORT + (idx * PORT_STEP)
            gpu_id = GPU_MAP.get(idx, 1)
            SERVER_STATUS[idx] = {
                'index': idx,
                'port': port,
                'gpu_id': gpu_id,
                'process_status': 'NOT_STARTED',
                'health_status': 'UNKNOWN',
                'last_health_check': None,
                'restart_count': 0,
                'uptime_start': None,
                'last_error': None,
                'pid': None,
                'memory_mb': 0,
                'cpu_percent': 0
            }


def update_server_status(idx: int, **kwargs):
    """Thread-safe update of server status."""
    with STATUS_LOCK:
        if idx in SERVER_STATUS:
            SERVER_STATUS[idx].update(kwargs)
            # Save status to file for dashboard
            save_status_to_file()


def get_system_status() -> Dict:
    """Get current system resource usage."""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory()
    
    # Get GPU info
    gpu_info = []
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 6:
                    gpu_info.append({
                        'index': int(parts[0]),
                        'name': parts[1],
                        'utilization': float(parts[2]),
                        'memory_used': float(parts[3]),
                        'memory_total': float(parts[4]),
                        'temperature': float(parts[5])
                    })
    except Exception:
        pass
    
    return {
        'cpu_percent': cpu_percent,
        'memory_percent': mem.percent,
        'memory_used_gb': mem.used / (1024**3),
        'memory_total_gb': mem.total / (1024**3),
        'gpus': gpu_info,
        'timestamp': datetime.now().isoformat()
    }


def save_status_to_file():
    """Save current status to JSON file for dashboard."""
    if LOG_DIR:
        try:
            status_file = os.path.join(LOG_DIR, 'dashboard_status.json')
            with open(status_file, 'w') as f:
                json.dump({
                    'servers': list(SERVER_STATUS.values()),
                    'timestamp': datetime.now().isoformat()
                }, f)
        except Exception:
            pass  # Ignore errors in status saving


def get_listening_ports_lsof() -> Set[int]:
    """
    Get all TCP ports that are currently in LISTEN state using lsof.
    Returns a set of port numbers.
    """
    try:
        result = subprocess.run(
            ['lsof', '-nP', '-iTCP', '-sTCP:LISTEN'],
            capture_output=True,
            text=True,
            timeout=LSOF_CHECK_TIMEOUT
        )
        
        listening_ports = set()
        
        # Parse lsof output
        # Format: COMMAND   PID   USER   FD   TYPE DEVICE SIZE/OFF NODE NAME
        # Example: CarlaUE4 12345 user   123u IPv4 123456 0t0 TCP *:2000 (LISTEN)
        for line in result.stdout.split('\n')[1:]:  # Skip header
            if line.strip():
                parts = line.split()
                if len(parts) >= 9:
                    # The NAME field is typically the last one and contains the port info
                    port_info = parts[8]
                    
                    # Extract port from formats like:
                    # *:2000 (LISTEN)
                    # localhost:2000 (LISTEN)
                    # 127.0.0.1:2000 (LISTEN)
                    if ':' in port_info:
                        port_str = port_info.split(':')[-1]
                        try:
                            port_num = int(port_str)
                            listening_ports.add(port_num)
                        except ValueError:
                            pass
        
        return listening_ports
    
    except subprocess.TimeoutExpired:
        logging.getLogger('health').warning("lsof command timed out")
        return set()
    except FileNotFoundError:
        logging.getLogger('health').error("lsof command not found - cannot check port status")
        return set()
    except Exception as e:
        logging.getLogger('health').error(f"Error running lsof: {e}")
        return set()


def check_port_listening(port: int, listening_ports: Optional[Set[int]] = None) -> bool:
    """
    Check if a specific port is in LISTEN state.
    If listening_ports is provided, use it; otherwise call lsof.
    """
    if listening_ports is None:
        listening_ports = get_listening_ports_lsof()
    
    return port in listening_ports


def get_process_memory_usage(pid: int) -> float:
    """Get memory usage of a process in MB."""
    try:
        process = psutil.Process(pid)
        return process.memory_info().rss / (1024 * 1024)  # Convert to MB
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0.0


def get_process_cpu_percent(pid: int) -> float:
    """Get CPU usage of a process."""
    try:
        process = psutil.Process(pid)
        return process.cpu_percent(interval=0.1)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0.0


def health_check_monitor(log_dir: str):
    """Continuously monitor health of all CARLA servers using lsof port checking."""
    logger = logging.getLogger('health')
    
    logger.info("Health check monitor started (using lsof)")
    
    while not STOP_MONITORING.is_set():
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Get all listening ports once per check cycle for efficiency
        listening_ports = get_listening_ports_lsof()
        logger.debug(f"Found {len(listening_ports)} listening ports: {sorted(listening_ports)}")
        
        for idx in range(NUM_SERVERS):
            port = START_PORT + (idx * PORT_STEP)
            
            # Check if process is running
            proc = SERVER_PROCESSES.get(idx)
            process_running = proc is not None and proc.poll() is None
            
            # Get process stats if running
            memory_mb = 0
            cpu_percent = 0
            if process_running and proc.pid:
                memory_mb = get_process_memory_usage(proc.pid)
                cpu_percent = get_process_cpu_percent(proc.pid)
            
            if process_running:
                # Check if port is in listening state using lsof
                is_listening = check_port_listening(port, listening_ports)
                
                if is_listening:
                    update_server_status(
                        idx,
                        health_status='HEALTHY',
                        process_status='RUNNING',
                        last_health_check=timestamp,
                        memory_mb=memory_mb,
                        cpu_percent=cpu_percent,
                        last_error=None
                    )
                    logger.info(f"Server {idx} (port {port}): HEALTHY - port listening (mem: {memory_mb:.1f}MB, cpu: {cpu_percent:.1f}%)")
                else:
                    update_server_status(
                        idx,
                        health_status='STARTING',
                        process_status='RUNNING',
                        last_health_check=timestamp,
                        last_error='Port not listening yet',
                        memory_mb=memory_mb,
                        cpu_percent=cpu_percent
                    )
                    logger.warning(f"Server {idx} (port {port}): STARTING - port not listening yet (mem: {memory_mb:.1f}MB, cpu: {cpu_percent:.1f}%)")
            else:
                update_server_status(
                    idx,
                    health_status='DOWN',
                    process_status='EXITED',
                    last_health_check=timestamp,
                    memory_mb=0,
                    cpu_percent=0
                )
                logger.warning(f"Server {idx} (port {port}): DOWN - process not running")
        
        # Wait for next check
        STOP_MONITORING.wait(HEALTH_CHECK_INTERVAL)
    
    logger.info("Health check monitor stopped")


def monitor_ports(log_dir: str) -> None:
    """Monitor ports every minute and log their status."""
    ports_log = os.path.join(log_dir, 'ports.log')
    expected_ports = [START_PORT + (i * PORT_STEP) for i in range(NUM_SERVERS)]
    
    with open(ports_log, 'w') as f:
        f.write("=== PORT MONITORING LOG (using lsof) ===\n")
        f.write(f"Expected ports: {expected_ports}\n")
        f.write("="*50 + "\n\n")
    
    while not STOP_MONITORING.is_set():
        try:
            result = subprocess.run(
                ['lsof', '-nP', '-iTCP', '-sTCP:LISTEN'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open(ports_log, 'a') as f:
                f.write(f"\n{'='*70}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"{'='*70}\n")
                
                listening_ports = {}
                for line in result.stdout.split('\n')[1:]:
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 9:
                            port_info = parts[8]
                            if ':' in port_info:
                                port = port_info.split(':')[-1]
                                try:
                                    port_num = int(port)
                                    listening_ports[port_num] = {
                                        'command': parts[0],
                                        'pid': parts[1],
                                        'user': parts[2]
                                    }
                                except ValueError:
                                    pass
                
                for idx in range(NUM_SERVERS):
                    port = expected_ports[idx]
                    gpu_id = GPU_MAP.get(idx, 1)
                    
                    process = SERVER_PROCESSES.get(idx)
                    process_status = "UNKNOWN"
                    if process:
                        if process.poll() is None:
                            process_status = "RUNNING"
                        else:
                            process_status = f"EXITED (code {process.returncode})"
                    else:
                        process_status = "NOT_STARTED"
                    
                    port_status = "NOT_LISTENING"
                    port_details = ""
                    if port in listening_ports:
                        port_status = "LISTENING"
                        info = listening_ports[port]
                        port_details = f" (PID: {info['pid']}, CMD: {info['command']})"
                    
                    f.write(f"  Server {idx} (Port {port}, GPU {gpu_id}):\n")
                    f.write(f"    Container Status: {process_status}\n")
                    f.write(f"    Port Status: {port_status}{port_details}\n")
                
                f.write("\n")
        
        except Exception as e:
            with open(ports_log, 'a') as f:
                f.write(f"ERROR in port monitoring: {e}\n")
        
        STOP_MONITORING.wait(60)


def monitor_system_resources(log_dir: str) -> None:
    """Monitor system resources and log them."""
    psutil_log = os.path.join(log_dir, 'psutil.log')
    
    with open(psutil_log, 'w') as f:
        f.write("=== SYSTEM RESOURCES MONITORING LOG ===\n\n")
    
    while not STOP_MONITORING.is_set():
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
            cpu_avg = psutil.cpu_percent(interval=1)
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            net = psutil.net_io_counters()
            
            with open(psutil_log, 'a') as f:
                f.write(f"\n{'='*70}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"{'='*70}\n")
                
                f.write(f"\nCPU Usage:\n")
                f.write(f"  Average: {cpu_avg:.1f}%\n")
                f.write(f"  Per Core: {', '.join([f'{c:.1f}%' for c in cpu_percent])}\n")
                
                f.write(f"\nMemory Usage:\n")
                f.write(f"  Total: {mem.total / (1024**3):.2f} GB\n")
                f.write(f"  Used: {mem.used / (1024**3):.2f} GB ({mem.percent}%)\n")
                f.write(f"  Available: {mem.available / (1024**3):.2f} GB\n")
                
                f.write(f"\nDisk Usage (/):\n")
                f.write(f"  Total: {disk.total / (1024**3):.2f} GB\n")
                f.write(f"  Used: {disk.used / (1024**3):.2f} GB ({disk.percent}%)\n")
                
                f.write(f"\nNetwork I/O:\n")
                f.write(f"  Bytes Sent: {net.bytes_sent / (1024**2):.2f} MB\n")
                f.write(f"  Bytes Recv: {net.bytes_recv / (1024**2):.2f} MB\n")
                
                f.write(f"\nActive Server Processes:\n")
                for idx, proc in SERVER_PROCESSES.items():
                    if proc and proc.poll() is None:
                        try:
                            p = psutil.Process(proc.pid)
                            mem_info = p.memory_info()
                            f.write(f"  Server {idx} (PID {proc.pid}):\n")
                            f.write(f"    Memory: {mem_info.rss / (1024**2):.2f} MB\n")
                            f.write(f"    CPU: {p.cpu_percent(interval=0.1):.1f}%\n")
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            f.write(f"  Server {idx}: Process info unavailable\n")
                
                f.write("\n")
        
        except Exception as e:
            with open(psutil_log, 'a') as f:
                f.write(f"ERROR in resource monitoring: {e}\n")
        
        STOP_MONITORING.wait(60)


def monitor_gpu_usage(log_dir: str) -> None:
    """Monitor GPU usage using nvidia-smi."""
    gpu_log = os.path.join(log_dir, 'gpu.log')
    
    with open(gpu_log, 'w') as f:
        f.write("=== GPU MONITORING LOG ===\n\n")
    
    while not STOP_MONITORING.is_set():
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            with open(gpu_log, 'a') as f:
                f.write(f"\n{'='*70}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"{'='*70}\n")
                
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 6:
                            idx, name, util, mem_used, mem_total, temp = parts
                            f.write(f"\nGPU {idx} ({name}):\n")
                            f.write(f"  Utilization: {util}%\n")
                            f.write(f"  Memory: {mem_used} MB / {mem_total} MB\n")
                            f.write(f"  Temperature: {temp}°C\n")
                
                f.write("\n")
        
        except Exception as e:
            with open(gpu_log, 'a') as f:
                f.write(f"ERROR in GPU monitoring: {e}\n")
        
        STOP_MONITORING.wait(60)


def supervise(idx: int) -> None:
    """Launch & babysit one CARLA instance identified by `idx`."""
    logger = logging.getLogger(f'servers.instance_{idx}')
    
    port = START_PORT + (idx * PORT_STEP)
    gpu_id = GPU_MAP.get(idx, 1)
    cmd = BASE_CMD + CARLA_BINARY + [
        f"-carla-rpc-port={port}",
        f"-graphicsadapter={gpu_id}",
    ]
    
    logger.info(f"Supervisor started for instance {idx} (port {port}, GPU {gpu_id})")
    
    restart_count = 0
    while True:
        try:
            logger.info(f"Launching server (restart #{restart_count}): {' '.join(cmd)}")
            
            update_server_status(
                idx,
                process_status='STARTING',
                restart_count=restart_count,
                uptime_start=time()
            )
            
            proc = subprocess.Popen(
                cmd,
                preexec_fn=os.setsid,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            SERVER_PROCESSES[idx] = proc

            update_server_status(
                idx,
                process_status='RUNNING',
                pid=proc.pid
            )

            stdout, stderr = proc.communicate()

            if stdout:
                logger.info(f"[STDOUT]\n{stdout}")

            if stderr:
                logger.error(f"[STDERR]\n{stderr}")
            
            proc.wait()
            
            update_server_status(
                idx,
                process_status='EXITED',
                last_error=f"Exit code: {proc.returncode}"
            )
            
            logger.warning(
                f"Server exited with code {proc.returncode}. "
                f"Restarting in {DELAY}s... (restart #{restart_count + 1})"
            )
            restart_count += 1
            sleep(DELAY)
            
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received - terminating server")
            update_server_status(idx, process_status='STOPPED')
            _kill_process_tree(proc)
            break
        except Exception as exc:
            logger.error(f"Error occurred: {exc}. Restarting in {DELAY}s...")
            update_server_status(
                idx,
                process_status='ERROR',
                last_error=str(exc)
            )
            restart_count += 1
            sleep(DELAY)
    
    if idx in SERVER_PROCESSES:
        del SERVER_PROCESSES[idx]


def _kill_process_tree(proc: subprocess.Popen) -> None:
    """SIGTERM the entire process group started via `setsid`."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        pass


def start_dashboard(log_dir: str) -> subprocess.Popen:
    """Start the Flask dashboard in a separate process."""
    dashboard_script = Path(__file__).parent / "dashboard.py"
    
    if not dashboard_script.exists():
        print(f"Warning: Dashboard script not found at {dashboard_script}")
        return None
    
    # Pass log directory and configuration via environment
    env = os.environ.copy()
    env['CARLA_LOG_DIR'] = log_dir
    env['CARLA_NUM_SERVERS'] = str(NUM_SERVERS)
    env['CARLA_START_PORT'] = str(START_PORT)
    
    proc = subprocess.Popen(
        ['python', str(dashboard_script)],
        env=env
    )
    
    print(f"Dashboard started on http://localhost:5000 (PID: {proc.pid})")
    return proc


def main() -> None:
    global LOG_DIR
    
    # Setup logging
    LOG_DIR = setup_logging()
    logger = logging.getLogger('servers.main')
    
    # Initialize status tracking
    initialize_server_status()
    
    logger.info(f"Starting {NUM_SERVERS} CARLA servers (first port {START_PORT})")
    logger.info(f"Using GPU map: {GPU_MAP}")
    logger.info(f"Log directory: {LOG_DIR}")
    logger.info(f"Health check method: lsof port monitoring")
    
    print(f"Starting {NUM_SERVERS} CARLA servers")
    print(f"Logs will be saved to: {LOG_DIR}")
    print(f"Health check method: lsof port monitoring")
    
    # Start dashboard
    dashboard_proc = start_dashboard(LOG_DIR) if ENABLE_DASHBOARD else None
    
    # Start monitoring threads
    port_monitor = threading.Thread(target=monitor_ports, args=(LOG_DIR,), daemon=True)
    resource_monitor = threading.Thread(target=monitor_system_resources, args=(LOG_DIR,), daemon=True)
    gpu_monitor = threading.Thread(target=monitor_gpu_usage, args=(LOG_DIR,), daemon=True)
    health_monitor = threading.Thread(target=health_check_monitor, args=(LOG_DIR,), daemon=True)
    
    port_monitor.start()
    resource_monitor.start()
    gpu_monitor.start()
    health_monitor.start()
    
    # Start server processes
    workers: List[mp.Process] = []
    for idx in range(NUM_SERVERS):
        p = mp.Process(target=supervise, args=(idx,), daemon=True)
        p.start()
        workers.append(p)
        logger.info(f"Started worker process for server {idx}")
    
    try:
        for p in workers:
            p.join()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt – shutting down all servers…")
        logger.info("Keyboard interrupt received - shutting down all servers")
        
        # Stop monitoring
        STOP_MONITORING.set()
        
        # Terminate workers
        for p in workers:
            p.terminate()
        for p in workers:
            p.join()
        
        # Stop dashboard
        if dashboard_proc:
            dashboard_proc.terminate()
            try:
                dashboard_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                dashboard_proc.kill()
        
        logger.info("All servers shut down successfully")
    
    print(f"Logs saved to: {LOG_DIR}")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()