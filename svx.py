#!/usr/bin/env python3
# svx.py - Simple Service Manager

import os
import sys
import json
import subprocess
import signal
import time
import argparse
import logging
import psutil
from typing import Dict, Any, Optional, List
import platform
from datetime import datetime
import shutil

# --- Configuration ---
DEFAULT_CONFIG_FILE = os.path.expanduser("~/.config/svx/svx_services.json")
DEFAULT_LOG_DIR = os.path.expanduser("~/.local/log/svx")
if platform.system() == "Windows":
    DEFAULT_CONFIG_FILE = os.path.expanduser(os.path.join(os.environ.get("APPDATA", "~"), "svx", "svx_services.json"))
    DEFAULT_LOG_DIR = os.path.expanduser(os.path.join(os.environ.get("APPDATA", "~"), "svx", "logs"))

# Ensure log directory exists before setting up logging
os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)

# --- Setup logging ---
logging.basicConfig(
    level=logging.DEBUG,  # Use DEBUG level for more verbose output
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(DEFAULT_LOG_DIR, "svx_manager.log"), 'a')
    ]
)
logger = logging.getLogger('svx')

# --- Signal Handling ---
class SignalHandler:
    """Handles graceful shutdown on signals."""
    def __init__(self):
        self.shutdown = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
        if platform.system() != "Windows":
            signal.signal(signal.SIGHUP, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.shutdown = True

class ServiceManager:
    def __init__(self, config_file: str = DEFAULT_CONFIG_FILE, log_dir: str = DEFAULT_LOG_DIR):
        self.config_file = config_file
        self.log_dir = log_dir
        self.signal_handler = SignalHandler()
        self.services: Dict[str, Any] = {}  # Initialize services here
        self._ensure_directories()
        self.load_services() # Load services immediately


    def _ensure_directories(self):
        """Ensures config and log directories exist."""
        config_dir = os.path.dirname(self.config_file)
        if config_dir:
            os.makedirs(config_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        logger.debug(f"Config directory: {config_dir}")
        logger.debug(f"Log directory: {self.log_dir}")

    def load_services(self) -> None:
        """Loads services from the configuration file."""
        try:
            if not os.path.exists(self.config_file):
                logger.info(f"Config file {self.config_file} not found, creating it.")
                self._write_config({})  # Create empty config
            else:
                with open(self.config_file, "r", encoding='utf-8') as f:
                    self.services = json.load(f)
                logger.debug(f"Loaded services: {self.services}")
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading services from {self.config_file}: {e}")
            logger.warning("Using empty services dictionary.")
            self.services = {} # Ensure self.services is always a dict
        except Exception as e:  # Catch any other potential errors
            logger.exception(f"Unexpected error loading config: {e}")
            self.services = {}


    def _write_config(self, data: Dict) -> None:
        """Writes the given data to the config file, with robust error handling."""
        temp_file = self.config_file + ".tmp"  # Use a temporary file
        try:
            with open(temp_file, "w", encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            # Ensure data is written to disk before renaming
            f.flush()
            os.fsync(f.fileno())  # Force write to disk

            # Atomically replace the old config file with the new one
            shutil.move(temp_file, self.config_file)
            logger.debug(f"Successfully wrote to config file: {self.config_file}")

        except Exception as e:  # Catch *any* exception during write/rename
            logger.error(f"Error writing to config file {self.config_file}: {e}")
            # Clean up the temporary file if an error occurred
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except OSError as remove_err:
                    logger.error(f"Error removing temporary file {temp_file}: {remove_err}")
            raise  # Re-raise the exception to halt execution

    def save_services(self) -> bool:
        """Saves the current services to the configuration file."""
        try:
            self._write_config(self.services)
            return True
        except Exception:
            return False

    def add_service(self, name: str, command: List[str], working_dir: Optional[str] = None,
                   env_vars: Optional[Dict[str, str]] = None, auto_restart: bool = True,
                   restart_delay: int = 5) -> bool:
        """Add a new service."""
        if name in self.services:
            logger.error(f"Service '{name}' already exists.")
            return False

        self.services[name] = {
            "command": command,
            "pid": None,
            "working_dir": working_dir or os.getcwd(),
            "env_vars": env_vars or {},
            "auto_restart": auto_restart,
            "restart_delay": restart_delay,
            "status": "Stopped",
            "last_start": None,
            "instance_location": platform.node()
        }

        try:
            self.save_services()  # Use try-except here too
            logger.info(f"Service '{name}' added and saved.") # Log successful save
            return True
        except Exception as e:
            logger.error(f"Failed to save service '{name}': {e}")
            del self.services[name]  # Rollback: Remove the service
            return False

    def start_service(self, name: str) -> bool:
        """Start a service."""
        if name not in self.services:
            logger.error(f"Service '{name}' not found.")
            return False

        service = self.services[name]

        if service["pid"] and self._is_pid_running_and_valid(service["pid"], service["command"]):
            logger.info(f"Service '{name}' is already running with PID {service['pid']}.")
            return True

        log_file = os.path.join(self.log_dir, f"{name}.log")
        # Ensure log file's directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)


        try:
            env = os.environ.copy()
            env.update(service.get("env_vars", {}))

            with open(log_file, "a", encoding='utf-8') as log:
                log.write(f"\n--- Starting service {name} at {datetime.now().isoformat()} ---\n")
                working_dir = service.get("working_dir", os.getcwd())
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if platform.system() == "Windows" else 0
                preexec_fn = os.setsid if platform.system() != "Windows" else None

                process = subprocess.Popen(
                    service["command"],
                    shell=False,
                    stdout=log,
                    stderr=log,
                    cwd=working_dir,
                    env=env,
                    creationflags=creationflags,
                    preexec_fn=preexec_fn
                )

            service["pid"] = process.pid
            self._update_service_status(name, "Running")
            service["last_start"] = datetime.now().isoformat()
            service["instance_location"] = platform.node()
            if not self.save_services():
                logger.error(f"Failed to save service state after starting '{name}'.")
                return False

            logger.info(f"Service '{name}' started with PID {process.pid}.")
            return True

        except Exception as e:
            logger.error(f"Failed to start service '{name}': {e}")
            self._update_service_status(name, "Failed")
            self.save_services()
            return False

    def stop_service(self, name: str, timeout: int = 10) -> bool:
        """Stop a service with grace period before force kill."""
        if name not in self.services:
            logger.error(f"Service '{name}' not found.")
            return False

        service = self.services[name]

        if not service["pid"]:
            logger.info(f"Service '{name}' is not running.")
            return True

        try:
            pid = service["pid"]

            if not self._is_pid_running(pid):
                logger.info(f"Process for service '{name}' (PID {pid}) is not running.")
                self._update_service_status(name, "Stopped")
                self.save_services()
                return True

            def send_signal(pid, sig):
                try:
                    if platform.system() == "Windows":
                        if sig == signal.SIGTERM:
                            subprocess.call(['taskkill', '/PID', str(pid)])
                        elif sig == signal.CTRL_BREAK_EVENT:
                            os.kill(pid, signal.CTRL_BREAK_EVENT)
                        elif sig == signal.SIGKILL:
                            subprocess.call(['taskkill', '/F', '/PID', str(pid)])
                    else:
                        pgid = os.getpgid(pid)
                        os.killpg(pgid, sig)
                except ProcessLookupError:
                    pass
                except OSError as e:
                    logger.error(f"Error sending signal {sig} to {pid}: {e}")

            send_signal(pid, signal.CTRL_BREAK_EVENT if platform.system() == "Windows" else signal.SIGTERM)

            for _ in range(timeout):
                if not self._is_pid_running(pid):
                    break
                time.sleep(1)

            if self._is_pid_running(pid):
                logger.warning(f"Service '{name}' did not terminate gracefully, forcing kill...")
                send_signal(pid, signal.SIGKILL)

            self._update_service_status(name, "Stopped")
            if not self.save_services():
                logger.error(f"Failed to save service state after stopping '{name}'.")
                return False

            logger.info(f"Service '{name}' stopped.")
            return True

        except Exception as e:
            logger.error(f"Failed to stop service '{name}': {e}")
            self._update_service_status(name, "Unknown")
            self.save_services()
            return False

    def remove_service(self, name: str) -> bool:
        """Remove a service."""
        if name not in self.services:
            logger.error(f"Service '{name}' not found.")
            return False

        if self.services[name]["pid"]:
            self.stop_service(name)

        del self.services[name]
        if not self.save_services():
            logger.error(f"Failed to save services after removing '{name}'.")
            return False

        logger.info(f"Service '{name}' removed.")
        return True

    def _update_service_status(self, name: str, status: str):
        """Helper to consistently update service status and PID."""
        self.services[name]["status"] = status
        if status in ("Stopped", "Dead", "Failed"):
            self.services[name]["pid"] = None

    def list_services(self) -> List[Dict[str, Any]]:
        """List all services with their status."""
        result = []
        for name, details in self.services.items():
            self._update_and_check_status(name)
            result.append({
                "name": name,
                "status": details["status"],
                "pid": details["pid"],
                "command": " ".join(details["command"]),
                "instance": details.get("instance_location", "Unknown"),
                "last_start": details.get("last_start", "Never")
            })
        return result

    def get_service_status(self, name: str) -> Dict[str, Any]:
        """Get detailed status of a specific service."""
        if name not in self.services:
            logger.error(f"Service '{name}' not found.")
            return {}

        self._update_and_check_status(name)
        service = self.services[name]

        return {
            "name": name,
            "status": service["status"],
            "pid": service["pid"],
            "command": " ".join(service["command"]),
            "working_dir": service.get("working_dir", "N/A"),
            "env_vars": service.get("env_vars", {}),
            "auto_restart": service.get("auto_restart", True),
            "restart_delay": service.get("restart_delay", 5),
            "instance": service.get("instance_location", platform.node()),
            "last_start": service.get("last_start", "Never")
        }

    def _update_and_check_status(self, name: str):
        """Helper method to update the status of a service."""
        details = self.services[name]
        if details["pid"]:
            if self._is_pid_running_and_valid(details["pid"], details["command"]):
                details["status"] = "Running"
            else:
                self._update_service_status(name, "Dead")
                self.save_services()

    def monitor_services(self, interval: int = 5) -> None:
        """Monitor services and restart failed ones if needed."""
        logger.info("Starting service monitor...")
        while not self.signal_handler.shutdown:
            for name, details in self.services.items():
                if self.signal_handler.shutdown:
                    break

                if not details["pid"]:
                    continue

                if not self._is_pid_running_and_valid(details["pid"], details["command"]):
                    logger.warning(f"Service '{name}' (PID {details['pid']}) is not running.")
                    self._update_service_status(name, "Dead")
                    self.save_services()

                    if details.get("auto_restart", True):
                        restart_delay = details.get("restart_delay", 5)
                        logger.info(f"Waiting {restart_delay} seconds before restarting '{name}'...")
                        time.sleep(restart_delay)
                        if not self.signal_handler.shutdown:
                            logger.info(f"Restarting service '{name}'...")
                            self.start_service(name)

            time.sleep(interval)
        logger.info("Monitor stopped.")

    def _is_pid_running(self, pid: int) -> bool:
        """Check if a process with the given PID is running."""
        try:
            return psutil.pid_exists(pid)
        except Exception:
            return False

    def _is_pid_running_and_valid(self, pid: int, expected_command: List[str]) -> bool:
        """Check if a process is running AND matches the expected command."""
        try:
            process = psutil.Process(pid)
            if process.is_running():
                process_command = process.cmdline()
                # Use a more robust command comparison
                if self._compare_commands(expected_command, process_command):
                    return True
                logger.debug(f"PID {pid} running, but command mismatch. Expected: {expected_command}, Got: {process_command}")

                return False
            return False
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return False

    def _compare_commands(self, expected: List[str], actual: List[str]) -> bool:
        """Compares the expected and actual commands, handling paths and executable names."""

        # Normalize paths for comparison (handle Windows vs. Unix differences, etc.)
        expected_normalized = [os.path.normpath(part) for part in expected]
        actual_normalized = [os.path.normpath(part) for part in actual]

        # Check if the expected command is a subset of the actual command
        for i in range(len(actual_normalized) - len(expected_normalized) + 1):
            if actual_normalized[i:i + len(expected_normalized)] == expected_normalized:
                return True

        # Handle cases where the executable name might be without the full path in `expected`
        if len(expected_normalized) > 0:
            expected_base = os.path.basename(expected_normalized[0])
            for part in actual_normalized:
                if os.path.basename(part) == expected_base:
                    return True
        return False

    def tail_log(self, name: str, lines: int = 10) -> List[str]:
        """Get the last N lines of a service's log file."""
        if name not in self.services:
            logger.error(f"Service '{name}' not found.")
            return []

        log_file = os.path.join(self.log_dir, f"{name}.log")

        if not os.path.exists(log_file):
            return [f"Log file for service '{name}' does not exist."]

        try:
            with open(log_file, "r", encoding='utf-8') as f:
                return list(f.readlines())[-lines:]
        except Exception as e:
            logger.error(f"Error reading log file for service '{name}': {e}")
            return [f"Error reading log: {e}"]

    def clear_log(self, name: str) -> bool:
        """Clear the log file for a service."""
        if name not in self.services:
            logger.error(f"Service '{name}' not found.")
            return False

        log_file = os.path.join(self.log_dir, f"{name}.log")

        if not os.path.exists(log_file):
            return True

        try:
            with open(log_file, "w", encoding='utf-8') as f:
                f.write(f"--- Log cleared at {datetime.now().isoformat()} ---\n")
            logger.info(f"Log cleared for service '{name}'.")
            return True
        except Exception as e:
            logger.error(f"Error clearing log for service '{name}': {e}")
            return False

def print_table(data, headers):
    """Print data in a formatted table."""
    col_widths = [len(h) for h in headers]
    for row in data:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    header_row = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    separator = "-+-".join("-" * w for w in col_widths)
    print(header_row)
    print(separator)

    for row in data:
        row_str = " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
        print(row_str)

def _find_executable_in_path(executable_name: str) -> Optional[str]:
    """Finds the full path to an executable, handling Windows and Unix."""
    if platform.system() == "Windows":
        paths = os.environ["PATH"].split(os.pathsep)
        for path in paths:
            full_path = os.path.join(path, executable_name + ".exe")
            if os.path.isfile(full_path):
                return full_path
        for path in paths:
            full_path = os.path.join(path, executable_name)
            if os.path.isfile(full_path):
                return full_path
    else:
        return shutil.which(executable_name)
    return None

def _install_svx():
    """Installs svx by creating a symlink (Unix) or batch file (Windows)."""
    script_path = os.path.abspath(__file__)

    if platform.system() == "Windows":
        target_dir = None
        for path_dir in os.environ["PATH"].split(os.pathsep):
            if os.path.isdir(path_dir) and os.access(path_dir, os.W_OK):
                target_dir = path_dir
                break

        if target_dir is None:
            print("Error: Could not find a writable directory in your PATH.")
            print("Please add a directory to your PATH and try again.")
            return

        batch_file_path = os.path.join(target_dir, "svx.bat")
        # Use sys.executable to get the exact Python interpreter running this script
        python_exe = sys.executable

        if not python_exe or not os.path.isfile(python_exe):
            print("Error: Could not determine Python executable path.")
            return

        with open(batch_file_path, "w", encoding='utf-8') as f:
            f.write(f"@\"{python_exe}\" \"{script_path}\" %*\n")
        print(f"svx installed to: {batch_file_path}")

    else:
        home_dir = os.path.expanduser("~")
        local_bin_dir = os.path.join(home_dir, ".local", "bin")
        target_dir = local_bin_dir if (os.path.isdir(local_bin_dir) and local_bin_dir in os.environ["PATH"]) else "/usr/local/bin"
        if not os.access(target_dir, os.W_OK):
            print(f"Cannot install svx, no write access to {target_dir}")
            return
        symlink_path = os.path.join(target_dir, "svx")
        if os.path.exists(symlink_path):
            print(f"Error, {symlink_path} already exists. Remove it first")
            return
        try:
            os.symlink(script_path, symlink_path)
            os.chmod(script_path, 0o755)
            print(f"svx installed to: {symlink_path}")
        except OSError as e:
            print(f"Error creating symlink: {e}")
            print("You may need to run this with sudo.")

def _uninstall_svx():
    """Uninstalls svx by removing the symlink (Unix) or batch file (Windows)."""
    if platform.system() == "Windows":
        target_dir = None
        for path_dir in os.environ["PATH"].split(os.pathsep):
            if os.path.isdir(path_dir):
                target_dir = path_dir
                break

        if target_dir:
            batch_file_path = os.path.join(target_dir, "svx.bat")
            if os.path.exists(batch_file_path):
                try:
                    os.remove(batch_file_path)
                    print(f"svx uninstalled from: {batch_file_path}")
                except OSError as e:
                    print(f"Error removing batch file: {e}")
        else:
            print("Could not find svx batch file.")
    else:
        home_dir = os.path.expanduser("~")
        local_bin_dir = os.path.join(home_dir, ".local", "bin")
        target_dir = local_bin_dir if os.path.isdir(local_bin_dir) else "/usr/local/bin"
        symlink_path = os.path.join(target_dir, "svx")

        if os.path.exists(symlink_path):
            try:
                os.remove(symlink_path)
                print(f"svx uninstalled from: {symlink_path}")
            except OSError as e:
                print(f"Error removing symlink: {e}")
                print("You may need to run this with sudo.")
                
def parse_arguments():
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(description="SVX - Simple Service Manager")
        parser.add_argument("--install", action="store_true", help="Install svx")
        parser.add_argument("--uninstall", action="store_true", help="Uninstall svx")
        parser.add_argument("--config", help=f"Configuration file path (default: {DEFAULT_CONFIG_FILE})", default=DEFAULT_CONFIG_FILE)
        parser.add_argument("--log-dir", help=f"Log directory (default: {DEFAULT_LOG_DIR})", default=DEFAULT_LOG_DIR)

        # Parse arguments before adding subparsers to handle --install/--uninstall
        args, remaining_args = parser.parse_known_args()

        if not (args.install or args.uninstall):
            subparsers = parser.add_subparsers(dest="command", help="Command to execute", required=True)

            add_parser = subparsers.add_parser("add", help="Add a new service")
            add_parser.add_argument("name", help="Service name")
            add_parser.add_argument("command", help="Command to run", nargs="+")
            add_parser.add_argument("--dir", help="Working directory")
            add_parser.add_argument("--env", help="Environment variables (format: KEY=VALUE)", nargs="*")
            add_parser.add_argument("--no-restart", help="Disable auto-restart", action="store_true")
            add_parser.add_argument("--restart_delay", help="Seconds to wait before restart", type=int, default=5)

            start_parser = subparsers.add_parser("start", help="Start a service")
            start_parser.add_argument("name", help="Service name")

            stop_parser = subparsers.add_parser("stop", help="Stop a service")
            stop_parser.add_argument("name", help="Service name")
            stop_parser.add_argument("--timeout", help="Seconds to wait before force kill", type=int, default=10)

            restart_parser = subparsers.add_parser("restart", help="Restart a service")
            restart_parser.add_argument("name", help="Service name")

            remove_parser = subparsers.add_parser("remove", help="Remove a service")
            remove_parser.add_argument("name", help="Service name")

            subparsers.add_parser("list", help="List all services")

            status_parser = subparsers.add_parser("status", help="Show service status")
            status_parser.add_argument("name", help="Service name", nargs="?")

            monitor_parser = subparsers.add_parser("monitor", help="Monitor services and restart failed ones")
            monitor_parser.add_argument("--interval", help="Check interval in seconds", type=int, default=5)

            log_parser = subparsers.add_parser("log", help="Display service logs")
            log_parser.add_argument("name", help="Service name")
            log_parser.add_argument("--lines", help="Number of lines to display", type=int, default=10)
            log_parser.add_argument("--clear", help="Clear the log file", action="store_true")

            # Parse again with subparsers included
            args = parser.parse_args(remaining_args, namespace=args) # Parse remaining args

        return args

def main():
    """Main entry point."""
    args = parse_arguments()
    logger.debug(f"Parsed args: {vars(args)}")

    if args.install:
        _install_svx()
        return 0
    if args.uninstall:
        _uninstall_svx()
        return 0
    
    try:
        import psutil
    except ImportError:
        print("Error: psutil is required. Please install it: pip install psutil")
        return 1

    manager = ServiceManager(args.config, args.log_dir)

    try:
        if args.command == "add":
            env_vars = {}
            if args.env:
                for env_var in args.env:
                    if "=" in env_var:
                        key, value = env_var.split("=", 1)
                        env_vars[key] = value
                    else:
                        logger.warning(f"Invalid environment variable format: {env_var}")
            success = manager.add_service(
                args.name,
                args.command,
                working_dir=args.dir,
                env_vars=env_vars,
                auto_restart=not args.no_restart,
                restart_delay=args.restart_delay
            )
            if not success:
                print(f"Failed to add service '{args.name}'.")
                return 1
            print(f"Service '{args.name}' added successfully.")

        elif args.command == "start":
            if not manager.start_service(args.name):
                print(f"Failed to start service '{args.name}'.")
                return 1
            print(f"Service '{args.name}' started.")

        elif args.command == "stop":
            if not manager.stop_service(args.name, timeout=args.timeout):
                print(f"Failed to stop service '{args.name}'.")
                return 1
            print(f"Service '{args.name}' stopped.")

        elif args.command == "restart":
            if not manager.stop_service(args.name):
                print(f"Failed to stop service '{args.name}' for restart.")
                return 1
            time.sleep(2)
            if not manager.start_service(args.name):
                print(f"Failed to start service '{args.name}' for restart.")
                return 1
            print(f"Service '{args.name}' restarted.")

        elif args.command == "remove":
            if not manager.remove_service(args.name):
                print(f"Failed to remove service '{args.name}'.")
                return 1
            print(f"Service '{args.name}' removed.")

        elif args.command == "list":
            services = manager.list_services()
            if not services:
                print("No services found.")
            else:
                data = []
                for service in services:
                    pid = service["pid"] if service["pid"] else "N/A"
                    last_start = service.get("last_start", "Never")
                    if last_start and last_start != "Never":
                        try:
                            dt = datetime.fromisoformat(last_start)
                            last_start = dt.strftime("%Y-%m-%d %H:%M:%S")
                        except ValueError:
                            pass
                    data.append([service["name"], service["status"], pid, service["instance"], last_start])
                headers = ["NAME", "STATUS", "PID", "INSTANCE", "LAST START"]
                print_table(data, headers)

        elif args.command == "status":
            if args.name:
                status = manager.get_service_status(args.name)
                if not status:
                    print(f"Service '{args.name}' not found.")
                    return 1
                print(f"Service: {status['name']}")
                print(f"Status: {status['status']}")
                print(f"PID: {status['pid'] if status['pid'] else 'N/A'}")
                print(f"Instance: {status['instance']}")
                print(f"Command: {status['command']}")
                print(f"Working Directory: {status['working_dir']}")
                print(f"Auto-restart: {'Enabled' if status['auto_restart'] else 'Disabled'}")
                print(f"Restart Delay: {status['restart_delay']} seconds")
                if status.get("last_start") and status["last_start"] != "Never":
                    try:
                        dt = datetime.fromisoformat(status['last_start'])
                        print(f"Last Start: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
                    except ValueError:
                        print(f"Last Start: {status['last_start']}")
                else:
                    print("Last Start: Never")
                if status.get("env_vars"):
                    print("\nEnvironment Variables:")
                    for key, value in status["env_vars"].items():
                        print(f"  {key}={value}")
            else:
                services = manager.list_services()
                if not services:
                    print("No services found.")
                else:
                    data = []
                    for service in services:
                        pid = service["pid"] if service["pid"] else "N/A"
                        data.append([service["name"], service["status"], pid, service["instance"]])
                    headers = ["NAME", "STATUS", "PID", "INSTANCE"]
                    print_table(data, headers)

        elif args.command == "monitor":
            manager.monitor_services(interval=args.interval)

        elif args.command == "log":
            if args.clear:
                if not manager.clear_log(args.name):
                    print(f"Failed to clear log for service '{args.name}'.")
                    return 1
                print(f"Log cleared for service '{args.name}'.")
            else:
                log_lines = manager.tail_log(args.name, args.lines)
                if log_lines:
                    print(f"Last {args.lines} lines of log for service '{args.name}':")
                    print("".join(log_lines), end="")
                else:
                    print(f"No log entries found for service '{args.name}'.")

    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        return 1
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        print(f"Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())