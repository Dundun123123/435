"""Read-only local capability report; never installs drivers or dependencies."""
import argparse
import importlib.util
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/local/doctor.json')
    args = parser.parse_args()
    report = {'python': sys.version, 'executable': sys.executable,
        'os': platform.platform(), 'processor': platform.processor(),
        'logical_cpus': os.cpu_count(), 'hostname': socket.gethostname(),
        'optional_modules': {name: importlib.util.find_spec(name) is not None
                             for name in ('sklearn', 'matplotlib', 'mpi4py', 'cupy')},
        'executables': {name: shutil.which(name) for name in ('git', 'mpiexec', 'mpirun', 'nvidia-smi', 'nvcc')},
        'note': 'Module availability is not proof of a working runtime. Run diagnostics probes. This local report may contain host/path information; do not publish unreviewed.'}
    if shutil.which('nvidia-smi'):
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
                                    capture_output=True, text=True, timeout=10)
            report['nvidia_query'] = {'returncode': result.returncode, 'stdout': result.stdout.strip()}
        except (OSError, subprocess.TimeoutExpired) as exc:
            report['nvidia_query'] = {'status': 'unavailable', 'error_type': type(exc).__name__}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
