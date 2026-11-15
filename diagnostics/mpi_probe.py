"""MPI connectivity probe only. This does NOT implement TF-IDF."""
import argparse
import json
import platform
import socket
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-hosts', type=int, default=1)
    parser.add_argument('--output', default='results/local/mpi_probe.json')
    args = parser.parse_args()
    try:
        from mpi4py import MPI
    except (ImportError, RuntimeError, OSError) as exc:
        print(json.dumps({'status': 'unavailable', 'probe_only': True, 'error_type': type(exc).__name__,
            'next_step': 'Install a compatible MPI runtime and requirements-mpi.txt; see docs/MPI_TWO_NODES.md'}))
        return 2
    comm = MPI.COMM_WORLD
    local = {'rank': comm.rank, 'hostname': socket.gethostname(), 'mpi_processor_name': MPI.Get_processor_name(),
             'executable': sys.executable, 'python': platform.python_version()}
    rows = comm.gather(local, root=0)
    ok = None
    if comm.rank == 0:
        hosts = sorted({row['hostname'] for row in rows})
        ok = len(hosts) >= args.require_hosts
        report = {'status': 'passed' if ok else 'failed', 'probe_only': True,
            'rank_count': comm.size, 'distinct_hosts': hosts, 'required_hosts': args.require_hosts,
            'library_version': MPI.Get_library_version(), 'ranks': rows,
            'note': 'Distinct hostnames are placement evidence, not proof of distinct physical machines. Confirm actual hardware separately.'}
        try:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2), encoding='utf-8')
        except OSError as exc:
            ok = False
            report['status'] = 'failed'
            report['write_error_type'] = type(exc).__name__
        print(json.dumps(report, indent=2), flush=True)
    ok = comm.bcast(ok, root=0)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
