"""Real-device array probe only. This does NOT implement TF-IDF."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/local/gpu_probe.json')
    args = parser.parse_args()
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError('No CUDA device available')
        with cp.cuda.Device(0):
            values = cp.arange(1024, dtype=cp.float64)
            squared_sum = (values * values).sum()
            cp.cuda.Stream.null.synchronize()
            actual = float(squared_sum.get())
            expected = float(sum(i * i for i in range(1024)))
            if actual != expected:
                raise AssertionError('GPU array result did not match expected value')
            properties = cp.cuda.runtime.getDeviceProperties(0)
            name = properties['name']
            if isinstance(name, bytes):
                name = name.decode('utf-8', errors='replace')
            report = {'status': 'passed', 'probe_only': True, 'cupy': cp.__version__,
                'gpu_name': name, 'gpu_memory_bytes': int(properties['totalGlobalMem']),
                'driver_version': cp.cuda.runtime.driverGetVersion(),
                'runtime_version': cp.cuda.runtime.runtimeGetVersion(),
                'check_sum_squares': actual}
            code = 0
    except Exception as exc:
        report = {'status': 'failed' if isinstance(exc, AssertionError) else 'unavailable', 'probe_only': True, 'error_type': type(exc).__name__,
                  'message': str(exc), 'next_step': 'Read docs/GPU_BACKEND.md; do not label a CPU fallback as GPU.'}
        code = 2
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
