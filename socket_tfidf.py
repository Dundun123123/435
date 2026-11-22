"""CPU worker protocol shared by one-host and two-host socket benchmarks."""
import multiprocessing as mp
from collections import Counter

from tfidf import count_batch, weight_batch


def _worker(connection, texts, top_k):
    try:
        counts, df = count_batch(texts)
        connection.send(('df', df))
        kind, idf = connection.recv()
        if kind != 'idf':
            raise ValueError(f'Expected idf, got {kind}')
        connection.send(('vectors', weight_batch(counts, idf, top_k)))
    except BaseException as exc:
        try:
            connection.send(('error', f'{type(exc).__name__}: {exc}'))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


def start_workers(shards, top_k):
    context = mp.get_context('spawn')
    workers = []
    for texts in shards:
        parent, child = context.Pipe()
        process = context.Process(target=_worker, args=(child, texts, top_k))
        process.start()
        child.close()
        workers.append((parent, process))
    return workers


def receive(workers, expected):
    values = []
    for connection, process in workers:
        try:
            kind, value = connection.recv()
        except EOFError as exc:
            raise RuntimeError(f'CPU worker exited with code {process.exitcode}') from exc
        if kind != expected:
            raise RuntimeError(value if kind == 'error' else f'Expected {expected}, got {kind}')
        values.append(value)
    return values


def send_idf(workers, idf):
    for connection, _ in workers:
        connection.send(('idf', idf))


def finish_workers(workers):
    for connection, process in workers:
        connection.close()
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join()


def merge_df(dfs):
    total = Counter()
    for df in dfs:
        total.update(df)
    return total
