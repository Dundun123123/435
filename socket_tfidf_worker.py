"""On laptop B: connect outward to coordinator A and run two CPU workers."""
import argparse
import multiprocessing
import socket
import time
from multiprocessing.connection import Client
from pathlib import Path

from socket_tfidf import finish_workers, merge_df, receive, send_idf, start_workers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', required=True)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--key-file', required=True)
    args = parser.parse_args()
    key = bytes.fromhex(Path(args.key_file).read_text(encoding='ascii').strip())
    if len(key) < 16:
        parser.error('Key file must contain at least 16 random bytes in hex')
    last_error = None
    for _ in range(30):
        try:
            connection = Client((args.server, args.port), authkey=key)
            break
        except OSError as exc:
            last_error = exc
            time.sleep(1)
    else:
        raise ConnectionError(f'Unable to connect to {args.server}:{args.port}') from last_error
    with connection:
        connection.send({'hostname': socket.gethostname(), 'worker_processes': 2})
        while True:
            message = connection.recv()
            if message['kind'] == 'stop':
                print('Coordinator finished; worker exiting.', flush=True)
                return
            if message['kind'] != 'task' or len(message['shards']) != 2:
                raise ValueError('Expected two-shard task')
            workers = start_workers(message['shards'], message['top_k'])
            try:
                connection.send(('df', merge_df(receive(workers, 'df'))))
                kind, idf = connection.recv()
                if kind != 'idf':
                    raise ValueError(f'Expected idf, got {kind}')
                send_idf(workers, idf)
                connection.send(('vectors', receive(workers, 'vectors')))
            finally:
                finish_workers(workers)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
