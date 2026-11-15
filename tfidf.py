"""Shared TF-IDF pipeline. Core computation uses only the standard library."""
import math
import heapq
import re
import multiprocessing as mp
import traceback
from array import array
from collections import Counter
from multiprocessing.connection import wait
from time import perf_counter

TOKEN = re.compile(r"[a-z]+")


def count_batch(texts):
    counts = [Counter(TOKEN.findall(text.lower())) for text in texts]
    df = Counter()
    for count in counts:
        df.update(count.keys())
    return counts, df


def weight_batch(counts, idf, top_k=None):
    result = []
    for count in counts:
        weights = {word: frequency * idf[word] for word, frequency in count.items()}
        norm = math.sqrt(sum(value * value for value in weights.values()))
        if top_k is not None:
            weights = dict(heapq.nsmallest(top_k, weights.items(), key=lambda item: (-item[1], item[0])))
        result.append({word: value / norm for word, value in weights.items()} if norm else {})
    return result


def pipeline(texts, batch_size, top_k=None, profile=None):
    start = perf_counter()
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    counted = list(map(count_batch, batches))
    counted_at = perf_counter()
    df = Counter()
    for _, local_df in counted:
        df.update(local_df)
    if not df:
        raise ValueError('The corpus contains no English tokens matching [a-z]+.')
    idf = {word: math.log((len(texts) + 1) / (frequency + 1)) + 1
           for word, frequency in df.items()}
    idf_at = perf_counter()
    weighted = (weight_batch(counts, idf, top_k) for counts, _ in counted)
    vectors = [vector for batch in weighted for vector in batch]
    if profile is not None:
        profile.update(stages={'count_seconds': counted_at - start,
            'df_merge_and_idf_seconds': idf_at - counted_at,
            'weight_and_collect_seconds': perf_counter() - idf_at}, active_workers=1)
    return vectors, idf


def _worker(connection, batches, top_k):
    """Keep term counts local across the global-IDF synchronization barrier."""
    try:
        start = perf_counter()
        counted = []
        local_df = Counter()
        for indices, texts in batches:
            counts, df = count_batch(texts)
            counted.append((indices, counts))
            local_df.update(df)
        connection.send(('df', local_df))
        counted_at = perf_counter()
        idf = connection.recv()
        idf_at = perf_counter()
        # Binary numeric arrays replace repeated Python dictionaries on the wire.
        # Parent and child see the same insertion-ordered global IDF vocabulary.
        word_ids = {word: i for i, word in enumerate(idf)}
        indices_out, offsets, terms, values = array('I'), array('Q', [0]), array('I'), array('d')
        for indices, counts in counted:
            vectors = weight_batch(counts, idf, top_k)
            for index, vector in zip(indices, vectors):
                indices_out.append(index)
                for word, value in vector.items():
                    terms.append(word_ids[word])
                    values.append(value)
                offsets.append(len(terms))
        stats = {'count_and_df_send_seconds': counted_at - start,
                 'idf_wait_seconds': idf_at - counted_at,
                 'weight_and_pack_seconds': perf_counter() - idf_at,
                 'documents': len(indices_out),
                 'numeric_payload_bytes': sum(len(a) * a.itemsize for a in (indices_out, offsets, terms, values))}
        connection.send(('vectors', (indices_out, offsets, terms, values, stats)))
    except BaseException:
        connection.send(('error', traceback.format_exc()))
    finally:
        connection.close()


def _receive(connection, process, expected):
    ready = wait([connection, process.sentinel])
    if connection not in ready:
        raise RuntimeError(f'Worker exited unexpectedly (exitcode={process.exitcode})')
    try:
        kind, value = connection.recv()
    except EOFError as exc:
        raise RuntimeError('Worker closed its connection unexpectedly') from exc
    if kind != expected:
        raise RuntimeError(value if kind == 'error' else f'Expected {expected}, got {kind}')
    return value


def parallel_pipeline(texts, workers, batch_size, top_k=None, profile=None):
    start = perf_counter()
    batches = [(list(range(i, min(i + batch_size, len(texts)))), texts[i:i + batch_size])
               for i in range(0, len(texts), batch_size)]
    count = min(workers, len(batches))
    shards = [[] for _ in range(count)]
    loads = [0] * count
    # Greedy assignment by text length reduces imbalance from unusually long posts.
    for batch in sorted(batches, key=lambda b: sum(map(len, b[1])), reverse=True):
        target = min(range(count), key=lambda i: loads[i])
        shards[target].append(batch)
        loads[target] += sum(map(len, batch[1]))
    context = mp.get_context('spawn')
    partitioned_at = perf_counter()
    connections, processes = [], []
    try:
        for shard in shards:
            parent, child = context.Pipe()
            process = context.Process(target=_worker, args=(child, shard, top_k))
            process.start()
            child.close()
            connections.append(parent)
            processes.append(process)
        dispatched_at = perf_counter()
        df = Counter()
        for connection, process in zip(connections, processes):
            df.update(_receive(connection, process, 'df'))
        counted_at = perf_counter()
        if not df:
            raise ValueError('The corpus contains no English tokens matching [a-z]+.')
        idf = {word: math.log((len(texts) + 1) / (frequency + 1)) + 1
               for word, frequency in df.items()}
        for connection in connections:
            connection.send(idf)
        idf_at = perf_counter()
        vocabulary = list(idf)
        vectors = [None] * len(texts)
        worker_stats = []
        for connection, process in zip(connections, processes):
            indices, offsets, terms, values, stats = _receive(connection, process, 'vectors')
            worker_stats.append(stats)
            for row, index in enumerate(indices):
                vectors[index] = {vocabulary[terms[j]]: values[j]
                                  for j in range(offsets[row], offsets[row + 1])}
        collected_at = perf_counter()
        for process in processes:
            process.join()
        if profile is not None:
            profile.update(active_workers=count, workers=worker_stats, stages={
                'partition_seconds': partitioned_at - start,
                'startup_and_dispatch_seconds': dispatched_at - partitioned_at,
                'count_wait_and_df_merge_seconds': counted_at - dispatched_at,
                'idf_compute_and_broadcast_seconds': idf_at - counted_at,
                'weight_wait_and_result_merge_seconds': collected_at - idf_at,
                'shutdown_seconds': perf_counter() - collected_at})
        return vectors, idf
    finally:
        for connection in connections:
            connection.close()
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join()


def compute_tfidf(texts, mode='sequential', workers=4, batch_size=500, top_k=None, profile=None):
    if mode not in ('sequential', 'parallel'):
        raise ValueError('mode must be sequential or parallel')
    if not texts or batch_size < 1 or workers < 1:
        raise ValueError('Nonempty input, positive workers and positive batch size required.')
    if top_k is not None and top_k < 1:
        raise ValueError('top_k must be positive or None')
    start = perf_counter()
    if profile is not None:
        profile.clear()
    if mode == 'sequential':
        vectors, idf = pipeline(texts, batch_size, top_k, profile)
    else:
        vectors, idf = parallel_pipeline(texts, workers, batch_size, top_k, profile)
    elapsed = perf_counter() - start
    if profile is not None:
        profile.update(total_seconds=elapsed, implementation='packed-shard-processes-v3',
            timing_note='Parent stages are consecutive wall-clock intervals; worker work overlaps them. Worker durations must not be added to parent total.')
    return vectors, idf, elapsed


def read_documents(path, limit=None):
    import json
    if limit is not None and limit < 1:
        raise ValueError('limit must be positive')
    documents, seen = [], set()
    with open(path, encoding='utf-8-sig') as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or not isinstance(record.get('id'), str) or not isinstance(record.get('text'), str):
                raise ValueError(f'Line {line_number}: expected string id and text')
            if record['id'] in seen:
                raise ValueError(f'Duplicate document id: {record["id"]}')
            seen.add(record['id'])
            documents.append(record)
            if limit and len(documents) >= limit:
                break
    if not documents:
        raise ValueError('Input has no documents')
    return documents
