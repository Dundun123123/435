"""Download real IMDb reviews; stream the archive without extracting 100k files."""
import argparse
import hashlib
import html
import json
import random
import re
import tarfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from tfidf import TOKEN

SOURCE = 'https://ai.stanford.edu/~amaas/data/sentiment/'
URL = 'https://d2l-data.s3-accelerate.amazonaws.com/aclImdb_v1.tar.gz'
EXPECTED_SHA1 = '01ada507287d82875905620988597833ad4e0903'


def write_jsonl(path, docs):
    with path.open('w', encoding='utf-8') as target:
        for doc in docs:
            target.write(json.dumps(doc, ensure_ascii=False) + '\n')
    with path.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    return {'file': path.name, 'documents': len(docs),
            'tokens': sum(d['tokens'] for d in docs), 'bytes': path.stat().st_size,
            'sha256': digest}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--length-subset-size', type=int, default=5000)
    args = parser.parse_args()
    if args.length_subset_size < 1:
        parser.error('--length-subset-size must be positive')
    root = Path(__file__).resolve().parent / 'data'
    cache = root / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / 'aclImdb_v1.tar.gz'
    if not archive.exists():
        temporary = archive.with_suffix('.download')
        print(f'Downloading {URL}', flush=True)
        request = urllib.request.Request(URL, headers={'User-Agent': 'CST435W educational TF-IDF project'})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open('wb') as dest:
            total = 0
            while block := response.read(1024 * 1024):
                dest.write(block)
                total += len(block)
                if total % (10 * 1024 * 1024) == 0:
                    print(f'Downloaded {total / 1024**2:.0f} MiB', flush=True)
        temporary.replace(archive)
    with archive.open('rb') as source:
        if hashlib.file_digest(source, 'sha1').hexdigest() != EXPECTED_SHA1:
            raise ValueError('Archive checksum does not match the D2L published SHA-1; remove the bad cached archive and retry.')
    seen, docs = set(), []
    original = duplicates = empty = 0
    pattern = re.compile(r'aclImdb/(train|test)/(pos|neg|unsup)/[^/]+\.txt$')
    with tarfile.open(archive, 'r|gz') as source:
        for member in source:
            if member.name == 'aclImdb/README' and member.isfile():
                (root / 'IMDB_ORIGINAL_README.txt').write_bytes(source.extractfile(member).read())
            match = pattern.fullmatch(member.name)
            if not match or not member.isfile():
                continue
            original += 1
            text = source.extractfile(member).read().decode('utf-8')
            text = ' '.join(html.unescape(re.sub(r'<[^>]+>', ' ', text)).split())
            digest = hashlib.sha256(text.encode('utf-8')).digest()
            if digest in seen:
                duplicates += 1
                continue
            seen.add(digest)
            tokens = len(TOKEN.findall(text.lower()))
            if not tokens:
                empty += 1
                continue
            docs.append({'id': 'imdb/' + member.name.removeprefix('aclImdb/'),
                         'text': text, 'tokens': tokens, 'split': match[1], 'label': match[2]})
            if len(docs) % 20000 == 0:
                print(f'Prepared {len(docs)} unique reviews', flush=True)
    random.Random(42).shuffle(docs)
    files = [write_jsonl(root / 'imdb.jsonl', docs)]
    # Equal-size samples from disjoint bottom/top quartiles, not duplicated text.
    ranked = sorted(docs, key=lambda d: (d['tokens'], d['id']))
    quartile = len(ranked) // 4
    size = min(args.length_subset_size, quartile)
    for name, population in [('short', ranked[:quartile]), ('long', ranked[-quartile:])]:
        subset = random.Random(42).sample(population, size)
        files.append(write_jsonl(root / f'imdb_{name}.jsonl', subset))
    with archive.open('rb') as source:
        archive_sha = hashlib.file_digest(source, 'sha256').hexdigest()
    manifest = {'dataset': 'Stanford Large Movie Review Dataset v1.0 (IMDb)',
        'source': SOURCE, 'download_url': URL, 'mirror_reference': 'https://d2l.ai/_modules/d2l/torch.html',
        'published_sha1': EXPECTED_SHA1, 'downloaded_utc': datetime.now(timezone.utc).isoformat(),
        'archive_sha256': archive_sha, 'original_reviews': original, 'removed_duplicates': duplicates,
        'removed_empty': empty, 'random_state': 42, 'files': files,
        'preprocessing': 'Remove HTML tags, unescape entities, collapse whitespace, deduplicate identical cleaned UTF-8 text. Labels are metadata only.',
        'length_sampling': 'Equal-size seeded random samples from bottom/top token-count quartiles; no overlap; length is confounded with content, not a controlled causal experiment.',
        'citation': 'Maas et al. (2011), Learning Word Vectors for Sentiment Analysis, ACL, https://aclanthology.org/P11-1015/',
        'usage': 'Original review rights are not relicensed by this project. Consult original dataset README and source conditions.'}
    (root / 'imdb_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
