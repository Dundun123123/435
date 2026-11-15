"""Download 20 Newsgroups via scikit-learn and export reproducible JSONL."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from sklearn.datasets import fetch_20newsgroups
from tfidf import TOKEN


def main():
    root = Path(__file__).resolve().parent / 'data'
    root.mkdir(exist_ok=True)
    corpus = fetch_20newsgroups(subset='all', data_home=str(root / 'cache'),
        remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=42)
    target = root / 'newsgroups.jsonl'
    kept = tokens = 0
    with target.open('w', encoding='utf-8') as dest:
        for index, (text, label) in enumerate(zip(corpus.data, corpus.target)):
            words = TOKEN.findall(text.lower())
            if not words:
                continue
            row = {'id': f'ng{index:05d}', 'text': text, 'category': corpus.target_names[int(label)]}
            dest.write(json.dumps(row, ensure_ascii=False) + '\n')
            kept += 1
            tokens += len(words)
    manifest = {'dataset': '20 Newsgroups (discussion posts, not news articles)',
        'source': 'https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset',
        'downloaded_utc': datetime.now(timezone.utc).isoformat(), 'random_state': 42,
        'removed': ['headers', 'footers', 'quotes'], 'original_documents': len(corpus.data),
        'documents': kept, 'filtered_empty_documents': len(corpus.data) - kept,
        'tokens': tokens, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
        'usage_note': 'Public availability is not a new license grant. Consult the original source terms; no ownership of source posts is claimed.'}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
