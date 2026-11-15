from tfidf import compute_tfidf, read_documents
from pathlib import Path
import math, json

def equal(a,b):
    assert len(a)==len(b)
    for x,y in zip(a,b):
        assert x.keys()==y.keys()
        assert all(math.isclose(x[k],y[k],rel_tol=1e-10,abs_tol=1e-12) for k in x)

if __name__=='__main__':
    texts=[d['text'] for d in read_documents('data/newsgroups.jsonl')]
    seq,si,_=compute_tfidf(texts)
    par,pi,_=compute_tfidf(texts,'parallel',4)
    assert si==pi
    equal(seq,par)
    del par
    expected=[dict(sorted(v.items(),key=lambda x:(-x[1],x[0]))[:10]) for v in seq]
    for mode in ('sequential','parallel'):
        actual,_,_=compute_tfidf(texts,mode,4,top_k=10)
        equal(expected,actual)
    report={'implementation':'packed-shard-processes-v3','documents':len(texts),'vocabulary':len(si),'full_vectors_match':True,'top10_matches_full_vectors_in_both_modes':True,'rel_tol':1e-10,'abs_tol':1e-12}
    Path('results/full_corpus_verification_v3.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))
