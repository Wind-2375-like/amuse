import sys
import pandas as pd
sys.path.insert(0, '/mnt/home/yfeng/repos/nlp_project')
from data.sentences import split_sentences

df = pd.read_parquet('/mnt/home/yfeng/repos/nlp_project/data/aggrefact/aggrefact.parquet')
print('all origins:', df['origin'].value_counts().to_dict(), flush=True)
print(flush=True)

# All origins present in the parquet, ordered by row count (most common first).
targets = df['origin'].value_counts().index.tolist()
for o in targets:
    sub = df[df['origin']==o]
    if not len(sub):
        print(o,'NOT FOUND', flush=True); continue
    print(f'=== {o} (n={len(sub)}) ===', flush=True)
    mc = sub['model'].value_counts()
    print(f'  models ({len(mc)} unique):', flush=True)
    for m, c in mc.head(20).items():
        print(f'    {m}: {c}', flush=True)
    print(f'  splits: {sub["split"].value_counts().to_dict()}', flush=True)
    print(f'  human_label mean: {sub["human_label"].astype(float).mean():.3f}', flush=True)
    print(f'  summary char-len mean/median: {sub["summary"].str.len().mean():.0f} / {sub["summary"].str.len().median():.0f}', flush=True)
    sample = sub.sample(n=min(150, len(sub)), random_state=0)
    n_sents = sample['summary'].apply(lambda s: len(split_sentences(str(s))))
    print(f'  spaCy sentence-count (n={len(sample)} sample): mean={n_sents.mean():.2f}  median={n_sents.median():.0f}  pct(==1)={(n_sents==1).mean():.2%}  pct(>=2)={(n_sents>=2).mean():.2%}  pct(>=3)={(n_sents>=3).mean():.2%}', flush=True)
    print(f'  example summary: {sub["summary"].iloc[0][:400]!r}', flush=True)
    print(flush=True)
