"""Pair hash-verified public ACS designs with the included frozen reference draws."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--prepared',type=Path,default=ROOT/'validation/datasets/processed/acs2024_ca_wages')
a=p.parse_args()
columns=json.loads((a.prepared/'columns.json').read_text(encoding='utf8'))
primary=np.load(a.prepared/'x.npy')
keep=[j for j,name in enumerate(columns) if not name.startswith('county_union[')]
for name,x in [('primary',primary),('pooled',primary[:,keep])]:
    frozen=ROOT/'validation/frozen'/name
    meta=json.loads((frozen/'meta.json').read_text())
    out=ROOT/'.cache'/f'acs_{name}';out.mkdir(parents=True,exist_ok=True)
    np.save(out/'x.npy',np.ascontiguousarray(x))
    for file in ('y.npy','cluster.npy'):shutil.copyfile(a.prepared/file,out/file)
    for file in ('W.npy','base_beta.npy'):shutil.copyfile(frozen/file,out/file)
    for file,expected in meta['input_sha256'].items():
        actual=hashlib.sha256((out/file).read_bytes()).hexdigest()
        if actual!=expected:
            raise ValueError(f'{name}/{file}: regenerated design differs from frozen reference; do not pair mismatched inputs')
    (out/'meta.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf8')
    print(f'{name}: verified {x.shape[0]} rows, {x.shape[1]} columns')
