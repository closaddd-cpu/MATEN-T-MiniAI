# BY MATEN/T | M/ License v1.0
"""Stream hash/line verification. No data loaded fully into RAM."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def verify(folder):
    folder=Path(folder); manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    total=0;failures=[]
    for name,rec in manifest['domains'].items():
        path=folder/rec['path'];count=0;digest=hashlib.sha256()
        with path.open('rb') as source:
            first=source.readline();count=1 if first else 0;digest.update(first)
            while True:
                chunk=source.read(4<<20)
                if not chunk:break
                count+=chunk.count(b'\n'); digest.update(chunk)
        # Builder appends newline to every row.
        if first and not first.endswith(b'\n'):failures.append((name,'no newline on first row'))
        try:
            line=json.loads(first)
            if line.get('license')!='M/ License v1.0' or line.get('created_by')!='MATEN/T':failures.append((name,'metadata mismatch'))
            if not line.get('instruction') or not line.get('response'):failures.append((name,'empty example'))
        except (ValueError,UnicodeError) as exc:failures.append((name,'invalid first line: '+str(exc)))
        if count!=rec['rows']:failures.append((name,f'count {count} != expected {rec["rows"]}'))
        if digest.hexdigest()!=rec['sha256']:failures.append((name,'SHA-256 mismatch'))
        total+=count
    if total!=manifest['total_records']:failures.append(('all',f'rows {total} != expected {manifest["total_records"]}'))
    print(f'BY MATEN/T | audited {len(manifest["domains"])} JSONL files, {total:,} rows. Problems: {len(failures)}')
    for x in failures:print('FAIL',*x)
    return not failures

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default=str(ROOT/'datasets'));args=p.parse_args()
    raise SystemExit(0 if verify(args.data) else 1)
