"""Summarize fixed-dev results without inventing a task-success definition."""
import argparse
import csv
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',type=Path,required=True)
    args=p.parse_args()
    rows=[]
    identities=None
    for path in sorted((args.run_dir/'validation').glob('dev64_step_*.json')):
        result=json.loads(path.read_text())
        ids=tuple(sorted(result['task_ids']))
        if len(ids)!=64 or len(set(ids))!=64: raise ValueError('Expected 64 unique validation identities')
        if identities is None: identities=ids
        if ids!=identities: raise ValueError('Validation set changed across steps')
        rows.append({'step':result['step'],**result['means']})
    if not rows: raise SystemExit('No completed fixed-dev validation files yet')
    keys=['step']+sorted(set().union(*(row.keys() for row in rows))-{'step'})
    output=args.run_dir/'validation_summary.csv'
    with output.open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    print(json.dumps({'updates':[r['step'] for r in rows],'same_dev64':True,'summary':str(output)},indent=2))


if __name__=='__main__':main()
