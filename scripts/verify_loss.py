"""Compare exported executable loss ASTs to the public pinned SDAR source."""
import ast
import copy
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def functions(text):
    return {n.name:n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}


def executable(node):
    node=copy.deepcopy(node)
    if (node.body and isinstance(node.body[0],ast.Expr) and
        isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str)):
        node.body.pop(0)
    return ast.dump(node,include_attributes=False)


def main():
    revision=json.loads((ROOT/'sources.lock.json').read_text())['SDAR']['commit']
    prefix=f'https://raw.githubusercontent.com/ZJU-REAL/SDAR/{revision}/verl/trainer/ppo/'
    ours=functions((ROOT/'overlays/verl/verl/trainer/ppo/sdar_utils.py').read_text())
    for filename,names in [('sdar_utils.py',['compute_sdar_loss']),
                           ('core_algos.py',['agg_loss','compute_policy_loss'])]:
        with urllib.request.urlopen(prefix+filename,timeout=60) as response:
            original=functions(response.read().decode())
        for name in names:
            assert executable(ours[name])==executable(original[name]),f'Public loss mismatch: {name}'
            print(name+': executable AST matches pinned public SDAR')


if __name__ == '__main__': main()
