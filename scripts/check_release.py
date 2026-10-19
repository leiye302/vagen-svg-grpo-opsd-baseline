"""Fail closed on accidental private payloads in the files intended for release."""
import argparse
import ast
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
EXCLUDE={'vendor','work','.venv','.git','__pycache__','.pytest_cache','outputs','wandb'}
PATTERNS={
    'credential':r'(?:oc_sk_|sk-[A-Za-z0-9]|ghp_|github_pat_)[A-Za-z0-9_+\-/]{18,}',
    'private_ssh_host':r'connect\.[A-Za-z0-9.]+\.seetacloud\.com',
    'machine_path':r'(?:[A-Za-z]:[\\/](?:Users|app|Desktop)[\\/]|/root/(?:autodl-tmp)/)',
    'private_key':r'-----BEGIN (?:OPENSSH|RSA|EC|PRIVATE) PRIVATE KEY-----',
}


def files(staged=False):
    if staged:
        names=subprocess.check_output(['git','-C',str(ROOT),'diff','--cached','--name-only','--diff-filter=ACM'],text=True).splitlines()
        return [ROOT/n for n in names]
    return [p for p in ROOT.rglob('*') if p.is_file() and not any(x in EXCLUDE for x in p.relative_to(ROOT).parts)]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--staged',action='store_true');args=parser.parse_args()
    issues=[]; paths=files(args.staged)
    for p in paths:
        rel=p.relative_to(ROOT).as_posix()
        if p.suffix.lower() in {'.pt','.safetensors','.parquet','.zip','.tar','.dpapi','.pem','.key','.png','.jpg'}:
            issues.append((rel,'private_or_binary_payload'));continue
        try: text=p.read_text(encoding='utf8')
        except UnicodeDecodeError: issues.append((rel,'non_text_payload'));continue
        for reason,pattern in PATTERNS.items():
            if re.search(pattern,text):issues.append((rel,reason))
        if p.suffix=='.py':
            try:ast.parse(text,filename=rel)
            except SyntaxError:issues.append((rel,'syntax_error'))
    if issues:
        for path,reason in issues:print(path+': '+reason)
        raise SystemExit(1)
    print(f'PASS: {len(paths)} release files; no matching credential/private host/path or binary payload')


if __name__=='__main__':main()
