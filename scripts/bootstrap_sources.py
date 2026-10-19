"""Fetch public pinned sources, verify preimages, then install reviewed overlays."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def git(directory, *args):
    return subprocess.check_output(['git', '-C', str(directory), *args], text=True).strip()


def accepted_hashes(hashes, runtime_after=None, *, verify_only=False):
    """A clean upstream preimage is installable, but is never a valid installation."""
    accepted = {hashes['overlay_sha256']}
    if not verify_only:
        accepted.add(hashes['upstream_sha256'])
    if runtime_after is not None:
        accepted.add(runtime_after)
    return accepted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    lock = json.loads((ROOT/'sources.lock.json').read_text())
    entries = json.loads((ROOT/'overlays/manifest.json').read_text())
    for name in ['VAGEN', 'verl']:
        dest = ROOT/'vendor'/name
        source = lock[name]
        if not (dest/'.git').exists():
            if args.verify_only:
                raise RuntimeError(f'Missing source {name}; run bootstrap_sources.py')
            if dest.exists() and any(dest.iterdir()):
                raise RuntimeError(f'Refusing nonempty non-git directory: {dest}')
            dest.mkdir(parents=True, exist_ok=True)
            subprocess.run(['git', 'init', str(dest)], check=True)
            git(dest, 'config', 'core.autocrlf', 'false')
            git(dest, 'remote', 'add', 'origin', source['url'])
        if git(dest, 'remote', 'get-url', 'origin') != source['url']:
            raise RuntimeError(f'{name}: source remote differs from public lock')
        head = subprocess.run(['git', '-C', str(dest), 'rev-parse', '--verify', 'HEAD'],
                              capture_output=True, text=True)
        if head.returncode:
            if args.verify_only:
                raise RuntimeError(f'{name}: initial source download is incomplete')
            # A network failure after git init must be restartable without deleting files.
            git(dest, 'fetch', '--depth', '1', 'origin', source['commit'])
            git(dest, 'checkout', '--detach', 'FETCH_HEAD')
        if git(dest, 'rev-parse', 'HEAD') != source['commit']:
            raise RuntimeError(f'{name}: wrong source commit; do not overwrite existing checkout')
        for key, hashes in entries.items():
            prefix, relative = key.split('/', 1)
            if prefix != name: continue
            target, overlay = dest/relative, ROOT/'overlays'/key
            data = overlay.read_bytes()
            if hashlib.sha256(data).hexdigest() != hashes['overlay_sha256']:
                raise RuntimeError(f'Overlay checksum mismatch: {key}')
            current = hashlib.sha256(target.read_bytes()).hexdigest() if target.exists() else None
            runtime = dest/'.baseline_runtime_fixes.json'
            after = json.loads(runtime.read_text()).get(relative, {}).get('after') if runtime.exists() else None
            if current not in accepted_hashes(hashes, after, verify_only=args.verify_only):
                raise RuntimeError(f'Missing overlay or locally modified source: {key}')
            if after is not None and current == after:
                continue
            if not args.verify_only and current != hashes['overlay_sha256']:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(overlay, target)
        print(f'{name}: pinned commit and overlay checks passed')
    if not args.verify_only:
        subprocess.run([sys.executable, str(ROOT/'scripts/apply_runtime_fixes.py')], check=True)


if __name__ == '__main__': main()
