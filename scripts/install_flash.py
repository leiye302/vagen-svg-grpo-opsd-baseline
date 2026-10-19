"""Install the official matching FA wheel; retain its digest for the run manifest."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request
import torch

ROOT = Path(__file__).resolve().parents[1]
assert torch.__version__.startswith('2.7.0'), 'Expected pinned torch 2.7.0'
assert sys.version_info[:2] == (3,12), 'Expected Python 3.12'
abi = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()
name = f'flash_attn-2.8.3+cu12torch2.7cxx11abi{abi}-cp312-cp312-linux_x86_64.whl'
url = 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/' + name
cache = ROOT/'work/install_cache'
cache.mkdir(parents=True, exist_ok=True)
wheel = cache/name
if not wheel.exists():
    temporary = wheel.with_suffix('.partial')
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as out:
        while block := response.read(8*1024**2): out.write(block)
    temporary.replace(wheel)
digest = hashlib.file_digest(wheel.open('rb'), 'sha256').hexdigest()
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', str(wheel)], check=True)
(cache/'flash_manifest.json').write_text(json.dumps({'url':url, 'sha256':digest},indent=2))
