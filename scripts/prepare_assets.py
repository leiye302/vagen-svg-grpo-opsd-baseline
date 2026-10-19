#!/usr/bin/env python3
"""Fetch public pinned assets once; record portable hashes for offline training.

No training, paid API, private backup, or GPU is used by this command.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
DINO_REPO = "facebook/dinov2-small"
DINO_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
DREAMSIM_VERSION = "0.2.1"
DREAMSIM_DINO_URL = "https://github.com/facebookresearch/dino.git"
DREAMSIM_DINO_REVISION = "7c446df5b9f45747937fb0d72314eb9f7b66930a"
MODEL_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken", "*.jinja", "LICENSE*", "README.md"]
DATA_PATTERNS = ["data/*.parquet", "README.md", ".gitattributes"]


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def checked_relative(work: Path, relative: str) -> Path:
    """Reject a modified manifest escaping the user-selected work directory."""
    work = Path(work).resolve()
    path = (work / relative).resolve()
    if path == work or not path.is_relative_to(work):
        raise ValueError(f"Asset path escapes work directory: {relative}")
    return path


def inventory(work: Path, root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or any(part in relative.parts for part in (".cache", ".git", "__pycache__")) or path.name.endswith((".tmp", ".lock", ".incomplete", ".pyc")):
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(work.resolve()):
            raise ValueError("Asset symlink points outside work directory")
        rows.append({"path": path.relative_to(work).as_posix(), "size_bytes": path.stat().st_size, "bytes": path.stat().st_size,
                     "sha256": file_hash(path)})
    if not rows:
        raise ValueError(f"Empty asset directory: {root.name}")
    return rows


def verify_inventory(work: Path, rows: list[dict]) -> None:
    if not rows or len({row["path"] for row in rows}) != len(rows):
        raise ValueError("Empty or duplicated asset inventory")
    for row in rows:
        path = checked_relative(work, row["path"])
        if not path.is_file() or path.stat().st_size != row["size_bytes"] or file_hash(path) != row["sha256"]:
            raise ValueError(f"Missing or modified asset: {row['path']}")


def hf_specs(lock: dict) -> dict:
    specs = {
        "policy": {**lock["policy"], "repo_type": "model", "directory": "models/Qwen2.5-VL-3B-Instruct", "patterns": MODEL_PATTERNS},
        "dataset": {**lock["dataset"], "repo_type": "dataset", "directory": "data/svg-icons-simple", "patterns": DATA_PATTERNS},
        "dino": {"repo_id": DINO_REPO, "revision": DINO_REVISION, "repo_type": "model", "directory": "models/dinov2-small", "patterns": MODEL_PATTERNS},
    }
    if "dino" in lock and (lock["dino"]["repo_id"], lock["dino"]["revision"]) != (DINO_REPO, DINO_REVISION):
        raise ValueError("DINO source lock differs from preparation contract")
    for spec in specs.values():
        if not re.fullmatch(r"[0-9a-f]{40}", spec["revision"]):
            raise ValueError("Every HF download must use a full commit revision")
    return specs


def validate_hf_files(name: str, directory: Path) -> None:
    if name == "dataset":
        required = [f"data/{split}-00000-of-00001.parquet" for split in ("train", "val", "test")]
    else:
        required = ["config.json", "preprocessor_config.json"]
        if name == "policy":
            required += ["tokenizer.json", "tokenizer_config.json"]
        index = directory / "model.safetensors.index.json"
        if index.exists():
            weights = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
            required += sorted(set(weights.values()))
        else:
            required += ["model.safetensors"]
    for relative in required:
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Incomplete {name} asset: {relative}")


def initialize_dreamsim(cache: Path, *, offline: bool = False) -> None:
    """Use the official ensemble loader, never a substitute perceptual metric."""
    if importlib.metadata.version("dreamsim") != DREAMSIM_VERSION:
        raise ValueError(f"Expected dreamsim=={DREAMSIM_VERSION}")
    from dreamsim import dreamsim

    def deny_network(*_args, **_kwargs):
        raise RuntimeError("DreamSim offline check attempted network access")

    if offline:
        with patch.object(socket.socket, "connect", deny_network), patch.object(socket.socket, "connect_ex", deny_network):
            model, _ = dreamsim(pretrained=True, device="cpu", cache_dir=str(cache), dreamsim_type="ensemble")
    else:
        model, _ = dreamsim(pretrained=True, device="cpu", cache_dir=str(cache), dreamsim_type="ensemble")
    model.eval().requires_grad_(False)
    del model


def prepare_dreamsim_source(cache: Path) -> dict:
    """Populate Torch Hub's official cache name with a fixed public commit.

    DreamSim 0.2.1 otherwise resolves facebookresearch/dino:main at runtime.
    The existing cached directory makes its official loader offline/repeatable.
    """
    directory = cache / "facebookresearch_dino_main"
    if directory.exists() and not (directory / ".git").is_dir():
        raise ValueError("Unversioned DreamSim DINO cache exists; use a clean work directory")
    if not directory.exists():
        directory.mkdir(parents=True)
        subprocess.run(["git", "init", "--quiet", str(directory)], check=True)
        subprocess.run(["git", "-C", str(directory), "remote", "add", "origin", DREAMSIM_DINO_URL], check=True)
    head = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"], capture_output=True, text=True)
    if head.returncode != 0:
        subprocess.run(["git", "-C", str(directory), "fetch", "--depth", "1", "origin", DREAMSIM_DINO_REVISION], check=True)
        subprocess.run(["git", "-C", str(directory), "checkout", "--detach", "--quiet", DREAMSIM_DINO_REVISION], check=True)
    actual = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(directory), "diff", "--name-only", "HEAD"], text=True).strip()
    if actual != DREAMSIM_DINO_REVISION or dirty:
        raise ValueError("DreamSim DINO cache source differs from pinned revision")
    return {"url": DREAMSIM_DINO_URL, "revision": actual, "cache_subdirectory": directory.name}


def prepare(work: Path, lock_path: Path, verify_only: bool = False, *, only_data: bool = False) -> dict:
    work = work.expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    specs = hf_specs(lock)
    manifest_path = work / "assets_manifest.json"
    existing = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    requested = {"dataset"} if only_data else {"policy", "dataset", "dino", "dreamsim"}
    if existing:
        if existing["sources_lock_sha256"] != file_hash(lock_path):
            raise ValueError("Source lock changed; use a new work directory")
        for name, spec in specs.items():
            if name not in existing["assets"]:
                continue
            entry = existing["assets"][name]
            if any(entry.get(key) != spec[key] for key in ("repo_id", "revision", "directory")):
                raise ValueError(f"Asset contract changed: {name}")
            validate_hf_files(name, checked_relative(work, spec["directory"]))
        verify_inventory(work, existing["files"])
        if requested.issubset(existing["assets"]):
            return existing
    if verify_only:
        raise ValueError("No complete assets manifest; run preparation with network first")

    from huggingface_hub import snapshot_download
    assets = dict(existing["assets"]) if existing else {}
    for name, spec in specs.items():
        if name not in requested or name in assets:
            continue
        directory = checked_relative(work, spec["directory"])
        print(f"Preparing public asset: {name}", flush=True)
        snapshot_download(repo_id=spec["repo_id"], repo_type=spec["repo_type"], revision=spec["revision"],
                          local_dir=str(directory), allow_patterns=spec["patterns"], token=False)
        validate_hf_files(name, directory)
        assets[name] = {key: value for key, value in spec.items() if key != "patterns"}

    if "dreamsim" in requested and "dreamsim" not in assets:
        cache = checked_relative(work, "cache/dreamsim")
        cache.mkdir(parents=True, exist_ok=True)
        dreamsim_source = prepare_dreamsim_source(cache)
        initialize_dreamsim(cache)
        initialize_dreamsim(cache, offline=True)
        from dreamsim.config import dreamsim_weights
        assets["dreamsim"] = {"directory": "cache/dreamsim", "package": "dreamsim", "version": DREAMSIM_VERSION,
                              "type": "ensemble", "official_weights_url": dreamsim_weights["ensemble"],
                              "source": "https://github.com/ssundaram21/dreamsim", "dino_backbone_source": dreamsim_source,
                              "offline_cpu_reload_passed": True}
    files = []
    for entry in assets.values():
        files.extend(inventory(work, checked_relative(work, entry["directory"])))
    manifest = {"schema": "svg-public-assets-v1", "sources_lock_sha256": file_hash(lock_path),
                "assets": assets, "files": files, "created_unix": time.time(),
                "complete": {"policy", "dataset", "dino", "dreamsim"}.issubset(assets),
                "training_started": False, "private_assets_required": False}
    atomic_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true", help="Verify hashes without downloads or model execution")
    parser.add_argument("--only-data", action="store_true", help="Fetch/verify only the public source dataset, no model weights")
    args = parser.parse_args()
    manifest = prepare(args.work_dir, REPO_ROOT / "sources.lock.json", args.verify_only, only_data=args.only_data)
    print(json.dumps({"complete": manifest["complete"], "requested_assets_ready": True, "files": len(manifest["files"]), "manifest": str(args.work_dir / "assets_manifest.json")}))


if __name__ == "__main__":
    main()
