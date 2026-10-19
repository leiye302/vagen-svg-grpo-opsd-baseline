#!/usr/bin/env python3
"""Independently rebuild pinned train500/dev64 from the public SVG source.

Selection follows the original seed-2026 recipe, not a private task bank.
Official test images are rendered only for duplicate protection, never rollout.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import random
import signal
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from svg_opsd.safety import svg_safety

DATA_REPO = "starvector/svg-icons-simple"
DATA_REVISION = "e1918a27ba6649e856e5db0710d8a6c7046762c1"


def digest(value) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value) -> str:
    return digest(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, data) -> None:
    atomic_text(path, json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def renderer_from_vendor():
    """Load only the pinned renderer, avoiding unrelated Ray/Gym imports."""
    renderer_path = REPO_ROOT / "vendor/VAGEN/vagen/env/svg/svg_utils.py"
    if not renderer_path.is_file():
        raise ValueError("Run scripts/bootstrap_sources.py before preparing data")
    spec = importlib.util.spec_from_file_location("svg_public_renderer", renderer_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(signal, "SIGALRM"):
        raise RuntimeError("Full rendering requires the documented Linux environment")

    def render(svg: str):
        def expired(*_args):
            raise TimeoutError("SVG render timeout")
        previous = signal.signal(signal.SIGALRM, expired)
        signal.alarm(15)
        try:
            _, image = module.process_and_rasterize_svg(svg)
            # Reject an upstream silent fallback: the original must render too.
            import cairosvg
            cairosvg.svg2png(bytestring=svg.encode("utf-8"), output_width=256, output_height=256,
                            background_color="white", dpi=128, scale=2)
            return image.convert("RGB")
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
    return render, file_hash(renderer_path)


def select_tasks(all_rows: dict, render, *, seed: int = 2026, train_tasks: int = 500, dev_tasks: int = 64):
    """Pure selection boundary; tests inject synthetic images, not GPU models."""
    if set(all_rows) != {"train", "val", "test"}:
        raise ValueError("Need all public splits for held-out duplicate protection")
    if min(train_tasks, dev_tasks) <= 0:
        raise ValueError("Task counts must be positive")
    exclusions, protected, public, images = [], set(), [], {}
    raw_heldout = {digest(row["Svg"]) for split in ("val", "test") for row in all_rows[split]}
    protection = {"test": {"rows": 0, "rendered": 0}, "val": {"rows": 0, "rendered": 0}}
    for split in ("test", "val"):
        for row in all_rows[split]:
            protection[split]["rows"] += 1
            safe, _ = svg_safety(row["Svg"])
            if not safe:
                continue
            try:
                image = render(row["Svg"]).convert("RGB")
                protected.add(digest(image.tobytes()))
                protection[split]["rendered"] += 1
            except Exception:
                continue
    for source, split, needed in (("train", "train", train_tasks), ("val", "dev", dev_tasks)):
        order = list(range(len(all_rows[source])))
        random.Random(seed + (0 if source == "train" else 1)).shuffle(order)
        seen, count = set(), 0
        for index in order:
            if count >= needed:
                break
            row = all_rows[source][index]
            svg = row["Svg"]
            code_sha = digest(svg)
            safe, reason = svg_safety(svg)
            if source == "train" and code_sha in raw_heldout:
                safe, reason = False, "heldout_source_duplicate"
            if not safe:
                exclusions.append({"source_split": source, "index": index, "reason": reason})
                continue
            try:
                image = render(svg).convert("RGB")
            except Exception as error:
                exclusions.append({"source_split": source, "index": index, "reason": type(error).__name__})
                continue
            visual_sha = digest(image.tobytes())
            if visual_sha in seen or (source == "train" and visual_sha in protected):
                exclusions.append({"source_split": source, "index": index, "reason": "visual_duplicate"})
                continue
            seen.add(visual_sha)
            task_id = "svg_" + digest(source + ":" + str(row["Filename"]) + ":" + code_sha)[:20]
            if task_id in images:
                raise ValueError("Duplicate public task identity")
            public.append({"task_id": task_id, "split": split, "source_split": source,
                           "source_index": index, "source_filename": str(row["Filename"]),
                           "source_svg_sha256": code_sha, "target_pixels_sha256": visual_sha,
                           "target_image": f"data/targets/{task_id}.png"})
            images[task_id] = image
            count += 1
        if count != needed:
            raise ValueError(f"Not enough eligible {split} tasks: {count}/{needed}")
    train_hashes = {item["target_pixels_sha256"] for item in public if item["split"] == "train"}
    dev_hashes = {item["target_pixels_sha256"] for item in public if item["split"] == "dev"}
    if train_hashes & dev_hashes:
        raise ValueError("Train/dev visual overlap")
    return public, images, exclusions, protection


def identity_hash(tasks: list[dict]) -> str:
    keys = ("task_id", "split", "source_split", "source_index", "source_filename", "source_svg_sha256", "target_pixels_sha256")
    return canonical_hash([{key: task[key] for key in keys} for task in tasks])


def build_rows(tasks: list[dict], work: Path, seed: int = 2026) -> dict:
    """Regenerate absolute environment paths after moving the work directory."""
    work = work.expanduser().resolve()
    result = {"train": [], "dev": []}
    if len({task["task_id"] for task in tasks}) != len(tasks):
        raise ValueError("Duplicate task IDs")
    for position, task in enumerate(tasks):
        split = task["split"]
        if split not in result or task["source_split"] != {"train": "train", "dev": "val"}[split]:
            raise ValueError("Only train/dev may enter rollout data")
        config = {"dataset_name": DATA_REPO, "data_dir": str(work / "data/svg-icons-simple"),
                  "split": task["source_split"], "prompt_format": "grounding_worldmodeling",
                  "fixed_source_index": task["source_index"], "fixed_source_filename": task["source_filename"],
                  "frozen_task_id": task["task_id"], "expected_target_pixels_sha256": task["target_pixels_sha256"],
                  "dino_weight": 0.0001, "dreamsim_weight": 5.0, "structural_weight": 0.0,
                  "format_reward": 0.5, "format_penalty": 0.0}
        result[split].append({"data_source": "svg", "prompt": [{"role": "user", "content": ""}],
                              "extra_info": {"split": split, "task_id": task["task_id"], "env_name": "svg",
                                             "env_config": config, "seed": seed + position * 1000}})
    return result


def check_previous_manifest(previous: dict | None, tasks: list[dict], source_shas: dict) -> None:
    if previous is None:
        return
    if previous["dataset_revision"] != DATA_REVISION or previous["seed"] != 2026:
        raise ValueError("Existing data uses another recipe; use a fresh work directory")
    if previous["source_file_sha256"] != source_shas or previous["identity_sha256"] != identity_hash(tasks):
        raise ValueError("Frozen selection or render changed; refusing to replace dev/train identities")


def prepare(work: Path) -> dict:
    work = work.expanduser().resolve()
    data = work / "data"
    data.mkdir(parents=True, exist_ok=True)
    lock_path = REPO_ROOT / "sources.lock.json"
    source_lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if source_lock["dataset"] != {"repo_id": DATA_REPO, "revision": DATA_REVISION}:
        raise ValueError("Unexpected dataset source lock")
    # Verify downloaded public inputs against the pinned acquisition manifest.
    from prepare_assets import prepare as verify_assets
    verify_assets(work, lock_path, verify_only=True, only_data=True)
    import pyarrow as pa
    import pyarrow.parquet as pq
    all_rows, source_shas = {}, {}
    for split in ("train", "val", "test"):
        path = data / "svg-icons-simple/data" / f"{split}-00000-of-00001.parquet"
        source_shas[split] = file_hash(path)
        all_rows[split] = pq.read_table(path, columns=["Svg", "Filename"]).to_pylist()
    render, render_sha = renderer_from_vendor()
    tasks, images, exclusions, protection = select_tasks(all_rows, render)
    manifest_path = data / "dataset_manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    check_previous_manifest(previous, tasks, source_shas)
    for task in tasks:
        path = work / task["target_image"]
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
        images[task["task_id"]].save(temporary, format="PNG")
        os.replace(temporary, path)
        task["target_image_sha256"] = file_hash(path)
    rows = build_rows(tasks, work)
    splits = {}
    for split, count in (("train", 500), ("dev", 64)):
        path = data / f"{split}{count}.parquet"
        temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
        pq.write_table(pa.Table.from_pylist(rows[split]), temporary)
        os.replace(temporary, path)
        splits[split] = {"count": len(rows[split]), "task_ids": [row["extra_info"]["task_id"] for row in rows[split]],
                         "path": path.relative_to(work).as_posix(), "sha256": file_hash(path)}
    tasks_path = data / "tasks.jsonl"
    exclusions_path = data / "exclusions.jsonl"
    atomic_text(tasks_path, "".join(json.dumps(task, sort_keys=True, ensure_ascii=False) + "\n" for task in tasks))
    atomic_text(exclusions_path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in exclusions))
    manifest = {"schema": "svg-public-dataset-v1", "dataset": DATA_REPO, "dataset_revision": DATA_REVISION,
                "seed": 2026, "splits": splits, "source_file_sha256": source_shas,
                "identity_sha256": identity_hash(tasks), "tasks_sha256": file_hash(tasks_path),
                "tasks_path": "data/tasks.jsonl", "exclusions_sha256": file_hash(exclusions_path),
                "exclusion_counts": dict(Counter(row["reason"] for row in exclusions)),
                "test_used": False, "test_used_definition": "No official-test rollout, gradient, or model selection",
                "official_test_used_for_dedup": True, "official_test_rollouts": 0,
                "heldout_duplicate_protection": protection, "train_heldout_visual_overlap": 0,
                "renderer_sha256": render_sha, "source_lock_sha256": file_hash(lock_path),
                "render_packages": {name: importlib.metadata.version(name) for name in ("CairoSVG", "Pillow", "svgpathtools")},
                "resolved_work_dir": str(work), "private_input_required": False}
    atomic_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    # Prevent two builders replacing one another's matched-evaluation files.
    work = args.work_dir.expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    lock = work / ".prepare_data.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise SystemExit("Data preparation lock exists; verify no builder is running before removing it")
    try:
        os.close(descriptor)
        manifest = prepare(work)
        print(json.dumps({"complete": True, "identity_sha256": manifest["identity_sha256"],
                          "train": manifest["splits"]["train"]["count"], "dev": manifest["splits"]["dev"]["count"]}))
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
