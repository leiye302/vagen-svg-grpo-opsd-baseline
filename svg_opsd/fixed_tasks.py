"""Frozen identity binding; no resampling validation icons at env.reset."""
import hashlib
import json
from pathlib import Path


def fixed_sample(dataset, config):
    index = config.fixed_source_index
    if index is None:
        raise ValueError("Matched SVG experiments require a fixed source index")
    sample = dataset[int(index)]
    filename = sample.get("Filename", sample.get("filename"))
    if str(filename) != str(config.fixed_source_filename):
        raise ValueError("Frozen SVG source identity mismatch")
    return sample


def verify_render_identity(image, config):
    expected = config.expected_target_pixels_sha256
    if not expected or hashlib.sha256(image.convert("RGB").tobytes()).hexdigest() != expected:
        raise ValueError("Frozen SVG target render differs from collected task")


def load_fixed_dataset(config):
    from datasets import Dataset
    source = Path(config.data_dir) / "data" / f"{config.split}-00000-of-00001.parquet"
    if config.split not in {"train", "val"}:
        raise ValueError("Periodic training/validation cannot consume official test")
    return Dataset.from_parquet(str(source))


def build_rows(taskbank, server_root):
    tasks = [json.loads(line) for line in Path(taskbank).read_text(encoding="utf-8").splitlines() if line]
    if len({task["task_id"] for task in tasks}) != len(tasks):
        raise ValueError("Duplicate task identity")
    output = {"train": [], "dev": []}
    for i, task in enumerate(tasks):
        split = task["split"]
        if split not in output or task["source_split"] != {"train": "train", "dev": "val"}[split]:
            raise ValueError("Unexpected train/dev split mapping")
        config = {
            "dataset_name": "starvector/svg-icons-simple",
            "data_dir": str(server_root) + "/data/svg-icons-simple",
            "split": task["source_split"], "prompt_format": "grounding_worldmodeling",
            "fixed_source_index": task["source_index"],
            "fixed_source_filename": str(task["source_filename"]),
            "frozen_task_id": task["task_id"],
            "expected_target_pixels_sha256": task["target_pixels_sha256"],
            "dino_weight": 0.0001, "dreamsim_weight": 5.0, "structural_weight": 0.0,
            "format_reward": 0.5, "format_penalty": 0.0,
        }
        output[split].append({"data_source": "svg", "prompt": [{"role": "user", "content": ""}],
            "extra_info": {"split": split, "task_id": task["task_id"], "env_name": "svg",
                           "env_config": config, "seed": 2026 + i * 1000}})
    if len(output["train"]) != 500 or len(output["dev"]) != 64:
        raise ValueError("Expected exactly frozen train500/dev64")
    return output

