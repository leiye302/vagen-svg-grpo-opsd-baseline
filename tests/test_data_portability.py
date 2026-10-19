"""Synthetic CPU tests: no downloads, original machine, pyarrow, or GPU needed."""
import importlib.util
import json
from pathlib import Path

from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


data = module("prepare_data")
assets = module("prepare_assets")


def svg(value):
    return f'<svg xmlns="http://www.w3.org/2000/svg"><rect id="{value}" width="1" height="1"/></svg>'


def row(value, filename=None):
    return {"Filename": filename or f"{value}.svg", "Svg": svg(value)}


def fake_renderer(code):
    # Different source strings can intentionally share visual identity.
    if 'id="broken"' in code:
        raise ValueError("invalid render")
    payload = code.replace('id="same-pixels"', 'id="heldout"')
    color = bytes.fromhex(data.digest(payload))[:3]
    return Image.new("RGB", (2, 2), tuple(color))


def fixture_rows():
    return {"train": [row(i) for i in range(30)] + [row("heldout"), row("same-pixels"), row("broken")],
            "val": [row("heldout")] + [row(f"val{i}") for i in range(6)],
            "test": [row("test0"), row("test1")]}


def selected():
    return data.select_tasks(fixture_rows(), fake_renderer, train_tasks=5, dev_tasks=3)


def test_deterministic_independent_rebuild():
    first, _, _, _ = selected()
    second, _, _, _ = selected()
    assert first == second
    assert data.identity_hash(first) == data.identity_hash(second)
    assert len(first) == 8


def test_original_task_id_formula():
    tasks, _, _, _ = selected()
    for task in tasks:
        value = task["source_split"] + ":" + task["source_filename"] + ":" + task["source_svg_sha256"]
        assert task["task_id"] == "svg_" + data.digest(value)[:20]


def test_heldout_source_and_pixel_duplicates_cannot_enter_train():
    rows = fixture_rows()
    rows["train"] = [row("heldout"), row("same-pixels"), row("test0"), row("unique")]
    tasks, _, exclusions, protection = data.select_tasks(rows, fake_renderer, train_tasks=1, dev_tasks=2)
    assert [t["source_filename"] for t in tasks if t["split"] == "train"] == ["unique.svg"]
    assert protection["test"]["rendered"] == 2
    assert protection["val"]["rendered"] == 7
    assert all(t["source_split"] != "test" for t in tasks)


def test_unsafe_svg_rejected_before_renderer():
    bad = '<svg xmlns="http://www.w3.org/2000/svg"><image href="https://invalid.example/a.png"/></svg>'
    rows = fixture_rows()
    rows["train"] = [{"Filename": "unsafe.svg", "Svg": bad}, row("unique")]
    seen = []
    def render(code):
        seen.append(code)
        assert code != bad
        return fake_renderer(code)
    tasks, _, _, _ = data.select_tasks(rows, render, train_tasks=1, dev_tasks=2)
    assert tasks[0]["source_filename"] == "unique.svg"


def test_insufficient_data_fails_not_resamples():
    rows = fixture_rows()
    rows["train"] = [row("heldout")]
    with pytest.raises(ValueError, match="Not enough"):
        data.select_tasks(rows, fake_renderer, train_tasks=1, dev_tasks=2)


def test_move_rebinds_paths_not_identities(tmp_path):
    tasks, _, _, _ = selected()
    old, new = tmp_path / "first machine", tmp_path / "second machine"
    first = data.build_rows(tasks, old)
    second = data.build_rows(tasks, new)
    for split in ("train", "dev"):
        for before, after in zip(first[split], second[split]):
            b, a = before["extra_info"], after["extra_info"]
            assert b["task_id"] == a["task_id"] and b["seed"] == a["seed"]
            assert a["env_config"]["data_dir"] == str(new.resolve() / "data/svg-icons-simple")
            assert a["env_config"]["data_dir"] != b["env_config"]["data_dir"]
    assert data.identity_hash(tasks) == data.identity_hash(tasks)


def test_test_split_rejected_from_training_rows(tmp_path):
    tasks, _, _, _ = selected()
    tasks[0]["split"] = "test"
    with pytest.raises(ValueError, match="train/dev"):
        data.build_rows(tasks, tmp_path)


def test_previous_identity_drift_is_blocked():
    tasks, _, _, _ = selected()
    previous = {"dataset_revision": data.DATA_REVISION, "seed": 2026,
                "source_file_sha256": {"train": "x"}, "identity_sha256": data.identity_hash(tasks)}
    data.check_previous_manifest(previous, tasks, {"train": "x"})
    tasks[0]["target_pixels_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Frozen selection"):
        data.check_previous_manifest(previous, tasks, {"train": "x"})


def test_hash_inventory_is_portable_and_detects_corruption(tmp_path):
    work = tmp_path / "work"
    directory = work / "models/test"
    directory.mkdir(parents=True)
    (directory / "config.json").write_text("{}", encoding="utf-8")
    rows = assets.inventory(work, directory)
    assert rows[0]["path"] == "models/test/config.json"
    assets.verify_inventory(work, rows)
    (directory / "config.json").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="modified"):
        assets.verify_inventory(work, rows)


def test_manifest_path_escape_rejected(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        assets.checked_relative(tmp_path / "work", "../../outside")


def test_hf_downloads_have_pinned_revisions_and_complete_metadata():
    lock = json.loads((ROOT / "sources.lock.json").read_text(encoding="utf-8"))
    specs = assets.hf_specs(lock)
    assert specs["dataset"]["directory"] == "data/svg-icons-simple"
    assert specs["dino"]["revision"] == "ed25f3a31f01632728cabb09d1542f84ab7b0056"
    assert "*.json" in specs["policy"]["patterns"]
    assert "*.safetensors" in specs["policy"]["patterns"]


def test_model_index_cannot_reference_outside_asset_root(tmp_path):
    for filename in ("config.json", "preprocessor_config.json"):
        (tmp_path / filename).write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "../secret"}}))
    with pytest.raises(ValueError, match="Incomplete"):
        assets.validate_hf_files("dino", tmp_path)


def test_data_only_manifest_verification_needs_no_models(tmp_path):
    lock = ROOT / "sources.lock.json"
    spec = assets.hf_specs(json.loads(lock.read_text(encoding="utf-8")))["dataset"]
    directory = tmp_path / spec["directory"]
    (directory / "data").mkdir(parents=True)
    for split in ("train", "val", "test"):
        (directory / "data" / f"{split}-00000-of-00001.parquet").write_bytes(b"fixture")
    manifest = {"sources_lock_sha256": assets.file_hash(lock), "complete": False,
                "assets": {"dataset": {key: value for key, value in spec.items() if key != "patterns"}},
                "files": assets.inventory(tmp_path, directory)}
    assets.atomic_json(tmp_path / "assets_manifest.json", manifest)
    assert assets.prepare(tmp_path, lock, verify_only=True, only_data=True) == manifest
    with pytest.raises(ValueError, match="network first"):
        assets.prepare(tmp_path, lock, verify_only=True)


def test_unversioned_dreamsim_source_is_not_silently_reused(tmp_path):
    (tmp_path / "facebookresearch_dino_main").mkdir()
    with pytest.raises(ValueError, match="Unversioned"):
        assets.prepare_dreamsim_source(tmp_path)


def test_runtime_cache_bytecode_is_not_an_asset(tmp_path):
    directory = tmp_path / "models/test"
    (directory / "__pycache__").mkdir(parents=True)
    (directory / "config.json").write_text("{}")
    (directory / "__pycache__" / "module.pyc").write_bytes(b"unrelated transient bytecode")
    assert [row["path"] for row in assets.inventory(tmp_path, directory)] == ["models/test/config.json"]
