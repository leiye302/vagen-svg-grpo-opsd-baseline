"""Current-Actor Teacher scoring via VAGEN's native image/token processing.

No policy generation takes place here. No reference worker is used. Each
decision gets a separate Teacher prefix, so a local hint for one triggered
decision cannot leak into a later, non-triggered Teacher decision.
"""
import contextlib
from pathlib import Path

STATIC_BEGIN = "[Teacher-only SVG skill]"
STATIC_END = "[End Teacher-only SVG skill]"
HINT_BEGIN = "[Actor-triggered SVG local hint]"
HINT_END = "[End Actor-triggered SVG local hint]"


def teacher_prompt(manager, recording, step, window_size=None,
                   is_final=False, prep_for_loss_mask=False):
    """Only called inside teacher_context; rollout uses the original function."""
    context = manager._svg_teacher_context
    if context is None or not is_final or not prep_for_loss_mask:
        raise RuntimeError("Teacher-only prompt attempted outside score construction")
    start = max(0, step - window_size) if window_size is not None else 0
    history = recording[start:step + 1]
    if len(history) != step + 1 - start:
        raise ValueError("Incomplete Teacher history")
    env_id = history[0]["env_id"]
    system = (manager.system_prompts[env_id] if hasattr(manager, "system_prompts")
              else manager.envs[env_id].system_prompt())
    chat = [{"role": "system", "content": system + "\n\n" + STATIC_BEGIN +
             "\n" + context["static_skill"] + "\n" + STATIC_END}]
    images, rewards = [], []
    for i, record in enumerate(history):
        if i > 0:
            response = manager._handle_special_tokens(
                record["info"]["llm_raw_response"], prep_for_loss_mask=True)
            chat.append({"role": "assistant", "content": response})
            rewards.append(record["reward"])
        if i < len(history) - 1:
            observation = record["obs_str"]
            if start + i == context["target_step"] and context["hint"]:
                observation += "\n\n" + HINT_BEGIN + "\n" + context["hint"] + "\n" + HINT_END
            chat.append({"role": "user", "content": observation})
            images.extend(record.get("image_data", []))
    prompt = manager.tokenizer.apply_chat_template(chat, add_generation_prompt=False,
                                                   tokenize=False)
    if not prompt.endswith("\n"):
        raise ValueError("Unexpected pinned Qwen chat template termination")
    prompt = prompt[:-1]
    end = manager.config.special_token_for_loss_mask[1]
    prompt = prompt.replace(end + manager.tokenizer.eos_token,
                            manager.tokenizer.eos_token + end)
    return {"prompt": prompt, "image_data": images, "rewards": rewards}


@contextlib.contextmanager
def teacher_context(manager, static_skill, target_step, hint):
    if getattr(manager, "_svg_teacher_context", None) is not None:
        raise RuntimeError("Nested Teacher context")
    if not static_skill.strip():
        raise ValueError("Missing shared SVG static skill")
    manager._svg_teacher_context = {"static_skill": static_skill,
                                    "target_step": target_step, "hint": hint}
    try:
        yield
    finally:
        manager._svg_teacher_context = None


def actor_segments(mask):
    """Absolute positions of each contiguous Actor span in a response array."""
    values = mask.tolist() if hasattr(mask, "tolist") else list(mask)
    segments, active = [], []
    for position, flag in enumerate(values):
        if flag:
            active.append(position)
        elif active:
            segments.append(active)
            active = []
    if active:
        segments.append(active)
    return segments


def assert_same_actor_tokens(student_ids, student_positions, teacher_ids, teacher_positions):
    left = [int(student_ids[i]) for i in student_positions]
    right = [int(teacher_ids[i]) for i in teacher_positions]
    if not left or left != right:
        raise RuntimeError("Teacher/Student sampled Actor token alignment mismatch")


def attach_teacher_scores(manager, student_batch, cfg):
    """Called after complete mini-rollout, before ANY Actor update or reordering.

    Uses native _generate_input_for_uptate for multimodal expansion and fresh
    3-axis RoPE. The public same-Actor compute_log_prob RPC is no-grad. Values
    are scattered back by exact sampled-token equality; environment positions
    stay zero and are excluded by the original Student loss_mask.
    """
    import torch
    from verl import DataProto
    from verl.utils.dataset.rl_dataset import collate_fn

    if manager.config.truncation != "error":
        raise ValueError("SVG OPSD requires explicit overflow error, not silent truncation")
    static_skill = Path(cfg.static_skill_file).read_text(encoding="utf-8").strip()
    student_ids = student_batch.batch["responses"]
    length = student_ids.shape[1]
    masks = student_batch.batch["loss_mask"][:, -length:]
    env_ids = list(manager.envs)
    if len(env_ids) != len(student_batch):
        raise RuntimeError("Trajectory ordering mismatch")
    teacher_rows, mapping = [], []
    for row_index, env_id in enumerate(env_ids):
        recording = manager.recorder[env_id]
        n_steps = manager.env_states[env_id]["step"]
        segments = actor_segments(masks[row_index])
        if len(segments) != n_steps:
            raise RuntimeError("Student mask must have exactly one Actor span per decision")
        if cfg.arm != "baseline":
            raise ValueError("This release is baseline-only; no external hint provider")
        hints = {}
        for target in range(n_steps):
            with teacher_context(manager, static_skill, target, hints.get(target)):
                row = manager._generate_input_for_uptate(recording, target + 1, window_size=None)
            tlength = len(row["responses"])
            tspans = actor_segments(row["loss_mask"][-tlength:])
            if len(tspans) != target + 1:
                raise RuntimeError("Teacher mask/history mismatch")
            assert_same_actor_tokens(student_ids[row_index], segments[target],
                                     row["responses"], tspans[-1])
            # Preserve actual pixels and image grid; never construct text-only Teacher.
            expected_images = sum(len(r.get("image_data", [])) for r in recording[:target + 1])
            if expected_images and ("multi_modal_inputs" not in row or
                    row["multi_modal_inputs"]["image_grid_thw"].shape[0] != expected_images):
                raise RuntimeError("Teacher lost or reordered image inputs")
            teacher_rows.append(row)
            mapping.append((row_index, segments[target], tspans[-1]))
    scores = torch.zeros_like(student_ids, dtype=torch.float32)
    # Score in bounded batches. All batches use identical current Actor weights,
    # since the caller cannot enter update_actor until this function returns.
    score_batch_size = int(cfg.get("teacher_score_batch_size", 8))
    world_size = int(manager.actor_rollout_wg.world_size)
    for start in range(0, len(teacher_rows), score_batch_size):
        rows = teacher_rows[start:start + score_batch_size]
        count = len(rows)
        while len(rows) % world_size:
            rows.append(rows[-1])
        teacher_batch = DataProto.from_single_dict(collate_fn(rows))
        teacher_batch.meta_info.update(student_batch.meta_info)
        with torch.no_grad():
            output = manager.actor_rollout_wg.compute_log_prob(teacher_batch)
        lp = output.batch["old_log_probs"].detach().cpu()
        for local in range(count):
            destination, spos, tpos = mapping[start + local]
            if not torch.isfinite(lp[local, tpos]).all():
                raise RuntimeError("Nonfinite Teacher log probabilities")
            scores[destination, spos] = lp[local, tpos]
    student_batch.batch["teacher_log_probs"] = scores.to(student_ids.device)
    return student_batch

