# Attribution and scope

This repository is a baseline integration, not an official VAGEN or SDAR release.

- **VAGEN** supplies the SVG visual environment, image feedback, multimodal
  rollout, and trajectory-GRPO trainer. MIT license: `licenses/VAGEN.txt`.
- **JamesKrW/verl** is the pinned legacy VAGEN training dependency, derived
  from veRL. Its Apache-2.0 license and notices are retained in `licenses/`.
- **SDAR** supplies the clipped policy objective (including dual clipping),
  loss aggregation, and detached-sigmoid-gate sampled-token OPSD functions.
  Apache-2.0 license: `licenses/SDAR.txt`.

Exact public URLs and revisions are recorded in `sources.lock.json`.
`overlays/` contains modified versions of the corresponding upstream files;
`overlays/manifest.json` records both original and replacement SHA-256 values.
The changes connect Teacher-only scoring to the actual Actor backward pass,
bind deterministic task identity, apply an explicit response/safety protocol,
preserve generated special tokens, and fix update counting. Runtime-only
compatibility and checkpoint fixes are separately recorded by
`scripts/apply_runtime_fixes.py`.

New integration code is distributed under Apache-2.0. Original copyright
notices and license obligations remain applicable to derived files. Model
weights and image datasets are downloaded separately from their public
publishers and remain subject to their respective licenses; this repository
does not redistribute them.
