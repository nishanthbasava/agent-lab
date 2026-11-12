# pi-zero

A small, readable PyTorch implementation of π0 (Black et al., 2024,
"π0: A Vision-Language-Action Flow Model for General Robot Control",
https://arxiv.org/abs/2410.24164), built stage by stage while reading the paper.
Clarity over performance; trains at small scale on one GPU.

π0's recipe: a pretrained VLM (PaliGemma-3B) encodes images + instruction into
prefix tokens; a separate, smaller **action expert** (its own transformer
weights sharing the same attention layers, mixture-of-transformers style)
carries proprio-state and a noisy action chunk; the expert is trained with
**conditional flow matching** to output the velocity that denoises the chunk.
Inference = ~10 Euler steps, re-running only the expert against cached prefix KV.

## Stages → paper (Sec IV "The π0 Model" + Fig 3 unless noted)

| stage                   | paper                                                      | status |
|-------------------------|------------------------------------------------------------|--------|
| 00_flow_matching_toy    | the flow-matching objective of Sec IV, alone on a 2D toy — conventions fixed here and used everywhere after | done |
| 01_backbone             | PaliGemma-3B prefix (Sec IV; π0's actual VLM)              | todo |
| 02_action_expert        | separate expert weights sharing attention with the VLM (Sec IV, "action expert"; Fig 3's two colors) | todo |
| 03_attention_mask       | block-wise causal mask: [images+text] [state] [actions] (Sec IV / Appendix B) + visualization script | todo |
| 04_flow_matching_head   | chunk of H actions, timestep fusion, shifted-Beta τ sampling, velocity MSE (Sec IV, Eq. for L^τ) | todo |
| 05_train / 06_eval      | small LIBERO subset; 10-step Euler inference with prefix KV cache; success rate (Sec VI, in miniature) | todo |
| 07_reference (optional) | compare shapes/masks/outputs against openpi/LeRobot π0 weights | todo |

## Conventions (fixed in stage 00, used everywhere)

We use the **openpi/LeRobot direction**: `t = 0` is data, `t = 1` is noise,
`x_t = (1−t)·x + t·ε`, velocity target `v* = ε − x`, and sampling integrates
from t=1 down to t=0 with Euler steps `dt = −1/K`. The paper's τ runs the other
way (τ = 1 − t; its A^τ = τ·A + (1−τ)·ε has τ=0 as pure noise) — see the long
comment in `00_flow_matching_toy/flow_matching_toy.py` for the exact mapping
and the sign trap.

## Shape walkthrough (planned; updated with real numbers as stages land)

```
images (B, 2, 3, 224, 224) ─ SigLIP ─► 2×256 tokens ┐
instruction ─ tokenizer ─► ~16 tokens ──────────────┴► prefix (B, ~528, 2048)  PaliGemma width
proprio state (B, 9) ─ linear ─► (B, 1, d_expert)   ┐
noisy chunk (B, 16, 7) + t-embed ─► (B, 16, d_expert)┴► suffix (B, 17, 512)    action-expert width
            └── the two streams share attention (block-wise mask) but not weights
action expert output ─► velocity (B, 16, 7)         paper: H=50, expert ~300M; ours: H=16, d_expert=512
```

## Reused from agent-lab

- **Environment**: `../mini-vla/.venv` (stage 00 needs nothing new; stages 01+
  add `peft`, `accelerate` — see `requirements.txt`).
- **Data**: `../mini-vla/data/libero_spatial/*.hdf5`, and mini-vla's
  `LiberoDataset` nearly unchanged — it already returns 16-step action chunks,
  proprio, and both cameras (π0's exact diet); only its z-score normalization
  gets swapped for π0-style quantile scaling. H=16 matches its `chunk_size`.
- **Sim eval**: `../mini-vla/third_party/LIBERO`.
- **Backbone**: PaliGemma-3B — same gated HF checkpoint discussed in
  `../rt-2/README.md` (accept license there); SmolVLM2 as ungated fallback.

## Run

```bash
source ../mini-vla/.venv/bin/activate
python 00_flow_matching_toy/flow_matching_toy.py   # trains in ~1 min on CPU/MPS
```
