# rt-2

A small, readable re-implementation of the RT-2 recipe (Brohan et al., 2023,
"RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control",
https://arxiv.org/abs/2307.15818), built module by module while reading the paper.
Clarity over performance.

RT-2's core claim: take a pre-trained VLM, express robot actions as *text tokens*,
and fine-tune on robot data **mixed with** the original web-scale vision-language
data ("co-fine-tuning"). The VLM's web knowledge then transfers to control —
better generalization to novel objects/instructions than robot-data-only training.

## Backbone

RT-2 weights are not public and the backbones (PaLI-X 55B, PaLM-E 12B) are huge.
We use **PaliGemma-3B** (`google/paligemma-3b-pt-224`) — the open descendant of
the PaLI family RT-2-PaLI-X was built on. Fits a single consumer/Colab GPU with
LoRA + bf16. Bonus: its vocab has 1024 reserved `<locXXXX>` tokens, so we can try
both of the paper's tokenization strategies (Sec 3.2): reserve clean tokens, or
overwrite the 256 least-frequent ones. (Gated on HF — accept the license at
https://huggingface.co/google/paligemma-3b-pt-224. Ungated fallback:
`HuggingFaceTB/SmolVLM2-2.2B-Instruct`; modules are written to be backbone-agnostic.)

## Modules (each maps to a paper section; see each module's README)

| module                   | paper section        | status |
|--------------------------|----------------------|--------|
| 01_action_tokenization   | Sec 3.2, Fig 3       | done   |
| 02_data                  | Sec 3.2, Fig 2       | todo   |
| 03_cofinetune            | Sec 3.2 ("co-fine-tuning"), ablation Sec 4.4 | todo |
| 04_constrained_decoding  | Sec 3.3              | todo   |
| 05_eval                  | Sec 4 (in miniature: LIBERO instead of real robots) | todo |
| 06_openvla_reference     | (not in paper; OpenVLA is the closest open reproduction) | optional |

Modules are standalone script folders, run from inside each folder. Later modules
import earlier ones with a two-line `sys.path` append — no packaging machinery.

## Setup — reuses mini-vla's environment

Everything heavy (torch, transformers, h5py, matplotlib, the LIBERO sim stack)
is already in `../mini-vla/.venv`, and module 05 needs that exact fragile
mujoco/robosuite pin set anyway. So:

```bash
source ../mini-vla/.venv/bin/activate
uv pip install -r requirements.txt   # additions only (peft, accelerate, ...)
```

Data: modules read the LIBERO demos already downloaded at
`../mini-vla/data/libero_spatial/*.hdf5`. Nothing is re-downloaded.

## Reading order

1. Read paper Sec 3.1–3.2 (action representation, Fig 3) → step through
   `01_action_tokenization/`.
2. Read Sec 3.2's data/prompt format (Fig 2) → `02_data/`.
3. Read the co-fine-tuning paragraphs of Sec 3.2 + ablation Sec 4.4 → `03_cofinetune/`.
4. Read Sec 3.3 (real-time inference, output constraint) → `04_constrained_decoding/`.
5. Read Sec 4 experiments → `05_eval/`.
