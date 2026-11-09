# rt-1

A small, readable PyTorch implementation of RT-1 (Brohan et al., 2022,
"RT-1: Robotics Transformer for Real-World Control at Scale",
https://arxiv.org/abs/2212.06817), scaled down to train on one GPU with LIBERO
data. Clarity over performance.

RT-1's recipe: turn each camera frame into a handful of language-conditioned
tokens, stack a short history of frames, run a small causal transformer over
them, and emit discretized actions as per-dimension classifications. (RT-2 in
`../rt-2/` keeps the action discretization but replaces this whole bespoke
tower with a pretrained VLM.)

## Files → paper (all in Sec 5.1 "Model" unless noted; Fig 1 is the overview)

| file                  | paper                                                       | status |
|-----------------------|-------------------------------------------------------------|--------|
| `film_efficientnet.py`| "Instruction and image tokenization": frozen sentence encoder + pretrained EfficientNet with identity-init FiLM | done |
| `token_learner.py`    | "TokenLearner" (Ryoo et al., 2021): spatial softmax-pooling down to 8 tokens | todo |
| `transformer.py`      | "Transformer": decoder-only, causal mask                    | todo |
| `action_tokenizer.py` | "Action tokenization": 256 uniform bins per dim (the scheme RT-2 later inherits — compare `../rt-2/01_action_tokenization/`) | todo |
| `model.py`            | assembles Fig 1 end to end                                  | todo |
| `shape_check.py`      | one forward pass on random inputs, prints every shape       | todo |
| `train.py`            | "Loss" (per-dim cross-entropy) on a small LIBERO subset (stands in for Sec 5.2's 130k-episode dataset) | todo |
| `eval.py`             | Sec 6 experiments in miniature: success rate in LIBERO sim  | todo |

## Shape walkthrough (ours vs paper)

```
                                    ours                      paper (B3, 300px)
instruction ─ frozen text enc ────► (B, 768)  SigLIP text     (B, 512)  USE
frames (B, 6, 3, 128, 128)
  └─ FiLM-EfficientNet-B0 ────────► (B·6, 320, 4, 4)          (B·6, 512, 9, 9)
  └─ 1×1 conv to token dim ───────► (B·6, 256, 4, 4)
  └─ flatten ─────────────────────► (B·6, 16, 256)            81 tokens
  └─ TokenLearner ────────────────► (B·6, 8, 256)             8 tokens
  └─ stack history + pos emb ─────► (B, 48, 256)              (B, 48, 512)
  └─ causal transformer, 4 layers ► (B, 48, 256)              8 layers
  └─ action head ─────────────────► (B, 7, 256) logits        11 dims × 256 bins
decode: argmax per dim → bin centers → LIBERO action [Δpos(3), Δrot(3), gripper]
```

## Reused from agent-lab

- **Environment**: `../mini-vla/.venv` (torch, transformers, h5py, the pinned
  LIBERO sim stack). Added for this project: `torchvision` (EfficientNet-B0) —
  see `requirements.txt`.
- **Language encoder**: the SigLIP checkpoint mini-vla already downloaded
  (`google/siglip-base-patch16-224`) — we use its *text* tower, frozen.
- **Data**: the LIBERO demos at `../mini-vla/data/libero_spatial/*.hdf5`, and
  `LiberoDataset`'s conventions (numeric demo sort, "index ends in 9 → val"
  split, lazy per-worker h5py handles) adapted in `train.py` — RT-1 needs raw
  actions (binning replaces z-normalization) and a 6-frame history window.
- **Sim eval**: the LIBERO clone at `../mini-vla/third_party/LIBERO`.

## Run

```bash
source ../mini-vla/.venv/bin/activate
python film_efficientnet.py   # each component has a smoke test under __main__
python shape_check.py         # full-model forward pass on random inputs
```
