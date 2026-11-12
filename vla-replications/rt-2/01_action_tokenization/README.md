# 01 — Action tokenization

**Paper: Sec 3.2 ("Robot-Action Fine-tuning"), Fig 3.** Background: RT-1
(Brohan et al., 2022) Sec 3.3, which RT-2 inherits the discretization from.

The whole trick of RT-2 is that actions become a *sentence*, so a VLM can emit
them with its ordinary text head. One timestep = 8 integers, each in [0, 255]:

```
terminate  Δx Δy Δz  Δroll Δpitch Δyaw  gripper
```

e.g. `"1 128 91 241 5 101 127 217"` (the paper's Fig 3 example). Each continuous
dimension is discretized into 256 **uniform** bins; the terminate flag just uses
bins 0/1 of the same scheme.

Those 8 integers then need to be *tokens the VLM already has*. Sec 3.2 gives two
strategies, both implemented here:

- **PaLI-X style**: the vocab already contains a dedicated token per small
  integer, so bins map onto those directly. PaliGemma's analog: its 1024
  reserved `<locXXXX>` tokens (we use `<loc0000>`–`<loc0255>`).
- **PaLM-E style**: no such tokens exist, so *overwrite the 256 least-frequent
  tokens*. In practice (and in OpenVLA, which reproduces this) that's the last
  256 ids of the vocab — rare subwords whose old meaning is sacrificed.

## Files

- `action_tokenizer.py` — `ActionBinner` (continuous ↔ bins) and
  `ActionTokenizer` (bins ↔ VLM token ids, both strategies).
- `test_roundtrip.py` — encode→decode round-trip assertions; run `python test_roundtrip.py`.
- `plot_quantization_error.py` — encodes/decodes every action in one real LIBERO
  demo file and plots the per-dimension error vs. the theoretical half-bin bound.
  Writes `quantization_error.png`.

## Things to notice while reading

- Decoding returns *bin centers*, so the irreducible error is ≤ half a bin width
  (≈ 0.0039 for a [-1, 1] dim). The plot confirms the error histogram lives
  strictly inside that bound — this is the precision the robot arm actually gets.
- 256 bins is one byte per dimension: coarse, but Sec 4 shows it's enough.
  RT-2 never regresses continuous values; everything stays in token-land so the
  VLM's pre-trained machinery is untouched (that's the point of the paper).
- LIBERO's action space is already the paper's: 7 dims = Δpos(3), Δrot(3),
  gripper, all in [-1, 1]. We add the terminate flag ourselves in module 02
  (1 on an episode's last step).
