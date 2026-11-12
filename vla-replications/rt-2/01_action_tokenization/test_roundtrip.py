"""Round-trip tests for the two tokenization layers. Run: python test_roundtrip.py

Checks the two properties the paper's recipe silently relies on:
  1. continuous -> bins -> continuous loses at most half a bin width per dim;
  2. bins -> token ids -> bins is *exact* (it's a pure relabeling).
"""

import numpy as np

from action_tokenizer import N_BINS, ActionBinner, ActionTokenizer

rng = np.random.default_rng(0)
binner = ActionBinner()

# --- 1. bins -> decode -> encode is exact (bin centers re-bin to themselves) --
bins = rng.integers(0, N_BINS, size=(1000, 7))
for b in bins:
    b8 = np.concatenate([[rng.integers(0, 2)], b])
    action, term = binner.decode(b8)
    assert np.array_equal(binner.encode(action, term), b8)
print("ok: bins -> continuous -> bins is exact")

# --- 2. continuous -> encode -> decode error is bounded by half a bin --------
actions = rng.uniform(-1, 1, size=(10_000, 7)).astype(np.float32)
half_bin = binner.bin_width / 2
worst = 0.0
for a in actions:
    recon, term = binner.decode(binner.encode(a, terminate=True))
    assert term is True
    err = np.abs(recon - a)
    assert np.all(err <= half_bin + 1e-6), (a, recon)
    worst = max(worst, err.max())
print(f"ok: |error| <= half bin width ({half_bin[0]:.5f}); worst seen {worst:.5f}")

# --- 3. boundary values stay representable -----------------------------------
for edge in (binner.low, binner.high, binner.high + 5.0):  # +5: out-of-range clips
    b8 = binner.encode(edge, terminate=False)
    assert b8[1:].min() >= 0 and b8[1:].max() <= N_BINS - 1
print("ok: bounds and out-of-range actions clip into valid bins")

# --- 4. bins <-> token ids is exact, for both of the paper's strategies ------
# PaLM-E style on a fake 257k vocab (PaliGemma's actual size):
tok_tail = ActionTokenizer.from_overwritten_tail(vocab_size=257_152)
b8 = binner.encode(rng.uniform(-1, 1, 7), terminate=True)
ids = tok_tail.encode_to_ids(b8)
assert np.array_equal(tok_tail.decode_from_ids(ids), b8)
assert ids.min() >= 257_152 - N_BINS  # really living in the vocab tail
print("ok: overwritten-tail strategy round-trips")

# PaLI-X / reserved-token style, with a stub standing in for PaliGemma's
# tokenizer so this test needs no model download:
class StubTokenizer:
    def convert_tokens_to_ids(self, t):  # <loc0000> -> 1000, <loc0001> -> 1001, ...
        return 1000 + int(t[4:8])

tok_loc = ActionTokenizer.from_reserved_tokens(StubTokenizer())
ids = tok_loc.encode_to_ids(b8)
assert np.array_equal(tok_loc.decode_from_ids(ids), b8)
assert len(set(tok_loc.valid_ids.tolist())) == N_BINS
print("ok: reserved-token strategy round-trips")

print("\nall round-trip tests passed")
