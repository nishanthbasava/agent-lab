"""RT-2 action tokenization (Sec 3.2, Fig 3).

Two layers, kept separate because they answer different paper questions:

  ActionBinner     continuous action  <->  8 integer bins in [0, 255]
                   ("how do we discretize?" — inherited from RT-1 Sec 3.3)
  ActionTokenizer  bins  <->  VLM token ids
                   ("which existing tokens do the bins live on?" — RT-2 Sec 3.2)
"""

import numpy as np

N_BINS = 256  # one byte per dimension; Sec 3.2: "256 bins" per action variable


class ActionBinner:
    """Continuous 7-dim action + terminate flag <-> 8 integers in [0, N_BINS).

    Order matches the paper's Fig 3 action string:
        terminate, Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper
    """

    # LIBERO (like the paper's robots) emits normalized deltas in [-1, 1].
    DEFAULT_LOW = np.full(7, -1.0, dtype=np.float32)
    DEFAULT_HIGH = np.full(7, 1.0, dtype=np.float32)

    def __init__(self, low=None, high=None):
        # Per-dim bounds. Uniform bins between them — the paper bins uniformly
        # rather than by quantile, accepting wasted bins where data is sparse.
        self.low = np.asarray(self.DEFAULT_LOW if low is None else low, dtype=np.float32)
        self.high = np.asarray(self.DEFAULT_HIGH if high is None else high, dtype=np.float32)
        self.bin_width = (self.high - self.low) / N_BINS

    def encode(self, action, terminate=False):
        """(7,) float action + terminate -> (8,) int bins."""
        action = np.asarray(action, dtype=np.float32)
        bins = np.floor((action - self.low) / self.bin_width).astype(np.int64)
        # An action exactly at the high bound lands in bin N_BINS; clip keeps
        # everything representable (also silently clips out-of-range actions,
        # same as clipping before binning).
        bins = np.clip(bins, 0, N_BINS - 1)
        # Terminate is binary but rides in the same 256-way vocab as bin 0/1,
        # so the model needs exactly one output mechanism for all 8 positions.
        return np.concatenate([[int(terminate)], bins])

    def decode(self, bins):
        """(8,) int bins -> ((7,) float action, terminate bool).

        Returns bin *centers*: the quantization error is therefore bounded by
        half a bin width (~0.0039 on a [-1,1] dim) — see the plot script.
        """
        bins = np.asarray(bins)
        action = self.low + (bins[1:] + 0.5) * self.bin_width
        return action.astype(np.float32), bool(bins[0])


class ActionTokenizer:
    """Maps bins onto 256 token ids the VLM already has (Sec 3.2).

    The fine-tuning loss is just next-token prediction over these ids — no new
    output head, which is what lets co-fine-tuning share all weights with VQA.
    """

    def __init__(self, bin_token_ids):
        self.bin_token_ids = np.asarray(bin_token_ids, dtype=np.int64)
        assert len(self.bin_token_ids) == N_BINS
        self._id_to_bin = {int(t): b for b, t in enumerate(self.bin_token_ids)}

    # --- the paper's two strategies -----------------------------------------

    @classmethod
    def from_reserved_tokens(cls, hf_tokenizer, fmt="<loc{:04d}>"):
        """PaLI-X style: the vocab has dedicated tokens we can claim without
        destroying anything. PaliGemma ships 1024 <locXXXX> tokens (meant for
        detection boxes); we take the first 256 as action bins."""
        ids = [hf_tokenizer.convert_tokens_to_ids(fmt.format(b)) for b in range(N_BINS)]
        assert all(i is not None and i >= 0 for i in ids), "tokenizer lacks reserved tokens"
        return cls(ids)

    @classmethod
    def from_overwritten_tail(cls, vocab_size):
        """PaLM-E style: no spare tokens, so overwrite the 256 *least frequently
        used* ones. Like OpenVLA, we approximate 'least frequent' with the last
        256 vocab ids — BPE vocabs are roughly frequency-ordered, so the tail is
        rare subwords whose old meaning we sacrifice."""
        return cls(np.arange(vocab_size - N_BINS, vocab_size))

    # --- bins <-> ids --------------------------------------------------------

    def encode_to_ids(self, bins):
        return self.bin_token_ids[np.asarray(bins)]

    def decode_from_ids(self, ids):
        return np.array([self._id_to_bin[int(i)] for i in ids], dtype=np.int64)

    @property
    def valid_ids(self):
        """The only ids a robot-mode decode may emit — module 04 masks the
        softmax to this set (Sec 3.3: 'constrain ... to only valid action tokens')."""
        return self.bin_token_ids
