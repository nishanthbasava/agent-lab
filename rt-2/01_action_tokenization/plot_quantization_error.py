"""How much precision does 256-bin discretization (Sec 3.2) actually cost?

Encodes/decodes every action in one real LIBERO demo file (reused from
../mini-vla) and histograms the per-dimension error against the theoretical
half-bin bound. Writes quantization_error.png next to this script.
"""

import pathlib

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from action_tokenizer import ActionBinner

DATA_DIR = pathlib.Path(__file__).parents[2] / "mini-vla/data/libero_spatial"
DIMS = ["Δx", "Δy", "Δz", "Δroll", "Δpitch", "Δyaw", "gripper"]

path = sorted(DATA_DIR.glob("*.hdf5"))[0]
with h5py.File(path, "r") as f:
    actions = np.concatenate(
        [f[f"data/{d}/actions"][:] for d in f["data"].keys()]
    ).astype(np.float32)
print(f"{path.name}: {len(actions)} action steps")

binner = ActionBinner()
recon = np.stack([binner.decode(binner.encode(a))[0] for a in actions])
err = recon - actions
half_bin = binner.bin_width / 2

fig, axes = plt.subplots(1, 7, figsize=(21, 3), sharey=True)
for i, ax in enumerate(axes):
    ax.hist(err[:, i], bins=60, color="tab:blue")
    # The error can never leave (-half_bin, +half_bin] — decode returns bin
    # centers. If a bar ever crossed the red lines, the binning would be broken.
    for s in (-1, 1):
        ax.axvline(s * half_bin[i], color="tab:red", lw=1)
    ax.set_title(f"{DIMS[i]}  (max |e| = {np.abs(err[:, i]).max():.4f})", fontsize=9)
    ax.set_xlim(-2.5 * half_bin[i], 2.5 * half_bin[i])
axes[0].set_ylabel("count")
fig.suptitle(
    f"Quantization error, 256 uniform bins on [-1, 1] — bound = ±{half_bin[0]:.4f} (red)"
)
fig.tight_layout()
out = pathlib.Path(__file__).parent / "quantization_error.png"
fig.savefig(out, dpi=120)
print(f"wrote {out}")
print(f"max |error| over all dims: {np.abs(err).max():.5f}  (bound {half_bin.max():.5f})")
