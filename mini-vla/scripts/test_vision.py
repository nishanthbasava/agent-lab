# Testing out that the frozen SigLIP actually works on ONE LIBERO frame
import pathlib
import sys
import time

# Make `import minivla` work when running `python scripts/test_vision.py`
# from the mini-vla folder: add the repo root to Python's import path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import h5py
import imageio
import torch

from minivla.vision import VisionEncoder

PATH = "data/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5"


# STEP 1, loading one frame from each camera
with h5py.File(PATH, "r") as f:
    g = f["data/demo_0/obs"]

    # [::-1] flips vertically (sim renders upside down). .copy() makes the array
    # contiguous in memory again, which torch needs
    agent = g["agentview_rgb"][0][::-1].copy()
    wrist = g["eye_in_hand_rgb"][0][::-1].copy()
    all_agent = g["agentview_rgb"][:32][:, ::-1].copy() # 32 frames, for timing

print("frame:", agent.shape, agent.dtype, "min", agent.min(), "max", agent.max())

# Save to look at them later
imageio.imwrite("data/frame_agent.png", agent)
imageio.imwrite("data/frame_wrist.png", wrist)

# 2. Load the encoder itself
enc = VisionEncoder()
n_total = sum(p.numel() for p in enc.model.parameters())
n_train = sum(p.numel() for p in enc.model.parameters() if p.requires_grad)
print(f"device: {enc.device} | params: {n_total/1e6:.1f}M | trainable: {n_train}")


# 3. Encode
tokens, pooled = enc([agent, wrist])
print("patch tokens:", tuple(tokens.shape))  # expect (2, 196, 768)
print("pooled:      ", tuple(pooled.shape))  # expect (2, 768)

# 4. Sanity Checks
# a. running the same image twice
t2, p2 = enc([agent])
print("deterministic:", torch.allclose(pooled[0], p2[0], atol=1e-4))

# b. two different cameras -> different embeddings (cosine sim well below 1).
cos = torch.nn.functional.cosine_similarity(pooled[0], pooled[1], dim=0).item()
print(f"cos sim agentview vs wrist: {cos:.3f}")

# c. two consecutive frames of the same camera -> very similar (close to 1).
with h5py.File(PATH, "r") as f:
    nxt = f["data/demo_0/obs/agentview_rgb"][1][::-1].copy()
_, p_next = enc([nxt])
cos_t = torch.nn.functional.cosine_similarity(pooled[0], p_next[0], dim=0).item()
print(f"cos sim frame 0 vs frame 1: {cos_t:.3f}")

# 5. Speed: how long will caching the whole dataset take
start = time.time()
enc(list(all_agent))
if enc.device == "mps":
    torch.mps.synchronize()  # GPU work is async; wait for it before stopping the clock
dt = time.time() - start
print(f"32 frames in {dt:.2f}s -> {32/dt:.0f} frames/sec")