"""Step 0 sanity check: replay a recorded LIBERO demo open-loop.

Loads one demo from an HDF5 file, resets the sim to the demo's exact initial
physics state, then feeds ONLY the recorded actions back through env.step().
If the (env, action space, demo) triple is coherent, the sim re-enacts the
demo and the task succeeds. This is the contract every later training step
depends on: actions in the dataset == actions the sim executes.

Usage: python scripts/replay_demo.py data/libero_spatial/<task>_demo.hdf5
"""
import argparse
import json
import pathlib

import h5py
import imageio
import numpy as np
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hdf5", type=pathlib.Path)
    ap.add_argument("--demo", type=int, default=0)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("data/replay.gif"))
    args = ap.parse_args()

    with h5py.File(args.hdf5, "r") as f:
        data = f["data"]
        # The stored bddl path is absolute on the dataset author's machine;
        # keep only the part after "bddl_files/" and re-root it locally.
        bddl_rel = data.attrs["bddl_file_name"].split("bddl_files/")[-1]
        instruction = json.loads(data.attrs["problem_info"])["language_instruction"]
        demo = data[f"demo_{args.demo}"]
        actions = demo["actions"][:]
        init_state = demo["states"][0]

    # action = [delta_ee_pos(3), delta_ee_ori(3), gripper(1)], OSC controller
    assert actions.ndim == 2 and actions.shape[1] == 7, actions.shape
    assert np.abs(actions[:, :6]).max() <= 1.0 + 1e-6, "pose deltas should be in [-1, 1]"
    assert set(np.round(actions[:, 6])) <= {-1.0, 0.0, 1.0}, "gripper is -1/0/1"

    bddl = pathlib.Path(get_libero_path("bddl_files")) / bddl_rel
    assert bddl.exists(), f"missing bddl file: {bddl}"
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.seed(0)
    env.reset()
    env.set_init_state(init_state)

    frames, success = [], False
    for a in actions:
        obs, _reward, _done, _info = env.step(a)
        frames.append(obs["agentview_image"][::-1])  # robosuite renders upside down
        success = success or env.check_success()
    env.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out, frames, fps=20)
    print(f"instruction : {instruction!r}")
    print(f"steps       : {len(frames)}  success: {success}")
    print(f"gif         : {args.out}")
    assert success, "replay failed — env/action mismatch, do NOT proceed to training"


if __name__ == "__main__":
    main()
