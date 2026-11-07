import json
import pathlib

import h5py
import numpy as np
import torch


def list_demos(f):
    """Demo keys in numeric order. Plain sorted() gives demo_0, demo_1, demo_10, demo_11, ...
    because it compares strings character by character; we sort by the integer instead."""
    return sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))


def is_train_demo(demo_key):
    """Hold out demos whose index ends in 9 (5 of every 50) for validation."""
    return int(demo_key.split("_")[1]) % 10 != 9


class LiberoDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, stats_path, split="train", chunk_size=16):
        assert split in ("train", "val")
        self.files = sorted(pathlib.Path(data_dir).glob("*.hdf5"))
        assert self.files, f"no .hdf5 files found in {data_dir}"
        self.chunk_size = chunk_size

        # Normalization stats (computed by scripts/compute_stats.py on train demos only)
        stats = json.loads(pathlib.Path(stats_path).read_text())
        self.a_mean = np.array(stats["action_mean"], dtype=np.float32)
        self.a_std = np.array(stats["action_std"], dtype=np.float32)
        self.p_mean = np.array(stats["proprio_mean"], dtype=np.float32)
        self.p_std = np.array(stats["proprio_std"], dtype=np.float32)

        # Build a flat index: sample i -> (file index, demo key, timestep).
        # Only metadata is read here; images stay on disk until __getitem__.
        self.instructions = []
        self.index = []
        for f, path in enumerate(self.files):
            with h5py.File(path, "r") as f:
                info = json.loads(f["data"].attrs["problem_info"])
                self.instructions.append(info["language_instruction"])
                for d in list_demos(f):
                    if is_train_demo(d) != (split == "train"):
                        continue
                    T = f[f"data/{d}/actions"].shape[0]
                    self.index += [(fi, d, t) for t in range(T)]

        self.handles = None  # opened lazily, once per DataLoader worker

    def __len__(self):
        return len(self.index)

    def _file(self, fi):
        # h5py file handles can't be shared across processes, so each worker
        # opens its own the first time it needs a file, then reuses it.
        if self.handles is None:
            self.handles = {}
        if fi not in self.handles:
            self.handles[fi] = h5py.File(self.files[fi], "r")
        return self.handles[fi]

    def __getitem__(self, i):
        fi, demo, t = self.index[i]
        g = self._file(fi)[f"data/{demo}"]

        # --- observations at time t ---
        agent = g["obs/agentview_rgb"][t][::-1].copy()    # (128, 128, 3) uint8
        wrist = g["obs/eye_in_hand_rgb"][t][::-1].copy()  # (128, 128, 3) uint8
        proprio = np.concatenate(
            [g["obs/joint_states"][t], g["obs/gripper_states"][t]]
        ).astype(np.float32)                              # (9,)

        # --- action chunk a_t ... a_{t+H-1} ---
        actions = g["actions"]
        T = actions.shape[0]
        chunk = actions[t:min(t + self.chunk_size, T)].astype(np.float32)  # (n_real, 7)
        n_real = len(chunk)
        n_pad = self.chunk_size - n_real
        if n_pad > 0:
            pad = np.zeros((n_pad, 7), dtype=np.float32)  # zero motion...
            pad[:, 6] = chunk[-1, 6]                      # ...gripper holds its last command
            chunk = np.concatenate([chunk, pad])
        mask = (np.arange(self.chunk_size) < n_real).astype(np.float32)  # 1 real, 0 padded

        return {
            "agentview": torch.from_numpy(agent),
            "wrist": torch.from_numpy(wrist),
            "proprio": torch.from_numpy((proprio - self.p_mean) / self.p_std),
            "actions": torch.from_numpy((chunk - self.a_mean) / self.a_std),  # (H, 7)
            "action_mask": torch.from_numpy(mask),                            # (H,)
            "instruction": self.instructions[fi],
            "task_id": fi,
        }

    def unnormalize_actions(self, a):
        """Model output (normalized, numpy) -> real robot actions. Used at rollout time."""
        return a * self.a_std + self.a_mean