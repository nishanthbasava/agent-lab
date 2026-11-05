import h5py, json 

class LiberoDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path = path
        with h5py.File(path, "r") as f:
            self.instruction = json.loads(f["data"].attrs["problem_info"])["language_instruction"]
            # flat index: sample i → (which demo, which timestep)
            self.index = [(d, t) for d in f["data"].keys()
                                 for t in range(f[f"data/{d}/actions"].shape[0])]
        self.f = None  # open lazily per worker, h5py handles don't fork safely

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        if self.f is None:
            self.f = h5py.File(self.path, "r")
        demo, t = self.index[i]
        g = self.f[f"data/{demo}"]
        img    = g["obs/agentview_rgb"][t][::-1]   # flip: opengl convention
        proprio = np.concatenate([g["obs/joint_states"][t], g["obs/gripper_states"][t]])
        action = g["actions"][t]
        return img, proprio, self.instruction, action