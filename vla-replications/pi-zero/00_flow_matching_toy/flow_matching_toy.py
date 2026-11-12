"""Conditional flow matching on two moons — π0's objective (Sec IV), minus robots.

CONVENTIONS — fixed here, reused by every later stage of pi-zero:

    t = 0  → DATA        t = 1 → NOISE          (openpi / LeRobot direction)
    x_t  = (1 - t) * x_data + t * eps,           eps ~ N(0, I)
    v*   = dx_t/dt = eps - x_data                (the regression target)
    sampling: start at x = eps (t = 1), Euler  x <- x + dt * v_theta(x, t)
              with dt = -1/K, down to t = 0.

Mapping to the paper: π0 writes A^τ = τ·A + (1-τ)·ε, so paper-τ = 1 - our-t
(its τ=0 is pure noise, τ=1 is data). Its printed target u = ε - A is the SAME
vector we regress here — and that's the trap: u points from data TOWARD noise,
so stepping "forward in paper-τ" with +δ·u walks back to noise. openpi resolves
it by flipping time to our convention and stepping with a negative dt; LeRobot
does the same. Pick one direction, write it down, never mix the two.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def two_moons(n, noise=0.07, rng=None):
    """Classic two-moons, standardized so N(0,I) noise covers the data —
    flow matching transports one distribution to the other, so they should
    occupy the same region of the plane."""
    rng = rng or np.random.default_rng(0)
    a = rng.uniform(0, np.pi, n // 2)
    b = rng.uniform(0, np.pi, n - n // 2)
    pts = np.concatenate([
        np.stack([np.cos(a), np.sin(a)], 1),              # upper moon
        np.stack([1 - np.cos(b), 0.5 - np.sin(b)], 1),    # lower moon
    ]) + rng.normal(0, noise, (n, 2))
    pts = rng.permutation(pts)  # interleave the moons so any slice shows both
    return ((pts - pts.mean(0)) / pts.std(0)).astype(np.float32)


class VelocityField(nn.Module):
    """v_theta(x, t): the whole 'model' at this scale. In π0 this role is
    played by the action expert reading the VLM prefix; the Fourier features
    here foreshadow stage 04's sinusoidal timestep embedding."""

    def __init__(self, dim=2, hidden=256, freqs=(1.0, 2.0, 4.0, 8.0)):
        super().__init__()
        self.freqs = torch.tensor(freqs)
        self.net = nn.Sequential(
            nn.Linear(dim + 1 + 2 * len(freqs), hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x, t):
        t = t.reshape(-1, 1).expand(x.shape[0], 1)
        ang = 2 * torch.pi * t * self.freqs
        return self.net(torch.cat([x, t, ang.sin(), ang.cos()], dim=1))


def train(model, data, steps=8000, batch=512, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    data = torch.from_numpy(data)
    for step in range(steps):
        x0 = data[torch.randint(len(data), (batch,))]   # x_data  (t=0 end)
        eps = torch.randn_like(x0)                       # noise   (t=1 end)
        t = torch.rand(batch, 1)                         # uniform; π0's shifted Beta comes in stage 04
        xt = (1 - t) * x0 + t * eps
        loss = ((model(xt, t) - (eps - x0)) ** 2).mean() # regress v* = eps - x_data
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0 or step == steps - 1:
            print(f"step {step:4d}  loss {loss.item():.4f}")


@torch.no_grad()
def sample(model, n=2000, K=10, seed=1):
    """Euler integration t: 1 -> 0. Returns snapshots [(t, points)] so the plot
    can show the flow mid-transport. K=10 matches π0's control-time setting."""
    torch.manual_seed(seed)
    x = torch.randn(n, 2)
    dt = -1.0 / K
    snaps = [(1.0, x.clone().numpy())]
    for k in range(K):
        t = torch.tensor(1.0 + k * dt)
        x = x + dt * model(x, t)
        snaps.append((round(1.0 + (k + 1) * dt, 3), x.clone().numpy()))
    return snaps


if __name__ == "__main__":
    torch.manual_seed(0)
    data = two_moons(4000)
    model = VelocityField()
    train(model, data)

    snaps = dict(sample(model, K=10))
    show_ts = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]

    fig, axes = plt.subplots(1, len(show_ts) + 1, figsize=(3 * (len(show_ts) + 1), 3))
    panels = [(f"t = {t}" + ("  (noise)" if t == 1.0 else "  (samples)" if t == 0.0 else ""),
               snaps[t], "#4477AA") for t in show_ts]
    panels.append(("two moons (data)", data[:2000], "#888888"))
    for ax, (title, pts, color) in zip(axes, panels):
        ax.scatter(pts[:, 0], pts[:, 1], s=3, alpha=0.5, color=color, linewidths=0)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-2.8, 2.8); ax.set_ylim(-2.8, 2.8)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#cccccc")
    fig.suptitle("Flow matching: Euler integration from noise (t=1) to data (t=0), K=10 steps")
    fig.tight_layout()
    out = __file__.replace("flow_matching_toy.py", "flow_samples.png")
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
