# 00 — Flow matching on a 2D toy

**Paper: Sec IV ("The π0 Model") — the conditional flow-matching objective,
isolated from everything robotic.** In π0 the thing being denoised is an action
chunk `A ∈ R^{H×d}` conditioned on a huge multimodal prefix; here it is a 2D
point from two moons conditioned on nothing. Same loss, same sampler, two
dimensions instead of 16×7 — so you can *see* the probability flow.

## Files

- `flow_matching_toy.py` — tiny MLP velocity field, training loop, Euler
  sampler, and the convention block-comment every later stage refers back to.
  Run it (`~1 min, CPU`); writes `flow_samples.png`.

## The convention (read the comment in the file)

`t=0` = data, `t=1` = noise, `x_t = (1−t)·x + t·ε`, regress `v* = ε − x`,
sample with Euler steps `dt = −1/K` from t=1 to t=0. This is the
openpi/LeRobot direction; the paper's τ runs opposite (τ = 1 − t). The file's
comment spells out why the paper's printed target `u = ε − A` is the *same
vector* we regress, and where the sign trap lies.

## Things to notice while reading

- The regression target `ε − x` is **constant along each interpolation path** —
  the network learns a velocity field whose integral curves transport noise to
  data. That's all "flow matching" is at this scale.
- K=10 Euler steps already yields clean moons — the same K π0 uses at control
  time (Sec IV / inference). Diffusion-style samplers need many more; this is
  one reason π0 can run chunks at 50Hz.
- **The training loss plateaus around ~1.4 and never approaches zero — that is
  correct, not underfitting.** Many (noise, data) pairs pass through the same
  `x_t`, each demanding a different `ε − x`; the best any network can do is
  their conditional mean. The irreducible variance stays in the loss, while the
  learned *mean* field is exactly the marginal velocity that transports the
  distributions. Watch the plot, not the loss.
- Training samples `t` uniformly here. π0 instead biases `t` toward the noisy
  end with a shifted Beta (Sec IV) — that arrives in stage 04, where it matters
  because high-noise timesteps dominate what the robot actually executes from.
