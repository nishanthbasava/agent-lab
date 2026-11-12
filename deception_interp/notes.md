## OVERVIEW

Probe
- Small model trained on frozen activations to predict a label. Tests whether information is decodable.
- Yes, it is supervised.

Linear Probes:
- A probe that is just logistic regression. Its weight vector is a candidate feature direction.
- Yes, it is supervised.

Autoencoder: 
- Encoder and decoder trained to reconstruct their input through a hidden layer
- No, it is not supervised

SAE (Sparse Autoencoder):
- Autoencoder on activations with a wider hidden layer and a sparsity penalty. Decoder columns are feature directions
- No, it is not supervised

VAE (Variational Autoencoder):
- Autoencoder whose latent is a distribution, regularlized toward a prior. A generative model, rarely used in interp.
- No, it is not supervised

Ablation:
- Remove a direction or component and measure the behavior change. Tests causality, which probes can't



