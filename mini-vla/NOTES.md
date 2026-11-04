# mini-vla build notes

One paragraph per piece, added after we've both confirmed it works and why.

## Step 0 — repo + environment + demo replay

What has been run so far is JUST the rendering of the simulation + execution of the 
pre-recorded action steps that LIBERO collected by human experimentation. There is NO
policy yet or VLA pipeline at all. 

## Step 1 - command embedding

I'll play around with a few methods of text-embeddings to see which is best for this dataset

## Step 2 - implementing a ViT

I'll choose a good ViT for this task (likely SigLIP or CLIP) 

## Step 3 - Data Projection

Implement either cross-attention or self-attention with a linear layer to bring to the 
latent space dimensions of the residual stream

# Step 4 - DiT

Initially, just use a pre-trained ViT (I won't pretrain for now). 

# Step 5 - GRPO 

This is what I really want to learn with this project (or at least one of the things I'm learning). I'll have to build the reward funcition, design the training harness, write out 
GRPO objective in NumPy and PyTorch. Will be good practice on fundamentals of the syntax as well as RL.

# Step 6 - Fine-tuning the DiT

Using just vanilla PyTorch and NumPy, I'll design the entire architecture for the DiT, intialize the weights stochastically with a simple Gaussian, and then fine-tune. Lwk will be hella fun.

# Step 7 - Dockerize and deploy

Just clean up the visuals and make some cool visualizations for the training process as well as how the denoising works during inference time. 
