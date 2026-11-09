"""Instruction and image tokenization (Sec 5.1, Fig 1 left half).

RT-1 conditions vision on language *early* — inside the CNN, via FiLM — rather
than concatenating a text embedding later. The claim (Sec 5.1): this lets the
network extract task-relevant image features from the start, instead of first
computing a generic image summary and reconciling it with the instruction later.
"""

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from transformers import AutoTokenizer, SiglipTextModel

# SUBSTITUTION: the paper embeds instructions with Universal Sentence Encoder
# (512-d, frozen). USE is TensorFlow-only, so we use the text tower of the
# SigLIP checkpoint mini-vla already downloaded (768-d, also frozen). Any
# frozen sentence embedding works here — it's only ever read through the
# zero-initialized FiLM linears below, which adapt to whatever width they get.
TEXT_MODEL = "google/siglip-base-patch16-224"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def preprocess(images_uint8):
    """(B, H, W, 3) uint8 -> (B, 3, H, W) float, ImageNet-normalized —
    the statistics EfficientNet's pretrained weights expect."""
    x = torch.as_tensor(images_uint8).float().permute(0, 3, 1, 2) / 255.0
    return (x - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]


class InstructionEncoder(nn.Module):
    """Frozen sentence encoder: list of strings -> (B, 768).

    Frozen as in the paper — the instruction embedding is a fixed conditioning
    signal; all language *adaptation* happens in the trainable FiLM layers."""

    def __init__(self, model_id=TEXT_MODEL):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = SiglipTextModel.from_pretrained(model_id)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.dim = self.model.config.hidden_size  # 768

    @torch.no_grad()
    def forward(self, instructions):
        batch = self.tokenizer(instructions, padding="max_length", return_tensors="pt")
        batch = {k: v.to(self.model.device) for k, v in batch.items()}
        return self.model(**batch).pooler_output  # (B, 768)


class FiLM(nn.Module):
    """Feature-wise affine transform: x * (1 + gamma(ctx)) + beta(ctx).

    gamma/beta come from zero-initialized linears, so at step 0 this is exactly
    the identity — the paper's trick (Sec 5.1) for inserting language
    conditioning into a pretrained CNN without disrupting its ImageNet features."""

    def __init__(self, ctx_dim, channels):
        super().__init__()
        self.to_gamma_beta = nn.Linear(ctx_dim, 2 * channels)
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)

    def forward(self, x, ctx):
        gamma, beta = self.to_gamma_beta(ctx)[:, :, None, None].chunk(2, dim=1)
        return x * (1 + gamma) + beta


class FiLMEfficientNet(nn.Module):
    """ImageNet-pretrained EfficientNet-B0 with a FiLM layer after every MBConv
    block (the paper does the same to B3's 26 blocks; B0 has 16).

    The whole backbone stays trainable — the paper fine-tunes it end-to-end;
    only the *initialization* is protected, by FiLM's identity init.

    (B, 3, 128, 128) images + (B, ctx_dim) instruction -> (B, 16, token_dim).
    The paper's B3 at 300px ends at a 9x9 map = 81 tokens; B0 at LIBERO's 128px
    ends at 4x4 = 16. We stop before B0's 1280-channel head conv (built for
    ImageNet classification, not needed) and use a 1x1 conv to set token width."""

    def __init__(self, ctx_dim=768, token_dim=256, pretrained=True):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        net = efficientnet_b0(weights=weights)
        self.stem = net.features[0]
        # features[1:-1] are the 7 MBConv stages; flatten to 16 blocks so each
        # gets its own FiLM, mirroring the paper's per-block insertion.
        self.blocks = nn.ModuleList(b for stage in net.features[1:-1] for b in stage)

        # Each block's output width varies (16..320 channels); discover them
        # with one dummy pass instead of hardcoding B0's channel table.
        with torch.no_grad():
            x = self.stem(torch.zeros(1, 3, 128, 128))
            channels = []
            for block in self.blocks:
                x = block(x)
                channels.append(x.shape[1])
        self.films = nn.ModuleList(FiLM(ctx_dim, c) for c in channels)
        self.project = nn.Conv2d(channels[-1], token_dim, kernel_size=1)

    def forward(self, images, ctx):
        x = self.stem(images)
        for block, film in zip(self.blocks, self.films):
            x = film(block(x), ctx)
        x = self.project(x)                  # (B, token_dim, 4, 4)
        return x.flatten(2).transpose(1, 2)  # (B, 16, token_dim) — 16 vision tokens


if __name__ == "__main__":
    torch.manual_seed(0)

    enc = InstructionEncoder()
    ctx = enc(["pick up the black bowl and place it on the plate"])
    print("instruction embedding:", tuple(ctx.shape))
    assert ctx.shape == (1, 768)

    net = FiLMEfficientNet().eval()  # eval: stochastic depth off for the comparison
    images = preprocess(torch.randint(0, 256, (2, 128, 128, 3), dtype=torch.uint8))
    ctx2 = ctx.expand(2, -1)
    tokens = net(images, ctx2)
    print("vision tokens:", tuple(tokens.shape))
    assert tokens.shape == (2, 16, 256)

    # Identity-init check: with gamma=beta=0, the FiLM'd backbone must produce
    # exactly the plain pretrained backbone's features (Sec 5.1's whole point).
    with torch.no_grad():
        x = net.stem(images)
        for block in net.blocks:
            x = block(x)
        plain = net.project(x).flatten(2).transpose(1, 2)
    assert torch.allclose(tokens, plain), "FiLM init is not identity!"
    print("ok: FiLM-at-init == plain pretrained EfficientNet")
