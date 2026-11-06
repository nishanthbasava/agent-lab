import torch
from transformers import AutoImageProcessor, SiglipVisionModel

MODEL_ID = "google/siglip-base-patch16-224"


def pick_device():
    """Use the fastest hardware available: NVIDIA GPU > Apple GPU (MPS) > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class VisionEncoder(torch.nn.Module):
    def __init__(self, model_id=MODEL_ID, device=None):
        super().__init__()
        self.device = device or pick_device()

        # this processor just resized, rescales, and normalizes images
        # in the same format that SigLIP saw them during pretraining
        self.processor = AutoImageProcessor.from_pretrained(model_id)

        #this half is just the vision half of SigLIP
        self.model = SiglipVisionModel.from_pretrained(model_id).to(self.device)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad() # no need to track backward graph during inference
    def forward(self, images):
        '''
        images: list of (H, W, 3) uint8 numpy arrays (values 0-255).
        returns:
            tokens: (B, 196, 768) one vector per 16x16 patch
            pooled: (B, 768)      one vector per image
        '''
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        out = self.model(pixel_values=pixel_values)
        return out.last_hidden_state, out.pooler_output