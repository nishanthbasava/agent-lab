import torch
import requests
from PIL import image

from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig #using this for quantization of ONLY WEIGHTS

bnb_config = BitsAndByesConfig(load_in_4bit=True)

#just using a pre-trained encoder
model = AutoModel.from_pretrained("google/siglip-base-patch16-224", quantization_config=bnb_config, device_map="auto", attn_implementation="sdpa")
#sdpa = scaled dot-product attention

#this processor is for turning the text directly into the tensors prior to encoding
#this is NOT a model
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")


# url = ... (get image path here), need to understand how the LIBERO HDF5 file structure is better for retrieval
# image = Image.open(requests.get(url, stream=True))


