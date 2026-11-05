import torch
import requests
from PIL import image
import h5py, json #HDF5 is LIBERO data format, it's Hierarchical Data Format, version 5
# container for storing large numerical arrays together with labels describing them 
# look into robomimic later

f = h5py.File("data/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5", "r")
# demos -> obs, actions, rewards (just values, not the reward function itself)

# text command (once per file) 
instruction = json.loads(f["data"].attrs["problem_info"])["language_instruction"]

#handle to a dataset - NO data read yet, it's lazy
imgs = f["data/demo_0/obs/agentview_rgb"]  # <HDF5 dataset, shape (98, 128, 128, 3)>

print(imgs[0])
# if I wanted to access the actions, then its basically at f["data/demo_0/actions"][5]
# f["data/demo_3/obs/eye_in_hand_rgb"][-1]

'''
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

'''