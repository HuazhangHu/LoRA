from diffusers import StableDiffusionPipeline, ControlNetModel, StableDiffusionControlNetPipeline
import torch
from diffusers.utils import load_image

from PIL import Image
import cv2
import numpy as np


controlnet_canny_path = 'lllyasviel/sd-controlnet-canny'
controlnet = ControlNetModel.from_pretrained(controlnet_canny_path, torch_dtype=torch.float16)

# sd_pipeline= StableDiffusionPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
# sd_pipeline.to("cuda")

sd_control_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16)
sd_control_pipeline.to("cuda")

control_img_path = 'dataset/Antique_style/035.jpg'
prompt_image = Image.open("fish.jpg").convert("RGB")

init_image = Image.open(control_img_path).convert("RGB")
init_image = init_image.resize((768, 512))
np_image = np.array(init_image)

# get canny image
np_image = cv2.Canny(np_image, 100, 200)
np_image = np_image[:, :, None]
np_image = np.concatenate([np_image, np_image, np_image], axis=2)
canny_image = Image.fromarray(np_image)
canny_image.save('tmp_edge.jpg')

sd_control_pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")

generator = torch.Generator(device="cpu").manual_seed(33)
images = sd_control_pipeline(
    prompt='best quality, high quality',  # text prompt
    image=canny_image,  # control condition
    ip_adapter_image=prompt_image, # prompt image
    negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality", 
    num_inference_steps=50,
    generator=generator,
).images
images[0].save("IP_adapter_controlnet.jpg")
