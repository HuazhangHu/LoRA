from diffusers import DiffusionPipeline
import torch

def inference_lora():
    lora_path1 = 'ckpt/checkpoint-1000'
    lora_path2 = 'ckpt/checkpoint-4000'

    pipe_id="runwayml/stable-diffusion-v1-5"
    pipe = DiffusionPipeline.from_pretrained(pipe_id, torch_dtype=torch.float16).to("cuda")
    pipe.load_lora_weights(lora_path1, adapter_name ="1000")
    pipe.load_lora_weights(lora_path2, adapter_name ="4000")
    pipe.set_adapters("4000")
    pipe.to("cuda")


    prompt = 'A women wearing ancient clothes, standing by the river with a damaged city behind him, \
        with a sad expression, is a Chinese style comic'
    neg_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, {multiple people}'

    for i in range(5):
        image = pipe(prompt,
                     negative_prompt = neg_prompt,
                     num_inference_steps=100,
                     ).images[0]  
        
        image.save("output/lora/lora_ancient_{0}.png".format(i))

def inference_without_lora():
    pipe_id="runwayml/stable-diffusion-v1-5"
    pipe = DiffusionPipeline.from_pretrained(pipe_id, torch_dtype=torch.float16).to("cuda")
    pipe.to("cuda")

    prompt = 'A women wearing ancient clothes, standing by the river with a damaged city behind him, \
        with a sad expression, is a Chinese style comic'
    neg_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, {multiple people}'

    for i in range(5):
        image = pipe(prompt,
                     negative_prompt = neg_prompt,
                     num_inference_steps=100,
                     ).images[0]  
        
        image.save("output/sd/ancient_{0}.png".format(i))

inference_lora()
inference_without_lora()

