import os
import json
from tqdm import tqdm 

import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16, device_map="cuda:0")
model.to('cuda:0')
image_paths= "dataset/cartoon"

metadata_file = 'dataset/cartoon/metadata.jsonl'
with open(metadata_file, 'w', encoding='utf-8') as f:
    # 遍历文件夹中的所有文件
    for filename in tqdm(sorted(os.listdir(image_paths))):
        if filename.endswith('.jpg'):
            image_path = os.path.join(image_paths, filename)
            raw_image = Image.open(image_path).convert('RGB')
            question = "What content is described in the picture?"
            inputs = processor(raw_image, return_tensors="pt").to("cuda", torch.float16)

            generated_ids = model.generate(**inputs)
            text_info =  processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            # print(text_info)
            metadata = {
                    "file_name": filename,
                    "text": text_info
                }
                # 将数据转换为JSON格式，并写入文件
                # 使用'ensure_ascii=False'参数以支持Unicode字符
            f.write(json.dumps(metadata, ensure_ascii=False) + '\n')
            
print('image caption over')