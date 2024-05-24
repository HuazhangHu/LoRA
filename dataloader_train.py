import os
from PIL import Image
import random
import numpy as np

import torch
import transformers
from torchvision import transforms

from transformers import CLIPTextModel, CLIPTokenizer

from torch.utils.data import Dataset, DataLoader

class TrainData(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.data_dir = cfg.data_dir
        self.labels_path = os.path.join(self.data_dir, 'metadata.jsonl')

        self.image_paths = self._get_images(self.data_dir)
        self.captions_json = self._get_captions(self.labels_path )

        self.train_transforms = transforms.Compose(
            [
                transforms.Resize(
                    cfg.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomCrop(cfg.resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        caption = self.captions_json[os.path.basename(image_path)]
        image, caption =self.preprocess_train(image_path, caption)
        image = image.to(memory_format=torch.contiguous_format).float()
        return image, caption

    def __len__(self):
        return len(self.image_paths)
    
    def _get_images(self, data_dir):
        if not os.path.exists(data_dir):
            raise(RuntimeError("folder %s doesn't exist. Check list. " % (data_dir)))
        image_paths = []
        for filename in os.listdir(data_dir):
            file_path = os.path.join(data_dir, filename)
            # 检查文件是否是图片
            if os.path.isfile(file_path) and filename.lower().endswith('.jpg'):
                # 将图片路径添加到列表中
                image_paths.append(file_path)

        return image_paths
    
    def _get_captions(self, labels_path):
        captions_json= {}
        import json
        with open(labels_path, 'r', encoding='utf-8') as file:
            for line in file:
                json_obj = json.loads(line)
                captions_json[json_obj["file_name"]]=json_obj["text"] # {"1.jpg": "caption"}

        return captions_json
    
    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(self, caption, is_train=True):
        captions = []
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption)
                            if is_train else caption[0])
        else:
            raise ValueError(
                f"Caption column `{self.caption_column}` should contain either strings or lists of strings."
            )
        tokenizer = CLIPTokenizer.from_pretrained(self.cfg.pretrained_model_name_or_path, subfolder="tokenizer")

        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    def preprocess_train(self, image, caption):
        image = Image.open(image).convert("RGB")
        image = self.train_transforms(image)
        caption = self.tokenize_captions(caption)

        return (image, caption)
    
# if __name__ == "__main__":
#     train_dataset = TrainData(cfg)

#     train_dataloader = torch.utils.data.DataLoader(
#         train_dataset,
#         shuffle=True,
#         collate_fn=collate_fn,
#         batch_size=cfg.train_batch_size,
#         num_workers=cfg.dataloader_num_workers,
#     )