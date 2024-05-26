
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import os
import json

# 指定图片所在的文件夹路径
image_folder = 'dataset/cartoon'
# 指定额外的文本信息
text_info = "Antique style"
# 指定metadata.jsonl文件的路径
metadata_file = 'metadata.jsonl'

# 支持的图片文件扩展名
extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']

# 打开metadata.jsonl文件准备写入
with open(metadata_file, 'w', encoding='utf-8') as f:
    # 遍历文件夹中的所有文件
    for filename in sorted(os.listdir(image_folder)):
        # 检查文件扩展名是否为图片格式
        if any(filename.lower().endswith(ext) for ext in extensions):
            # 构造图片的完整路径
            image_path = os.path.join(image_folder, filename)
            # 准备要写入的数据
            metadata = {
                "file_name": filename,
                "text": text_info
            }
            # 将数据转换为JSON格式，并写入文件
            # 使用'ensure_ascii=False'参数以支持Unicode字符
            f.write(json.dumps(metadata, ensure_ascii=False) + '\n')

print("Metadata writing complete.")