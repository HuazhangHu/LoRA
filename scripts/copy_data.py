import os
import shutil

# 设置源文件夹路径
source_dir = 'dataset/cartoon'
# 设置目标文件夹路径
dest_dir = 'dataset/cartoon'

# 确保目标文件夹存在
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

# 支持的图片格式
image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']

# 计数器，用于给图片文件重新命名
counter = 1

# 遍历源文件夹及其所有子文件夹
for root, dirs, files in os.walk(source_dir):
    for file in files:
        # 检查文件扩展名是否是图片
        if any(file.lower().endswith(ext) for ext in image_extensions):
            # 构建完整的文件路径
            file_path = os.path.join(root, file)
            # 构建新的文件名，格式为 "序号.扩展名"
            new_file_name = f"{counter:04d}{os.path.splitext(file)[1]}"
            new_file_path = os.path.join(dest_dir, new_file_name)
            # 复制文件到目标文件夹
            shutil.copy2(file_path, new_file_path)
            # 更新计数器
            counter += 1

print("图片复制完成。")