import os

# 定义文件夹路径
folder_path = '/workspace/crm_inference_new/data1_trained/F0003RL5S3R1'

# 定义旧前缀和新前缀
old_prefix = 'CT0002'
new_prefix = 'F0003RL5S3R1'

# 遍历文件夹中的子文件夹
for folder_name in os.listdir(folder_path):
    # 检查是否是目录且前缀为旧前缀
    if os.path.isdir(os.path.join(folder_path, folder_name)) and folder_name.startswith(old_prefix):
        # 生成新的文件夹名称
        new_folder_name = folder_name.replace(old_prefix, new_prefix, 1)

        # 获取完整的旧文件夹路径和新文件夹路径
        old_folder_path = os.path.join(folder_path, folder_name)
        new_folder_path = os.path.join(folder_path, new_folder_name)

        # 重命名文件夹
        os.rename(old_folder_path, new_folder_path)
        print(f'Renamed: {old_folder_path} -> {new_folder_path}')

print("重命名完成")
