import os

# 定义文件夹路径
folder_path = '/workspace/data1/sam_data/F0003RL5S3R1/masks/'

# 获取文件夹名称
folder_name = os.path.basename(folder_path)

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 检查文件名是否以 'CT0002' 开头
    if filename.startswith('CT0002'):
        # 构造新的文件名，将 'CT0002' 替换为文件夹名称
        new_filename = filename.replace('CT0002', 'F0003RL5S3R1', 1)

        # 获取文件的完整路径
        old_file_path = os.path.join(folder_path, filename)
        new_file_path = os.path.join(folder_path, new_filename)

        # 重命名文件
        os.rename(old_file_path, new_file_path)
        print(f'Renamed: {filename} -> {new_filename}')
