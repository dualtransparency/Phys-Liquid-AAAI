import os
import hashlib
import json

def calculate_md5(sample_name):
    """根据样本名称计算MD5哈希"""
    return hashlib.md5(sample_name.encode()).hexdigest()

def find_liquid_dirs(root_dirs):
    """找到所有包含液体模型的文件夹"""
    liquid_dirs = []
    for root_dir in root_dirs:
        for subdir, dirs, files in os.walk(root_dir):
            if os.path.basename(subdir).startswith('F00') and 'liquid' in dirs:
                liquid_path = os.path.join(subdir, 'liquid')
                liquid_dirs.append(liquid_path)
    return liquid_dirs

def map_hash_to_original(validate_folder, input_folders, output_folder):
    """只溯源 validate_folder 下的哈希目录"""
    hash_mapping = {}

    # 获取 validate_folder 目录下所有的哈希文件夹名
    hash_folders = [folder for folder in os.listdir(validate_folder) if os.path.isdir(os.path.join(validate_folder, folder))]

    # 遍历原始输入文件夹，计算每个文件的哈希值
    for input_folder in input_folders:
        for filename in os.listdir(input_folder):
            if filename.endswith('.png'):
                parts = filename.split('_')
                sample_name = f"{parts[0]}_{parts[1]}"
                sample_hash = calculate_md5(sample_name)

                # 如果哈希值在 validate_folder 中存在，则记录对应关系
                if sample_hash in hash_folders:
                    hash_mapping[sample_hash] = os.path.join(input_folder, sample_name)

    # 将哈希值与原始文件名的映射保存为JSON文件
    output_json_path = os.path.join(output_folder, 'hash_to_original.json')
    with open(output_json_path, 'w') as json_file:
        json.dump(hash_mapping, json_file, indent=4)

    print(f"哈希值与原始文件名的映射已保存到 {output_json_path}")

if __name__ == '__main__':
    # 原始数据的根文件夹
    root_dirs = [
        r'/workspace/data1',
        r'/workspace/data2',
        r'/workspace/data3',
        r'/workspace/data4',
    ]

    # 要溯源的 validate_data 文件夹
    validate_folder = r'/workspace/CRM/validate_data/'

    # 查找所有包含液体模型的文件夹
    input_folders = find_liquid_dirs(root_dirs)

    # 生成哈希值与原始文件名的映射，只针对 validate_data 中的哈希文件夹
    map_hash_to_original(validate_folder, input_folders, validate_folder)
