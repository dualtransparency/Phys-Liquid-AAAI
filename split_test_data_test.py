import os
import json

def validate_directories(json_file, target_root_1, target_root_2):
    # 打开并读取JSON文件
    with open(json_file, 'r') as f:
        data = json.load(f)

    missed_samples = []  # 用于记录未找到的样本
    extra_samples = []  # 用于记录多余的样本

    # 遍历JSON中的每个测试集路径
    for key, test_path in data.items():
        # 提取dataX和样本名
        path_parts = test_path.split('/')
        data_dir = path_parts[2]  # data1, data2, etc.
        sample_name = path_parts[3]  # F0006YL7S2R5

        # 生成测试结果的根路径
        trained_data_dir = f'/workspace/crm_inference_new/{data_dir}_trained/{sample_name}'

        # 获取原始测试集结果目录下的所有文件夹
        if os.path.exists(trained_data_dir):
            original_dirs = set(os.listdir(trained_data_dir))  # 原始目录中的文件夹
        else:
            print(f"Warning: {trained_data_dir} does not exist.")
            continue

        # 获取target_root_1和target_root_2中对应的文件夹
        target_dirs_1 = set(os.listdir(os.path.join(target_root_1, sample_name))) if os.path.exists(
            os.path.join(target_root_1, sample_name)) else set()
        target_dirs_2 = set(os.listdir(os.path.join(target_root_2, sample_name))) if os.path.exists(
            os.path.join(target_root_2, sample_name)) else set()

        # 合并两个目标目录的文件夹集合
        combined_target_dirs = target_dirs_1.union(target_dirs_2)

        # 校验：检查原始目录中的文件夹是否都在目标目录中
        missing_dirs = original_dirs - combined_target_dirs
        extra_dirs = combined_target_dirs - original_dirs

        if missing_dirs:
            missed_samples.append((sample_name, missing_dirs))
        if extra_dirs:
            extra_samples.append((sample_name, extra_dirs))

    # 打印校验结果
    if missed_samples:
        print("Missing directories in target directories:")
        for sample, missing in missed_samples:
            print(f"- Sample: {sample}, Missing: {missing}")
    else:
        print("All original directories are present in target directories.")

    if extra_samples:
        print("\nExtra directories found in target directories:")
        for sample, extra in extra_samples:
            print(f"- Sample: {sample}, Extra: {extra}")
    else:
        print("No extra directories found in target directories.")


# 示例用法
json_file_path = '/workspace/CRM/validate_data/hash_to_original.json'
target_root_directory_1 = '/workspace/crm_inference_test'
target_root_directory_2 = '/workspace/crm_inference_train'
validate_directories(json_file_path, target_root_directory_1, target_root_directory_2)
