import os
import json
import shutil


def copy_test_results(json_file, target_root_1, target_root_2):
    # 打开并读取JSON文件
    with open(json_file, 'r') as f:
        data = json.load(f)

    # 定义视角后缀
    views = ['000', '001', '003', '004']

    # 记录已处理的文件夹，避免重复
    processed_dirs = set()

    # 遍历JSON中的每个测试集路径
    for key, test_path in data.items():
        # 提取dataX和样本名
        path_parts = test_path.split('/')
        data_dir = path_parts[2]  # data1, data2, etc.
        sample_name = path_parts[3]  # F0006YL7S2R5
        sample_name_frame = path_parts[-1]  # F0006YL7S2R5_0061

        # 生成测试结果的根路径
        trained_data_dir = f'/workspace/crm_inference_new/{data_dir}_trained/{sample_name}'

        # 遍历四个视角
        for view in views:
            result_dir = os.path.join(trained_data_dir, f'{sample_name_frame}_{view}')

            # 确保结果目录存在
            if os.path.exists(result_dir):
                # 目标路径1是target_root_1下的同样目录结构
                target_dir_1 = os.path.join(target_root_1, sample_name, f'{sample_name_frame}_{view}')

                # 创建目标目录
                os.makedirs(target_dir_1, exist_ok=True)

                # 将结果文件夹下的所有文件复制到目标目录1
                for item in os.listdir(result_dir):
                    src_item = os.path.join(result_dir, item)
                    dst_item = os.path.join(target_dir_1, item)

                    if os.path.isdir(src_item):
                        shutil.copytree(src_item, dst_item, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src_item, dst_item)

                # 记录已处理的文件夹，避免重复
                processed_dirs.add(f'{sample_name_frame}_{view}')

                print(f"Copied {result_dir} to {target_dir_1}")
            else:
                print(f"Warning: {result_dir} does not exist.")

    # 第二步：处理未在JSON文件中指示的文件夹，复制到target_root_2
    for key, test_path in data.items():
        # 提取dataX和样本名
        path_parts = test_path.split('/')
        data_dir = path_parts[2]  # data1, data2, etc.
        sample_name = path_parts[3]  # F0006YL7S2R5

        # 生成测试结果的根路径
        trained_data_dir = f'/workspace/crm_inference_new/{data_dir}_trained/{sample_name}'

        # 确保训练数据目录存在
        if os.path.exists(trained_data_dir):
            # 遍历该目录下的所有文件夹
            for item in os.listdir(trained_data_dir):
                # 只处理未被处理过的文件夹
                if item not in processed_dirs:
                    src_item = os.path.join(trained_data_dir, item)
                    target_dir_2 = os.path.join(target_root_2, sample_name, item)

                    # 创建目标目录2
                    os.makedirs(target_dir_2, exist_ok=True)

                    # 复制未处理的文件夹到target_root_2
                    if os.path.isdir(src_item):
                        shutil.copytree(src_item, target_dir_2, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src_item, target_dir_2)

                    print(f"Copied {src_item} to {target_dir_2}")


# 示例用法
json_file_path = '/workspace/CRM/validate_data/hash_to_original.json'
target_root_directory_1 = '/workspace/crm_inference_test'
target_root_directory_2 = '/workspace/crm_inference_train'
copy_test_results(json_file_path, target_root_directory_1, target_root_directory_2)
