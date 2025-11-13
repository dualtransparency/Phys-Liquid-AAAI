import os
import shutil


def copy_specific_folders(folder_list, source_root, target_root):
    """
    复制指定的文件夹到目标目录。

    :param folder_list: 包含要复制的文件夹名称的列表
    :param source_root: 测试集结果路径的根目录
    :param target_root: 要复制到的目标路径
    """
    for folder_name in folder_list:
        # 遍历每个文件夹名称
        found = False
        for data_dir in os.listdir(source_root):
            # 构建完整的源文件夹路径
            source_folder = os.path.join(source_root, data_dir, folder_name)

            # 检查源文件夹是否存在
            if os.path.exists(source_folder):
                found = True
                # 构建目标文件夹路径
                target_folder = os.path.join(target_root, folder_name)

                # 创建目标文件夹
                os.makedirs(target_folder, exist_ok=True)

                # 复制文件夹内容
                shutil.copytree(source_folder, target_folder, dirs_exist_ok=True)

                print(f"Copied {source_folder} to {target_folder}")
                break

        if not found:
            print(f"Warning: {folder_name} not found in {source_root}")


# 要复制的文件夹列表
folder_list = [
    "F0002PL8S3R5_0066_000",
    "F0003CL8S3R2_0061_000",
    "F0005PL1S1R2_0051_000",
    "F0006RL5S2R2_0076_000",
    "F0007YL5S1R5_0066_000",
    "F0008RL3S2R4_0026_000",
    "F0009YL1S2R3_0061_000",
    "F0010YL4S2R2_0076_000",
    "F0011RL5S3R2_0031_000",
    "F0012YL2S2R6_0006_000",
    "F0016CL7S3R3_0039_000",
    "F0019CL3S1R1_0006_000",
    "F0020PL1S1R1_0011_000"
]

# 示例用法
source_root_directory = '/workspace/crm_inference_new/data4_trained'  # 测试集结果的根目录
target_root_directory = '/workspace/crm_oneview_multiview'  # 目标根目录

copy_specific_folders(folder_list, source_root_directory, target_root_directory)
