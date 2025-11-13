"""

/workspace/data/crm_data/
    |-- F0001CL2S1R5/
    |    |-- F0001CL2S1R5_0076_002/
    |    |    |-- pixel_images.png
    |    |    |-- xyz_images.png
    |    |    |-- output3d.glb
    |    |    `-- output3d.obj
    |    |-- ...
    |-- F0001OL1S2R2/
    |    |-- F0001OL1S2R2_0078_003/
    |    |    |-- pixel_images.png
    |    |    |-- xyz_images.png
    |    |    |-- output3d.glb
    |    |    `-- output3d.obj
    |    |-- ...
    ...

"""

import os

# 要检查的目录路径
base_dir = '/workspace/crm_inference_new/data2_original/'

# 所需的文件列表
required_files = ['pixel_images.png', 'xyz_images.png', 'preprocessed_image.png', 'output3d.obj']


def check_directory_structure(base_dir):
    """
    检查 base_dir 中的目录结构是否完备。
    """
    incomplete_dirs = []  # 用于存储不完整的目录
    total_dirs = 0  # 记录总的子目录数量
    complete_dirs = 0  # 记录完整的子目录数量

    # 遍历 base_dir 中的每个子目录
    for sample_dir in os.listdir(base_dir):
        sample_path = os.path.join(base_dir, sample_dir)

        # 确保是目录
        if os.path.isdir(sample_path):
            # 遍历子目录中的每个图片子目录
            for sub_dir in os.listdir(sample_path):
                sub_dir_path = os.path.join(sample_path, sub_dir)

                # 确保是目录
                if os.path.isdir(sub_dir_path):
                    total_dirs += 1
                    missing_files = []

                    # 检查该子目录是否包含所有必需的文件
                    for required_file in required_files:
                        file_path = os.path.join(sub_dir_path, required_file)
                        if not os.path.exists(file_path):
                            missing_files.append(required_file)

                    # 如果有缺失的文件，记录下来
                    if missing_files:
                        incomplete_dirs.append({
                            'path': sub_dir_path,
                            'missing_files': missing_files
                        })
                    else:
                        complete_dirs += 1

    # 输出结果
    print(f"总共检查了 {total_dirs} 个子目录。")
    print(f"其中 {complete_dirs} 个子目录是完整的。")

    if incomplete_dirs:
        print(f"{len(incomplete_dirs)} 个子目录不完整，缺少文件如下：")
        for entry in incomplete_dirs:
            print(f"目录: {entry['path']}")
            print(f"缺少的文件: {', '.join(entry['missing_files'])}")
    else:
        print("所有子目录都完整。")


if __name__ == "__main__":
    check_directory_structure(base_dir)
