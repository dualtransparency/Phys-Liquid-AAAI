import os
import csv


def check_train_data_integrity(train_data_folder, csv_path):
    """
    检查 train_data 文件夹的完整性，确保每个样本文件夹中包含完整的 6 个视角文件。
    如果某个样本缺少视角，打印出该样本的 id 和缺少的视角信息。

    :param train_data_folder: 生成的训练数据文件夹路径
    :param csv_path: caption.csv 文件路径
    """
    # 读取 caption.csv 文件
    with open(csv_path, mode='r', newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        sample_ids = [row['id'] for row in reader]

    # 定义需要的视角文件名
    required_views = {f'{i:03d}.png' for i in range(6)}  # 000.png ~ 005.png

    # 遍历每个样本 ID，检查文件夹中的视角文件是否齐全
    for sample_id in sample_ids:
        sample_folder = os.path.join(train_data_folder, sample_id)

        if not os.path.exists(sample_folder):
            print(f"Error: Sample folder for ID '{sample_id}' not found!")
            continue

        # 获取该文件夹中的所有文件名
        view_files = set(os.listdir(sample_folder))

        # 检查缺少的视角文件
        missing_views = required_views - view_files

        if missing_views:
            print(f"Sample ID '{sample_id}' is missing the following views: {', '.join(sorted(missing_views))}")
        else:
            print(f"Sample ID '{sample_id}' is complete.")


if __name__ == '__main__':
    # 指定 train_data 文件夹路径和 caption.csv 文件路径
    train_data_folder = r'/workspace/CRM/train_data/'
    csv_path = os.path.join(train_data_folder, 'caption.csv')

    # 检查数据完整性
    check_train_data_integrity(train_data_folder, csv_path)
