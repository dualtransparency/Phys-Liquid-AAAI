import os
import shutil


def organize_folders_by_prefix(root_dir):
    # 遍历root_dir目录下的所有文件夹
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)

        # 只处理文件夹
        if os.path.isdir(item_path):
            # 提取文件夹名的第一个字段作为父文件夹名
            parent_folder_name = item.split('_')[0]
            parent_folder_path = os.path.join(root_dir, parent_folder_name)

            # 创建父文件夹，如果不存在
            if not os.path.exists(parent_folder_path):
                os.makedirs(parent_folder_path)

            # 将当前文件夹移动到父文件夹中
            new_path = os.path.join(parent_folder_path, item)
            shutil.move(item_path, new_path)
            print(f"Moved {item_path} to {new_path}")


# 示例用法
root_directory = '/workspace/crm_oneview_multiview'
organize_folders_by_prefix(root_directory)
