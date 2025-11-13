import os

# 要删除的文件列表
files_to_delete = [
    'inference.json',
    'material.mtl',
    'material_0.png',
    'resize3d.obj',
    'resize3d.json'
]

# 删除文件的函数
def delete_files_in_directory(directory):
    # 遍历目录及其子目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 如果文件在要删除的列表中
            if file in files_to_delete:
                file_path = os.path.join(root, file)
                try:
                    # 删除文件
                    os.remove(file_path)
                    print(f"已删除文件: {file_path}")
                except Exception as e:
                    print(f"删除文件 {file_path} 时出错: {e}")

# 指定要处理的根目录
root_directory = "/workspace/crm_multiview_new/"  # 修改为你的根目录路径

# 执行删除操作
delete_files_in_directory(root_directory)
