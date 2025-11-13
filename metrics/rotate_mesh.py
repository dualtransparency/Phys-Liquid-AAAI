import trimesh
import numpy as np
import os


# 定义旋转操作的函数
def rotate_mesh(mesh):
    # 旋转 90 度绕 X 轴
    rotation_x = trimesh.transformations.rotation_matrix(
        angle=np.pi / 2,  # 90° = π/2 弧度
        direction=[1, 0, 0],  # X轴
        point=[0, 0, 0]
    )

    # 旋转 180 度绕 Z 轴
    rotation_z = trimesh.transformations.rotation_matrix(
        angle=np.pi,  # 180° = π 弧度
        direction=[0, 0, 1],  # Z轴
        point=[0, 0, 0]
    )

    # 应用旋转矩阵
    mesh.apply_transform(rotation_x)  # 先绕 X 轴旋转
    mesh.apply_transform(rotation_z)  # 再绕 Z 轴旋转


# 遍历目录并处理所有 .obj 文件
def process_directory(directory):
    # 遍历目录下的所有文件
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 只处理 .obj 文件
            if file.endswith('.obj'):
                file_path = os.path.join(root, file)
                try:
                    # 加载文件
                    mesh = trimesh.load(file_path)

                    # 旋转网格
                    rotate_mesh(mesh)

                    # 保存旋转后的网格，替换原文件
                    mesh.export(file_path)

                    print(f"文件已处理并保存: {file_path}")
                except Exception as e:
                    print(f"处理文件 {file_path} 时出错: {e}")


# 指定要处理的目录
directory_path = "/workspace/crm_oneview_multiview3/"  # 修改为你的目录路径

# 开始处理目录
process_directory(directory_path)
