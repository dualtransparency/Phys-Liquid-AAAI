import os
import json
import trimesh
import numpy as np
from scipy.spatial import cKDTree
from collections import defaultdict


# Chamfer Distance 计算函数
def chamfer_distance(point_cloud1, point_cloud2):
    # 使用KD树加速最近邻查找
    tree1 = cKDTree(point_cloud1)
    tree2 = cKDTree(point_cloud2)

    # 对于 point_cloud1 中的每个点，找到 point_cloud2 中的最近点
    dist1, _ = tree1.query(point_cloud2, k=1)
    dist2, _ = tree2.query(point_cloud1, k=1)

    # Chamfer Distance 是双向距离的平均值
    return np.mean(dist1) + np.mean(dist2)


# 计算质心
def compute_centroid(point_cloud):
    return np.mean(point_cloud, axis=0)


# 将点云平移到原点
def center_point_cloud(point_cloud):
    centroid = compute_centroid(point_cloud)
    return point_cloud - centroid


# 加载 OBJ 文件并返回对齐到原点的顶点的 numpy 数组
def load_obj_vertices(file_path):
    try:
        mesh = trimesh.load(file_path, process=False)

        # 如果返回的是 Scene 对象，提取其中的 Mesh
        if isinstance(mesh, trimesh.Scene):
            # 尝试从 Scene 中提取第一个 mesh
            if len(mesh.geometry) == 0:
                raise ValueError(f"No geometry found in scene: {file_path}")
            # 获取 scene 中的第一个 mesh
            mesh = mesh.dump(concatenate=True)

        # 将顶点转换为 numpy 数组
        vertices = np.array(mesh.vertices)

        # 将点云平移到质心对齐到原点
        centered_vertices = center_point_cloud(vertices)

        return centered_vertices
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None  # 如果加载失败，返回 None


# 遍历推理目录和 Ground Truth 目录，计算 Chamfer Distance
def compute_chamfer_distances(gt_roots, inference_root, output_file):
    chamfer_results = {}
    sample_errors = defaultdict(list)  # 用于存储每个样本的 Chamfer Distance

    # 遍历推理目录
    for sample_dir in os.listdir(inference_root):
        sample_path = os.path.join(inference_root, sample_dir)

        if not os.path.isdir(sample_path):
            continue

        # 遍历每个推理文件夹
        for frame_view_dir in os.listdir(sample_path):
            frame_view_path = os.path.join(sample_path, frame_view_dir)
            resize3d_obj_path = os.path.join(frame_view_path, 'resize3d.obj')

            # 如果没有 resize3d.obj，跳过
            if not os.path.exists(resize3d_obj_path):
                continue

            # 提取帧数
            parts = frame_view_dir.split('_')
            frame_number = parts[1]  # 帧数

            # 尝试在多个 Ground Truth 目录中找到匹配的 ground truth 文件
            gt_obj_file = None
            for gt_root in gt_roots:
                gt_sample_dir = os.path.join(gt_root, sample_dir, 'OBJ', 'mesh')
                potential_gt_obj_file = os.path.join(gt_sample_dir, f'fluid_mesh_{frame_number}.obj')

                if os.path.exists(potential_gt_obj_file):
                    gt_obj_file = potential_gt_obj_file
                    break

            # 如果没有找到 ground truth 文件，跳过
            if gt_obj_file is None:
                print(f"Ground truth file for frame {frame_view_dir} not found in any specified directories, skipping.")
                continue

            # 加载 resize3d.obj 和 ground truth obj
            resize3d_vertices = load_obj_vertices(resize3d_obj_path)
            gt_vertices = load_obj_vertices(gt_obj_file)

            # 如果加载失败，跳过此帧
            if resize3d_vertices is None or gt_vertices is None:
                print(f"Skipping frame {frame_view_dir} due to loading error.")
                continue

            # 计算 Chamfer Distance
            chamfer_dist = chamfer_distance(resize3d_vertices, gt_vertices)

            # 存储每个帧的结果
            chamfer_results[frame_view_dir] = chamfer_dist

            # 提取样本名（如 "F0001PL7S3R4"）
            sample_name = frame_view_dir.split('_')[0]

            # 将误差加入到对应样本的列表中
            sample_errors[sample_name].append(chamfer_dist)

            print(f"Processed {frame_view_dir}: Chamfer Distance = {chamfer_dist}")

    # 计算每个样本的平均 Chamfer Distance
    all_sample_errors = []  # 用于保存所有样本的平均 Chamfer Distance
    for sample_name, errors in sample_errors.items():
        average_error = sum(errors) / len(errors)
        chamfer_results[sample_name] = average_error
        all_sample_errors.append(average_error)

    # 计算所有样本的平均 Chamfer Distance
    if all_sample_errors:
        overall_average_error = sum(all_sample_errors) / len(all_sample_errors)
        chamfer_results["overall_average_error"] = overall_average_error

    # 将结果写入 JSON 文件
    with open(output_file, 'w') as f:
        json.dump(chamfer_results, f, indent=4)


# 主函数
if __name__ == "__main__":
    # Ground Truth 根目录可以是多个
    gt_roots = [
        "/workspace/data1",
        "/workspace/data2",
        "/workspace/data3",
        "/workspace/data4",
    ]

    inference_root = "/workspace/crm_oneview_multiview3/"  # 推理结果根目录
    output_file = os.path.join(inference_root, "chamfer_distances.json")

    compute_chamfer_distances(gt_roots, inference_root, output_file)

