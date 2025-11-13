import os
import json
import trimesh
import numpy as np
from scipy.spatial import cKDTree
from collections import defaultdict


# 加载 .obj 文件并返回对齐到原点的网格
def load_mesh(obj_path):
    try:
        mesh = trimesh.load(obj_path, process=False)

        # 如果返回的是 Scene 对象，提取其中的 Mesh
        if isinstance(mesh, trimesh.Scene):
            if len(mesh.geometry) == 0:
                raise ValueError(f"No geometry found in scene: {obj_path}")
            mesh = mesh.dump(concatenate=True)

        vertices = np.array(mesh.vertices)

        # 将点云平移到质心对齐到原点
        centered_vertices = center_point_cloud(vertices)

        # 使用 trimesh 的点云采样功能
        mesh.vertices = centered_vertices
        return mesh
    except Exception as e:
        print(f"Error loading mesh from {obj_path}: {e}")
        return None


# 计算质心
def compute_centroid(point_cloud):
    return np.mean(point_cloud, axis=0)


# 将点云平移到原点
def center_point_cloud(point_cloud):
    centroid = compute_centroid(point_cloud)
    return point_cloud - centroid


# 从网格中采样点云
def sample_points(mesh, num_points=10000):
    return mesh.sample(num_points)


# 计算点云之间的最近距离
def compute_distances(points_src, points_tgt):
    tree = cKDTree(points_tgt)
    distances, _ = tree.query(points_src)
    return distances


# 计算 Precision 和 Recall
def precision_recall(points1, points2, threshold):
    distances_1_to_2 = compute_distances(points1, points2)
    distances_2_to_1 = compute_distances(points2, points1)

    precision = (distances_1_to_2 < threshold).mean()
    recall = (distances_2_to_1 < threshold).mean()

    return precision, recall


# 计算 F-score
def compute_fscore(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


# 计算 F-score、Precision 和 Recall
def calculate_fscore(obj_path1, obj_path2, num_points=10000, threshold=0.02):
    # 加载网格并对齐质心
    mesh1 = load_mesh(obj_path1)
    mesh2 = load_mesh(obj_path2)

    # 如果加载失败，跳过
    if mesh1 is None or mesh2 is None:
        return None, None, None

    # 从网格中采样点云
    points1 = sample_points(mesh1, num_points)
    points2 = sample_points(mesh2, num_points)

    # 计算 Precision 和 Recall
    precision, recall = precision_recall(points1, points2, threshold)

    # 计算 F-score
    fscore = compute_fscore(precision, recall)

    return fscore, precision, recall


# 遍历推理目录和 Ground Truth 目录，计算 F-score、Precision 和 Recall
def compute_fscore_for_samples(gt_roots, inference_root, output_file, num_points=10000, threshold=0.02):
    fscore_results = {}
    sample_fscore = defaultdict(list)
    sample_precision = defaultdict(list)
    sample_recall = defaultdict(list)

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

            # 计算 F-score、Precision 和 Recall
            fscore, precision, recall = calculate_fscore(resize3d_obj_path, gt_obj_file, num_points, threshold)

            # 如果计算失败，跳过此帧
            if fscore is None or precision is None or recall is None:
                print(f"Skipping frame {frame_view_dir} due to calculation error.")
                continue

            # 存储每个帧的结果
            fscore_results[frame_view_dir] = {
                "fscore": fscore,
                "precision": precision,
                "recall": recall
            }

            # 提取样本名（如 "F0001PL7S3R4"）
            sample_name = frame_view_dir.split('_')[0]

            # 将 F-score、Precision 和 Recall 加入到对应样本的列表中
            sample_fscore[sample_name].append(fscore)
            sample_precision[sample_name].append(precision)
            sample_recall[sample_name].append(recall)

            print(f"Processed {frame_view_dir}: F-score = {fscore}, Precision = {precision}, Recall = {recall}")

    # 计算每个样本的平均 F-score、Precision 和 Recall
    all_fscore = []
    all_precision = []
    all_recall = []
    for sample_name in sample_fscore.keys():
        avg_fscore = sum(sample_fscore[sample_name]) / len(sample_fscore[sample_name])
        avg_precision = sum(sample_precision[sample_name]) / len(sample_precision[sample_name])
        avg_recall = sum(sample_recall[sample_name]) / len(sample_recall[sample_name])

        fscore_results[sample_name] = {
            "average_fscore": avg_fscore,
            "average_precision": avg_precision,
            "average_recall": avg_recall
        }

        all_fscore.append(avg_fscore)
        all_precision.append(avg_precision)
        all_recall.append(avg_recall)

    # 计算所有样本的总体平均 F-score、Precision 和 Recall
    if all_fscore:
        overall_fscore = sum(all_fscore) / len(all_fscore)
        overall_precision = sum(all_precision) / len(all_precision)
        overall_recall = sum(all_recall) / len(all_recall)

        fscore_results["overall_average"] = {
            "fscore": overall_fscore,
            "precision": overall_precision,
            "recall": overall_recall
        }

    # 将结果写入 JSON 文件
    with open(output_file, 'w') as f:
        json.dump(fscore_results, f, indent=4)


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
    output_file = os.path.join(inference_root, "fscore_results.json")

    compute_fscore_for_samples(gt_roots, inference_root, output_file)
