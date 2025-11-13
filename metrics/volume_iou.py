import os
import json
import open3d as o3d
import numpy as np
from collections import defaultdict


# 计算质心
def compute_centroid(point_cloud):
    return np.mean(point_cloud, axis=0)


# 将点云平移到原点
def center_point_cloud(point_cloud):
    centroid = compute_centroid(point_cloud)
    return point_cloud - centroid


# 加载 .obj 文件并转换为对齐到原点的体素网格
def load_voxel_mesh(obj_path, voxel_size=0.01):
    mesh = o3d.io.read_triangle_mesh(obj_path)

    # 获取顶点并对齐到质心
    vertices = np.asarray(mesh.vertices)
    centered_vertices = center_point_cloud(vertices)
    mesh.vertices = o3d.utility.Vector3dVector(centered_vertices)

    # 将对齐后的网格转换为体素网格
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size)
    return voxel_grid


# 计算体素的 IoU
def calculate_voxel_iou(voxel_a, voxel_b, voxel_size=0.01):
    # 获取体素的坐标
    voxels_a = set([tuple(voxel.grid_index) for voxel in voxel_a.get_voxels()])
    voxels_b = set([tuple(voxel.grid_index) for voxel in voxel_b.get_voxels()])

    # 计算交集
    intersection_voxels = voxels_a.intersection(voxels_b)

    # 计算各自的体积和交集体积
    volume_a = len(voxels_a) * voxel_size ** 3
    volume_b = len(voxels_b) * voxel_size ** 3
    volume_intersection = len(intersection_voxels) * voxel_size ** 3

    # 计算IoU
    iou = volume_intersection / (volume_a + volume_b - volume_intersection)

    return iou, volume_a, volume_b, volume_intersection


# 遍历推理目录和 Ground Truth 目录，计算 IoU
def compute_voxel_iou_for_samples(gt_roots, inference_root, output_file, voxel_size=0.01):
    iou_results = {}
    sample_iou = defaultdict(list)

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

            # 尝试在多个 Ground Truth 目录中找到对应的 ground truth 文件
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

            # 加载推理结果和 ground truth 的体素网格
            voxel_a = load_voxel_mesh(gt_obj_file, voxel_size)
            voxel_b = load_voxel_mesh(resize3d_obj_path, voxel_size)

            # 计算 IoU
            iou, volume_a, volume_b, volume_intersection = calculate_voxel_iou(voxel_a, voxel_b, voxel_size)

            # 存储每个帧的结果
            iou_results[frame_view_dir] = {
                "iou": iou,
                "volume_a": volume_a,
                "volume_b": volume_b,
                "volume_intersection": volume_intersection
            }

            # 提取样本名（如 "F0001PL7S3R4"）
            sample_name = frame_view_dir.split('_')[0]

            # 将 IoU 加入到对应样本的列表中
            sample_iou[sample_name].append(iou)

            print(f"Processed {frame_view_dir}: IoU = {iou}")

    # 计算每个样本的平均 IoU
    all_iou = []
    for sample_name in sample_iou.keys():
        avg_iou = sum(sample_iou[sample_name]) / len(sample_iou[sample_name])
        iou_results[sample_name] = {
            "average_iou": avg_iou
        }

        all_iou.append(avg_iou)

    # 计算所有样本的总体平均 IoU
    if all_iou:
        overall_iou = sum(all_iou) / len(all_iou)

        iou_results["overall_average"] = {
            "iou": overall_iou
        }

    # 将结果写入 JSON 文件
    with open(output_file, 'w') as f:
        json.dump(iou_results, f, indent=4)


# 主函数
if __name__ == "__main__":
    gt_roots = [
        "/workspace/data1",
        "/workspace/data2",
        "/workspace/data3",
        "/workspace/data4",
    ]
    inference_root = "/workspace/crm_oneview_multiview3/"  # 推理结果根目录

    output_file = os.path.join(inference_root, "voxel_iou_results.json")

    compute_voxel_iou_for_samples(gt_roots, inference_root, output_file)
