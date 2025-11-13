import os
import json
from collections import defaultdict

# Ground truth 和推理结果的根目录
ground_truth_roots = [
    "/workspace/data1",
    "/workspace/data2",
    "/workspace/data3",
    "/workspace/data4",
]
inference_root = "/workspace/crm_oneview_multiview3/"

# 结果输出文件路径
output_file = os.path.join(inference_root, "relative_bbox_volume_errors.json")

# 初始化结果字典
relative_bbox_volume_errors = {}
sample_errors = defaultdict(list)  # 用于存储每个样本的误差

# 遍历推理结果目录中的每个样本文件夹
for sample_folder in os.listdir(inference_root):
    sample_path = os.path.join(inference_root, sample_folder)

    # 检查是否是目录
    if os.path.isdir(sample_path):
        # 遍历样本文件夹中的每个推理结果子文件夹（文件名_帧数_视角）
        for sub_folder in os.listdir(sample_path):
            sub_folder_path = os.path.join(sample_path, sub_folder)

            # 确保子文件夹存在并且包含 resize3d.json 文件
            real_volume_file = os.path.join(sub_folder_path, "resize3d.json")
            if os.path.exists(real_volume_file):
                # 读取推理结果的 bbox 体积
                with open(real_volume_file, 'r') as f:
                    predicted_data = json.load(f)
                    predicted_bbox_x = predicted_data["resize3d.obj"]["x_real"]
                    predicted_bbox_y = predicted_data["resize3d.obj"]["y_real"]
                    predicted_bbox_z = predicted_data["resize3d.obj"]["z_real"]

                    # 计算推理结果的 bbox 体积
                    predicted_bbox_volume = predicted_bbox_x * predicted_bbox_y * predicted_bbox_z

                # 从文件名中提取帧号（假设帧号是文件名中的第二部分）
                frame_number = sub_folder.split('_')[1]

                # 尝试在多个 Ground Truth 目录中找到对应的 volumes.json 文件
                ground_truth_volume_file = None
                for ground_truth_root in ground_truth_roots:
                    ground_truth_folder = os.path.join(ground_truth_root, sample_folder, "OBJ", "mesh")
                    potential_gt_volume_file = os.path.join(ground_truth_folder, "volumes.json")

                    if os.path.exists(potential_gt_volume_file):
                        ground_truth_volume_file = potential_gt_volume_file
                        break

                # 如果没有找到 ground truth volumes.json 文件，跳过
                if ground_truth_volume_file is None:
                    print(f"Warning: Ground truth volumes.json not found for {sample_folder}")
                    continue

                # 读取 ground truth 的 volumes.json 文件
                with open(ground_truth_volume_file, 'r') as f:
                    ground_truth_data = json.load(f)

                    # 构造对应的 obj 文件名
                    obj_filename = f"fluid_mesh_{frame_number.zfill(4)}.obj"

                    # 确保该帧的 bbox 数据存在
                    if obj_filename in ground_truth_data:
                        ground_truth_bbox_x = ground_truth_data[obj_filename]["x_width"]
                        ground_truth_bbox_y = ground_truth_data[obj_filename]["y_width"]
                        ground_truth_bbox_z = ground_truth_data[obj_filename]["z_width"]

                        # 计算 ground truth 的 bbox 体积
                        ground_truth_bbox_volume = ground_truth_bbox_x * ground_truth_bbox_y * ground_truth_bbox_z

                        # 计算相对 bbox 体积误差
                        relative_bbox_error = abs(predicted_bbox_volume - ground_truth_bbox_volume) / ground_truth_bbox_volume

                        # 保存每个帧的结果
                        relative_bbox_volume_errors[sub_folder] = relative_bbox_error

                        # 提取样本名（如 "F0001PL7S3R4"）
                        sample_name = sub_folder.split('_')[0]

                        # 将误差加入到对应样本的列表中
                        sample_errors[sample_name].append(relative_bbox_error)
                    else:
                        print(f"Warning: {obj_filename} not found in ground truth volumes.json")

# 计算每个样本的平均误差
all_sample_errors = []  # 用于保存所有样本的平均误差
for sample_name, errors in sample_errors.items():
    average_error = sum(errors) / len(errors)
    relative_bbox_volume_errors[sample_name] = average_error
    all_sample_errors.append(average_error)

# 计算所有样本的平均误差
if all_sample_errors:
    overall_average_error = sum(all_sample_errors) / len(all_sample_errors)
    relative_bbox_volume_errors["overall_average_error"] = overall_average_error

# 将结果写入 JSON 文件
with open(output_file, 'w') as f:
    json.dump(relative_bbox_volume_errors, f, indent=4)

print(f"Relative bbox volume errors written to {output_file}")
