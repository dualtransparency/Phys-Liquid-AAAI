import os
import json
import trimesh
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm  # 进度条库
import argparse


def calculate_volume_and_dimensions(obj_file_path):
    """计算 .obj 文件的体积和 x, y, z 方向的宽度"""
    try:
        mesh = trimesh.load_mesh(obj_file_path)
        volume = mesh.volume
        bounding_box_min = mesh.bounds[0]  # 最小值 (x_min, y_min, z_min)
        bounding_box_max = mesh.bounds[1]  # 最大值 (x_max, y_max, z_max)
        x_width = bounding_box_max[0] - bounding_box_min[0]
        y_width = bounding_box_max[1] - bounding_box_min[1]
        z_width = bounding_box_max[2] - bounding_box_min[2]
        return {
            "volume": volume,
            "x_width": x_width,
            "y_width": y_width,
            "z_width": z_width
        }
    except Exception as e:
        print(f"Error processing {obj_file_path}: {e}")
        return None


def process_directory(root_dir, mode, max_workers=4):
    """遍历根目录，查找 .obj 文件并计算体积和尺寸（多进程处理）"""
    obj_files_to_process = []
    json_paths = {}
    success_count = 0
    failure_count = 0
    error_file_path = os.path.join(root_dir, "error_files.txt")  # 报错文件路径

    # 遍历目录，收集 .obj 文件路径
    for dirpath, dirnames, filenames in os.walk(root_dir):
        obj_files = [f for f in filenames if f.lower().endswith('.obj')]

        if obj_files:
            json_file_path = os.path.join(dirpath, 'volumes.json')
            volume_data = {}

            # 如果是追加模式且已经存在 volumes.json 文件，先加载它
            if mode == "append" and os.path.exists(json_file_path):
                with open(json_file_path, 'r') as json_file:
                    try:
                        volume_data = json.load(json_file)
                    except json.JSONDecodeError:
                        print(f"Warning: {json_file_path} is not a valid JSON file, it will be overwritten.")

            # 根据模式选择要处理的 .obj 文件
            for obj_file in obj_files:
                obj_file_path = os.path.join(dirpath, obj_file)
                if mode == "overwrite" or obj_file not in volume_data:
                    obj_files_to_process.append((obj_file_path, obj_file, json_file_path))
                    json_paths[json_file_path] = volume_data

    # 如果没有需要处理的文件，直接返回
    if not obj_files_to_process:
        print("没有需要处理的文件。")
        return

    # 打开报错文件
    with open(error_file_path, 'w') as error_file:

        # 多进程处理 .obj 文件
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(calculate_volume_and_dimensions, obj_file_path): (obj_file_path, obj_file, json_file_path)
                       for obj_file_path, obj_file, json_file_path in obj_files_to_process}

            # 使用 tqdm 显示进度条
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing .obj files"):
                obj_file_path, obj_file, json_file_path = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        # 获取对应的 volume_data 并更新
                        volume_data = json_paths[json_file_path]
                        volume_data[obj_file] = result  # 保存体积和宽度信息
                        success_count += 1  # 统计成功的文件
                    else:
                        failure_count += 1  # 统计失败的文件
                        error_file.write(f"{obj_file_path}\n")  # 将失败的文件路径写入报错文件
                        # 如果是 overwrite 模式，删除旧的条目
                        if mode == "overwrite" and obj_file in json_paths[json_file_path]:
                            del json_paths[json_file_path][obj_file]
                except Exception as e:
                    print(f"Failed to process {obj_file_path}: {e}")
                    failure_count += 1
                    error_file.write(f"{obj_file_path}\n")  # 将失败的文件路径写入报错文件
                    # 如果是 overwrite 模式，删除旧的条目
                    if mode == "overwrite" and obj_file in json_paths[json_file_path]:
                        del json_paths[json_file_path][obj_file]

    # 将所有体积和尺寸数据写入对应的 JSON 文件
    for json_file_path, volume_data in json_paths.items():
        # 如果是 overwrite 模式且 volume_data 为空，删除该 JSON 文件
        if mode == "overwrite" and not volume_data:
            try:
                os.remove(json_file_path)
                print(f"Deleted empty JSON file: {json_file_path}")
            except OSError as e:
                print(f"Error deleting {json_file_path}: {e}")
        else:
            with open(json_file_path, 'w') as json_file:
                json.dump(volume_data, json_file, indent=4)
            print(f"Saved volume and dimensions data to {json_file_path}")

    # 计算并打印成功率
    total_files = success_count + failure_count
    if total_files > 0:
        success_rate = (success_count / total_files) * 100
        print(f"成功处理的文件数量: {success_count}")
        print(f"失败的文件数量: {failure_count}")
        print(f"成功率: {success_rate:.2f}%")
    else:
        print("没有处理任何文件。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process .obj files and calculate their volumes and dimensions.")
    parser.add_argument(
        "--rootdir",
        type=str,
        default="/workspace/crm_inference_new/data4_original",
        help="Root directory to search for .obj files.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["append", "overwrite"],
        default="overwrite",
        help="Mode to run the script in: 'append' to only process new files, 'overwrite' to recalculate all volumes.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Maximum number of workers for parallel processing.",
    )
    args = parser.parse_args()

    root_directory = args.rootdir
    if os.path.isdir(root_directory):
        process_directory(root_directory, args.mode, max_workers=args.max_workers)
        print("处理完成！")
    else:
        print("无效的目录路径。")
