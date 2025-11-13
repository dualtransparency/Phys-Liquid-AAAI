import os
import json
import trimesh


# 1. 读取 .obj 文件并返回 mesh 对象
def load_obj_file(obj_path):
    mesh = trimesh.load_mesh(obj_path)
    return mesh


# 2. 对 mesh 进行放缩
def scale_mesh(mesh, ratio):
    # 使用 apply_scale 对 mesh 进行缩放
    mesh.apply_scale(ratio)
    return mesh


# 3. 保存放缩后的 .obj 文件
def save_obj_file(obj_path, mesh):
    mesh.export(obj_path, file_type='obj')


# 4. 计算 3D 模型的体积（使用 trimesh 库）
def calculate_volume(mesh):
    volume = mesh.volume  # 计算体积
    return volume


# 5. 完整流程：加载、缩放、保存、计算体积
def process_obj_file(obj_path, output_path, ratio):
    # 1. 加载 obj 文件
    mesh = load_obj_file(obj_path)

    # 2. 对 mesh 进行放缩
    scaled_mesh = scale_mesh(mesh, ratio)

    # 3. 保存放缩后的 obj 文件
    save_obj_file(output_path, scaled_mesh)

    # 4. 计算体积
    volume = calculate_volume(scaled_mesh)
    print(f"Scaled 3D model volume: {volume}")
    return volume


# 6. 递归处理目录
def process_directory(root_dir, input_obj_filename, output_obj_filename, volumes_filename, inference_filename, real_volumes_filename):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if volumes_filename in filenames and inference_filename in filenames:
            volumes_path = os.path.join(dirpath, volumes_filename)
            inference_path = os.path.join(dirpath, inference_filename)
            obj_path = os.path.join(dirpath, input_obj_filename)
            real_obj_path = os.path.join(dirpath, output_obj_filename)
            real_volumes_path = os.path.join(dirpath, real_volumes_filename)

            # 读取 volumes.json
            with open(volumes_path, 'r') as f:
                volumes_data = json.load(f)
                fake_volume_data = volumes_data[input_obj_filename]
                x_fake = fake_volume_data['x_width']
                y_fake = fake_volume_data['y_width']
                z_fake = fake_volume_data['z_width']
                fake_volume = fake_volume_data['volume']

            # 读取 inference.json
            with open(inference_path, 'r') as f:
                inference_data = json.load(f)
                xyz_ratio_geometric = inference_data['xyz_ratio_geometric']

            # 计算真实的 x, y, z 宽度
            x_real = x_fake / (xyz_ratio_geometric / 0.01)
            y_real = y_fake / (xyz_ratio_geometric / 0.01)
            z_real = z_fake / (xyz_ratio_geometric / 0.01)

            # 处理 obj 文件，生成 real3d.obj 并计算真实体积
            real_volume = process_obj_file(obj_path, real_obj_path, 0.01 / xyz_ratio_geometric)

            # 保存真实体积数据到 real_volumes.json
            real_volume_data = {
                output_obj_filename: {
                    "x_real": x_real,
                    "y_real": y_real,
                    "z_real": z_real,
                    "volume": real_volume
                }
            }
            with open(real_volumes_path, 'w') as f:
                json.dump(real_volume_data, f, indent=4)

            print(f"Processed {dirpath}: {output_obj_filename} and {real_volumes_filename} generated.")


# 主函数
if __name__ == "__main__":
    root_dir = "/root/CRM/out_old/"  # 输入根目录

    # 定义文件名变量
    input_obj_filename = 'output3d.obj'  # 输入的 .obj 文件名
    volumes_filename = 'volumes.json'  # 输入的体积数据文件名
    inference_filename = 'inference.json'  # 输入的推理结果文件名
    output_obj_filename = 'resize3d.obj'  # 输出的缩放后的 .obj 文件名
    real_volumes_filename = 'resize3d.json'  # 输出的真实体积数据文件名

    # 调用处理函数
    process_directory(root_dir, input_obj_filename, output_obj_filename, volumes_filename, inference_filename, real_volumes_filename)
