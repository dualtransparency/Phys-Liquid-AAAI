import os
import json

# 定义路径
multiview_data_root = '/workspace/data3/instantmesh_data/'  # 伪3D体积目录
real_volume_root = '/workspace/data3'  # 真实3D体积根目录


def process_volume_data():
    # 遍历 multiview_data 目录下的每个样本集文件夹
    for sample_dir in os.listdir(multiview_data_root):
        multiview_sample_path = os.path.join(multiview_data_root, sample_dir)

        # 确保这是一个目录
        if os.path.isdir(multiview_sample_path):
            print(f"Processing sample: {sample_dir}")

            # 对应的真实3D体积目录
            real_volume_path = os.path.join(real_volume_root, sample_dir, 'OBJ', 'mesh')

            # 真实体积的 volumes.json 文件路径
            real_volumes_file = os.path.join(real_volume_path, 'volumes.json')

            # 检查真实体积的 volumes.json 文件是否存在
            if not os.path.exists(real_volumes_file):
                print(f"Warning: {real_volumes_file} not found. Skipping {sample_dir}.")
                continue

            # 读取真实体积的 volumes.json 文件
            with open(real_volumes_file, 'r') as f:
                real_volumes = json.load(f)

            # 遍历样本集目录下的帧文件夹
            for frame_dir in os.listdir(multiview_sample_path):
                frame_path = os.path.join(multiview_sample_path, frame_dir)

                # 确保这是一个目录
                if os.path.isdir(frame_path):
                    # 从帧文件夹名称中提取帧号 (例如 F0006CL2S1R6_0001 -> 0001)
                    frame_number = frame_dir.split('_')[1]

                    # 构造真实体积的文件名 (例如 fluid_mesh_0001.obj)
                    real_volume_key = f"fluid_mesh_{frame_number}.obj"

                    # 检查真实体积数据中是否有该帧号
                    if real_volume_key in real_volumes.keys():
                        real_volume_value = real_volumes[real_volume_key]

                        # 构造 real_volume.json 的内容，包含 real_volume_key 和对应的体积和 x, y, z 宽度
                        real_volume_data = {
                            real_volume_key: {
                                "volume": real_volume_value.get("volume"),
                                "x_width": real_volume_value.get("x_width"),
                                "y_width": real_volume_value.get("y_width"),
                                "z_width": real_volume_value.get("z_width")
                            }
                        }

                        # 将 real_volume.json 写入到对应的帧目录中
                        real_volume_file = os.path.join(frame_path, 'real_volume.json')
                        with open(real_volume_file, 'w') as f:
                            json.dump(real_volume_data, f, indent=4)
                        print(f"Generated {real_volume_file} for frame {frame_number}.")
                    else:
                        print(f"Warning: No real volume data found for frame {frame_number} in {sample_dir}.")


if __name__ == "__main__":
    process_volume_data()
