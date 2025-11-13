import os
import json
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
import numpy as np
from torchvision import transforms

class MultiViewDataset(Dataset):
    def __init__(self, root_dir, transform=None, label_mode=1):
        """
        Args:
            root_dir (string): 数据集根目录
            transform (callable, optional): 可选的图像预处理变换
            label_mode (int): 标签生成模式，1-4
        """
        self.root_dir = root_dir
        self.transform = transform
        self.label_mode = label_mode
        self.data = []

        # 遍历根目录下的所有样本集文件夹
        for sample_dir in os.listdir(root_dir):
            sample_path = os.path.join(root_dir, sample_dir)

            if os.path.isdir(sample_path):
                # 遍历样本集文件夹中的每个帧文件夹
                for frame_dir in os.listdir(sample_path):
                    frame_path = os.path.join(sample_path, frame_dir)

                    if os.path.isdir(frame_path):
                        multiview_image_path = os.path.join(frame_path, 'multiview_images.png')
                        volume_json_path = os.path.join(frame_path, 'volumes.json')
                        real_volume_json_path = os.path.join(frame_path, 'real_volume.json')

                        # 检查 multiview_images.png 是否存在
                        if not os.path.exists(multiview_image_path):
                            continue

                        # 检查 volumes.json 和 real_volume.json 是否存在
                        if not os.path.exists(volume_json_path) or not os.path.exists(real_volume_json_path):
                            continue

                        # 读取 volumes.json 和 real_volume.json
                        with open(volume_json_path, 'r') as f:
                            volume_data = json.load(f)

                        with open(real_volume_json_path, 'r') as f:
                            real_volume_data = json.load(f)

                        # 获取伪3D体积数据 (乘以 1e-6)
                        if "output3d.obj" not in volume_data:
                            continue
                        pseudo_volume_data = volume_data["output3d.obj"]
                        pseudo_volume = pseudo_volume_data.get("volume", 0) * 1e-6

                        # 获取伪3D的 x, y, z 宽度 (乘以 1e-2)
                        pseudo_x = pseudo_volume_data.get("x_width", 0) * 1e-2
                        pseudo_y = pseudo_volume_data.get("y_width", 0) * 1e-2
                        pseudo_z = pseudo_volume_data.get("z_width", 0) * 1e-2

                        # 获取真实3D体积数据
                        frame_number = frame_dir.split('_')[-1]
                        real_volume_key = f"fluid_mesh_{frame_number}.obj"
                        if real_volume_key not in real_volume_data:
                            continue

                        # 先通过 real_volume_key 访问到内部字典
                        real_volume_info = real_volume_data[real_volume_key]

                        # 获取真实的体积和宽度
                        real_volume = real_volume_info["volume"]
                        real_x = real_volume_info["x_width"]
                        real_y = real_volume_info["y_width"]
                        real_z = real_volume_info["z_width"]

                        # 根据 label_mode 计算标签
                        if self.label_mode == 1:
                            # ① 体积比值
                            label = pseudo_volume / real_volume
                        elif self.label_mode == 2:
                            # ② x、y、z 比例值
                            label = {
                                "x_ratio": pseudo_x / real_x,
                                "y_ratio": pseudo_y / real_y,
                                "z_ratio": pseudo_z / real_z
                            }
                            # fake * 1e-2 / real = ratio
                        elif self.label_mode == 3:
                            # ③ 三个比例值的算术平均值
                            x_ratio = pseudo_x / real_x
                            y_ratio = pseudo_y / real_y
                            z_ratio = pseudo_z / real_z
                            label = (x_ratio + y_ratio + z_ratio) / 3
                        elif self.label_mode == 4:
                            # ④ 三个比例值的几何平均值
                            x_ratio = pseudo_x / real_x
                            y_ratio = pseudo_y / real_y
                            z_ratio = pseudo_z / real_z
                            label = (x_ratio * y_ratio * z_ratio) ** (1 / 3)
                        else:
                            raise ValueError("Invalid label_mode. Must be 1, 2, 3, or 4.")

                        # 将样本添加到数据集中
                        self.data.append({
                            'multiview_image_path': multiview_image_path,
                            'label': label
                        })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取样本数据
        sample = self.data[idx]
        multiview_image_path = sample['multiview_image_path']
        label = sample['label']

        # 读取 multiview_images.png
        multiview_image = Image.open(multiview_image_path)

        # 将 multiview_images.png 切分为 6 张 256x256 的图片
        # multiview_image.size = (1536, 256)
        views = []
        for i in range(6):
            view = multiview_image.crop((i * 256, 0, (i + 1) * 256, 256))  # 切分出每张 256x256 的图片
            if self.transform:
                view = self.transform(view)  # 应用预处理
            views.append(view)

        # 将 views 转换为形状为 (6, 3, 256, 256) 的张量
        views = torch.stack(views, dim=0)

        # 如果 label 是字典（第二种情况），将其转换为张量
        if isinstance(label, dict):
            label = torch.tensor([label["x_ratio"], label["y_ratio"], label["z_ratio"]], dtype=torch.float32)
        else:
            label = torch.tensor(label, dtype=torch.float32)

        return views, label


if __name__ == '__main__':

    # 定义 4 个数据集的根目录
    root_dirs = [
        '/workspace/crm_multiview_new/data1/',
        '/workspace/crm_multiview_new/data2/',
        '/workspace/crm_multiview_new/data3/',
        '/workspace/crm_multiview_new/data4/'
    ]

    # 定义图像预处理
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为 Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])

    # 创建 4 个独立的数据集
    label_mode = 1  # 可以选择 1, 2, 3, 4
    datasets = [MultiViewDataset(root_dir=root_dir, transform=transform, label_mode=label_mode) for root_dir in root_dirs]

    # 使用 ConcatDataset 将 4 个数据集拼接
    combined_dataset = ConcatDataset(datasets)

    # 打印拼接后的数据集长度
    print("len of combined dataset:", len(combined_dataset))

    # 创建 DataLoader
    dataloader = DataLoader(combined_dataset, batch_size=4, shuffle=True, num_workers=4)

    # 测试 DataLoader
    for batch_idx, (images, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}")
        print("Images shape:", images.shape)  # 预期形状: (batch_size, 6, 3, 256, 256)
        print("Labels:", labels)
        break
