import os
import json
import torch
from PIL import Image
from torchvision import transforms
from model3 import MultiViewViT

# 1. 定义输入和输出文件名变量
input_image_filename = 'pixel_images_new.png'  # 输入文件名
output_inference_filename = 'inference.json'  # 输出文件名

# 2. 设置设备和加载模型
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = MultiViewViT(freeze_vit=True, num_unfreeze_layers=6).to(device)

# 加载模型权重
checkpoint_path = '/workspace/CRM/volume_estimate/multiviewvit_xyz_checkpoints/model_epoch_483_val_error_0.0626.pth'
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 3. 定义图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整大小
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

# 4. 定义推理函数
def infer_single_image(image_path, model, device, label_mode):
    # 加载并切分输入图像
    multiview_image = Image.open(image_path)
    views = []
    for i in range(6):
        view = multiview_image.crop((i * 256, 0, (i + 1) * 256, 256))  # 切分出每张 256x256 的图片
        view = transform(view)  # 应用预处理
        views.append(view)

    # 将视角图像堆叠为 (6, 3, 224, 224) 的张量
    views = torch.stack(views, dim=0).unsqueeze(0)  # 增加 batch 维度，形状为 (1, 6, 3, 224, 224)

    # 推理
    views = views.to(device)
    with torch.no_grad():
        output = model(views)  # 输出形状为 (1, 3) 或 (1,) 取决于 label_mode

    # 根据 label_mode 处理输出
    if label_mode == 1:
        volume_ratio = output.squeeze().cpu().item()  # 转换为标量
        return {"volume_ratio": float(volume_ratio)}

    elif label_mode == 2:
        x_ratio, y_ratio, z_ratio = output.squeeze().cpu().numpy()  # 转换为 numpy 数组
        return {
            "x_ratio": float(x_ratio),
            "y_ratio": float(y_ratio),
            "z_ratio": float(z_ratio)
        }

    elif label_mode == 3:
        xyz_ratio = output.squeeze().cpu().numpy()
        return {"xyz_ratio_arithmetic": float(xyz_ratio)}

    elif label_mode == 4:
        xyz_ratio = output.squeeze().cpu().numpy()
        return {"xyz_ratio_geometric": float(xyz_ratio)}

    else:
        raise ValueError(f"Invalid label_mode: {label_mode}")

# 5. 保存推理结果，避免覆盖已存在的数据
def save_inference_results(folder_path, inference_result, output_filename):
    inference_path = os.path.join(folder_path, output_filename)
    # 如果文件已经存在，读取现有内容
    if os.path.exists(inference_path):
        with open(inference_path, 'r') as f:
            existing_data = json.load(f)
    else:
        existing_data = {}
    # 更新现有内容，不覆盖
    existing_data.update(inference_result)
    # 写入更新后的数据
    with open(inference_path, 'w') as f:
        json.dump(existing_data, f, indent=4)

# 6. 遍历文件夹进行批量推理
root_dir = '/root/CRM/out_old/'
label_modes = [4]

# 遍历根目录下的所有一级文件夹
for folder in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder)

    # 检查是否是一级文件夹
    if os.path.isdir(folder_path):
        # 遍历一级文件夹中的所有子文件夹
        for subfolder in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder)

            # 检查是否是子文件夹
            if os.path.isdir(subfolder_path):
                image_path = os.path.join(subfolder_path, input_image_filename)

                # 检查输入图像文件是否存在
                if os.path.exists(image_path):
                    for label_mode in label_modes:
                        # 进行推理
                        inference_result = infer_single_image(image_path, model, device, label_mode)

                        # 保存结果，不覆盖之前模式的结果
                        save_inference_results(subfolder_path, inference_result, output_inference_filename)

                    print(f"Inference results saved to {subfolder_path}/{output_inference_filename}")
                else:
                    print(f"{input_image_filename} not found in {subfolder_path}")
