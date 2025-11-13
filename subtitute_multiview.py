import os
from PIL import Image
import numpy as np

# GT图片的根目录可以是data1到data4
GT_ROOT_DIRS = ["/workspace/data1", "/workspace/data2", "/workspace/data3", "/workspace/data4"]
GT_SUB_DIR = "sam_data"
GT_CROPPED_IMAGES_DIR = "cropped_images"

# 目标拼接顺序
VIEW_ORDER = [1, 2, 3, 4, 5, 0]  # 对应视角[001, 002, 003, 004, 005, 000]

# 替换的视角
TARGET_VIEWS = [4, 5]  # 对应视角[004, 005]

# 灰色背景值
GRAY_BG = np.array([127, 127, 127])

# 白色检测阈值
WHITE_COLOR = np.array([255, 255, 255])
WHITE_THRESHOLD = 240  # 定义接近白色的阈值

def color_distance(c1, c2):
    """
    计算两个颜色之间的欧几里得距离，用于判断颜色相似度
    :param c1: 颜色1 (RGB)
    :param c2: 颜色2 (RGB)
    :return: 欧几里得距离
    """
    return np.sqrt(np.sum((c1 - c2) ** 2, axis=-1))

def replace_gt_in_pixel_images(sample_dir):
    """
    替换pixel_images.png中的指定视角为GT图片，并保存为pixel_images_new.png
    :param sample_dir: 样本文件夹路径
    """
    for root, dirs, files in os.walk(sample_dir):
        # 第一层文件夹是样本名（例如 F0002PL8S3R5）
        for sample_name in dirs:
            sample_path = os.path.join(root, sample_name)

            # 进入第二层文件夹（例如 F0002PL8S3R5_0066_000）
            for sub_dir in os.listdir(sample_path):
                sub_dir_path = os.path.join(sample_path, sub_dir)

                # 确保是文件夹
                if not os.path.isdir(sub_dir_path):
                    continue

                try:
                    # 提取样本名、帧数和视角名
                    sample_name, frame_num, _ = sub_dir.split('_')
                except ValueError:
                    print(f"Skipping invalid directory structure: {sub_dir}")
                    continue

                # 构建pixel_images.png的路径
                pixel_image_path = os.path.join(sub_dir_path, "pixel_images.png")
                if not os.path.exists(pixel_image_path):
                    print(f"pixel_images.png not found in {sub_dir_path}")
                    continue

                # 打开pixel_images.png
                pixel_image = Image.open(pixel_image_path)

                # 遍历所有需要替换的视角
                for view in TARGET_VIEWS:
                    # GT的文件名格式：样本名_帧数_视角.png
                    gt_filename = f"{sample_name}_{frame_num}_{view:03d}.png"

                    # 在所有GT根目录中查找GT图片
                    gt_image_path = None
                    for gt_root in GT_ROOT_DIRS:
                        potential_gt_path = os.path.join(gt_root, GT_SUB_DIR, sample_name, GT_CROPPED_IMAGES_DIR, gt_filename)
                        if os.path.exists(potential_gt_path):
                            gt_image_path = potential_gt_path
                            break

                    if gt_image_path is None:
                        print(f"GT image not found for view {view} in {sample_name}_{frame_num}")
                        continue

                    # 打开GT图片
                    gt_image = Image.open(gt_image_path)

                    # 将GT图片缩小到256x256
                    gt_image_resized = gt_image.resize((256, 256))

                    # 将GT图片转换为numpy数组
                    gt_image_np = np.array(gt_image_resized)

                    # 计算每个像素与白色的距离
                    distances_to_white = color_distance(gt_image_np, WHITE_COLOR)

                    # 创建一个掩码，检测接近白色的像素
                    mask = distances_to_white < (255 - WHITE_THRESHOLD)

                    # 对接近白色的像素进行渐进替换
                    alpha = distances_to_white[mask] / (255 - WHITE_THRESHOLD)  # 计算透明度比例
                    alpha = alpha[:, np.newaxis]  # 扩展维度以适应RGB通道

                    # 使用alpha通道进行渐进替换，避免硬边
                    gt_image_np[mask] = (1 - alpha) * GRAY_BG + alpha * gt_image_np[mask]

                    # 将修改后的图像转换回PIL格式
                    gt_image_modified = Image.fromarray(gt_image_np.astype('uint8'))

                    # 根据视角找到在pixel_images中的位置
                    view_index = VIEW_ORDER.index(view)
                    x_offset = view_index * 256

                    # 将GT图片粘贴到pixel_images的对应位置
                    pixel_image.paste(gt_image_modified, (x_offset, 0))

                # 保存修改后的图片为pixel_images_new.png
                new_pixel_image_path = os.path.join(sub_dir_path, "pixel_images_new.png")
                pixel_image.save(new_pixel_image_path)
                print(f"Updated pixel_images_new.png for {sub_dir_path}")

if __name__ == "__main__":
    # 替换所有样本文件夹中的pixel_images.png
    sample_root_dir = "/workspace/crm_oneview_multiview3"
    replace_gt_in_pixel_images(sample_root_dir)
