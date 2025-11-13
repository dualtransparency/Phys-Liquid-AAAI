import os
import numpy as np
import csv
import hashlib
from PIL import Image
import cv2

def filter_largest_region(alpha_channel, min_size=500):
    """
    只保留面积最大的连通区域，去除所有其他小的噪点。
    使用OpenCV的连通区域分析来过滤掉小的噪声。
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(alpha_channel, connectivity=8)

    if num_labels <= 1:
        return alpha_channel

    filtered_image = np.zeros_like(alpha_channel, dtype=np.uint8)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_size:
            filtered_image[labels == i] = 255

    return filtered_image


def crop_and_resize_image(image_path, output_path, target_size=(512, 512), object_ratio=0.9, min_region_size=500, alpha_threshold=128):
    with Image.open(image_path) as img:
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        img_array = np.array(img)
        alpha_channel = img_array[:, :, 3]
        alpha_channel = np.where(alpha_channel > alpha_threshold, 255, 0).astype(np.uint8)
        largest_alpha = filter_largest_region(alpha_channel, min_size=min_region_size)
        non_transparent = np.where(largest_alpha > 0)

        if len(non_transparent[0]) == 0 or len(non_transparent[1]) == 0:
            resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
            resized_img.save(output_path, "PNG")
            return

        top, bottom = np.min(non_transparent[0]), np.max(non_transparent[0])
        left, right = np.min(non_transparent[1]), np.max(non_transparent[1])
        cropped_img = img.crop((left, top, right, bottom))

        target_width = int(target_size[0] * object_ratio)
        target_height = int(target_size[1] * object_ratio)
        width, height = cropped_img.size
        scale_w = target_width / width
        scale_h = target_height / height
        scale = min(scale_w, scale_h)

        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_img = cropped_img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        new_img = Image.new("RGBA", target_size, (255, 255, 255, 0))
        x_pos = (target_size[0] - new_width) // 2
        y_pos = (target_size[1] - new_height) // 2
        new_img.paste(resized_img, (x_pos, y_pos), resized_img)
        new_img.save(output_path, "PNG")


def create_output_structure(input_folders, output_folder, target_size=(512, 512), object_ratio=0.9,
                            min_region_size=500, err_csv_path="err.csv"):
    os.makedirs(output_folder, exist_ok=True)
    csv_path = os.path.join(output_folder, 'caption.csv')

    if not os.path.exists(csv_path):
        with open(csv_path, mode='w', newline='') as csv_file:
            fieldnames = ['id', 'caption']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

    processed_samples = set()

    with open(csv_path, mode='a', newline='') as csv_file, open(err_csv_path, mode='w', newline='') as err_file:
        writer = csv.DictWriter(csv_file, fieldnames=['id', 'caption'])
        err_writer = csv.DictWriter(err_file, fieldnames=['sample_name', 'missing_views', 'source_folder'])
        err_writer.writeheader()

        required_views = {f'{i:03d}' for i in range(6)}  # 000 ~ 005

        for input_folder in input_folders:
            sample_views = {}

            for filename in os.listdir(input_folder):
                if filename.endswith('.png'):
                    parts = filename.split('_')
                    sample_name = f"{parts[0]}_{parts[1]}"
                    view_id = parts[-1].split('.')[0]

                    if sample_name not in sample_views:
                        sample_views[sample_name] = set()
                    sample_views[sample_name].add(view_id)

            for sample_name, views in sample_views.items():
                missing_views = required_views - views

                if missing_views:
                    err_writer.writerow({
                        'sample_name': sample_name,
                        'missing_views': ','.join(missing_views),
                        'source_folder': input_folder
                    })
                    continue

                sample_hash = hashlib.md5(sample_name.encode()).hexdigest()
                sample_folder = os.path.join(output_folder, sample_hash)
                os.makedirs(sample_folder, exist_ok=True)

                for view_id in sorted(views):
                    image_path = os.path.join(input_folder, f"{sample_name}_{view_id}.png")
                    output_image_path = os.path.join(sample_folder, f"{view_id}.png")
                    crop_and_resize_image(image_path, output_image_path, target_size=target_size,
                                          object_ratio=object_ratio, min_region_size=min_region_size)

                if sample_hash not in processed_samples:
                    caption = f"A high-quality 3D transparent liquid model."
                    writer.writerow({'id': sample_hash, 'caption': caption})
                    processed_samples.add(sample_hash)


def find_liquid_dirs(root_dirs):
    liquid_dirs = []
    for root_dir in root_dirs:
        for subdir, dirs, files in os.walk(root_dir):
            if os.path.basename(subdir).startswith('F00') and 'liquid' in dirs:
                liquid_path = os.path.join(subdir, 'liquid')
                liquid_dirs.append(liquid_path)
    return liquid_dirs


if __name__ == '__main__':
    root_dirs = [
        r'/workspace/data1',
        r'/workspace/data2',
        r'/workspace/data3',
        r'/workspace/data4',
    ]

    input_folders = find_liquid_dirs(root_dirs)
    output_folder = r'/workspace/CRM/train_data/'
    err_csv_path = os.path.join(output_folder, 'err.csv')

    os.makedirs(output_folder, exist_ok=True)

    create_output_structure(input_folders, output_folder, target_size=(512, 512), object_ratio=0.9, min_region_size=500, err_csv_path=err_csv_path)
