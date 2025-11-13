import numpy as np
import cv2
from PIL import Image


def extract_foreground_mask(image, edge_detection=True):
    """
    提取图像中的前景掩码。使用边缘检测和 GrabCut 算法。
    返回一个二值掩码，前景为1，背景为0。
    """
    image_array = np.array(image)

    # 转换为 BGR 格式以便使用 OpenCV
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    if edge_detection:
        # 使用 Canny 边缘检测
        gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_image, threshold1=50, threshold2=150)

        # 扩展边缘以确保前景区域被完全包含
        kernel = np.ones((5, 5), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)

        # 创建初始掩码。我们将背景标记为 0，前景标记为 1，未知区域标记为 2。
        mask = np.zeros(image_bgr.shape[:2], np.uint8)
        mask[edges_dilated == 255] = 2  # 未知区域
        mask[edges_dilated == 0] = 0  # 背景区域

        # 创建背景模型和前景模型（必须是浮点型的数组）
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        # 应用 GrabCut 算法
        rect = (1, 1, image_bgr.shape[1] - 1, image_bgr.shape[0] - 1)  # 初始化矩形
        cv2.grabCut(image_bgr, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)

        # 将前景和可能的前景区域标记为 1，其他区域标记为 0
        foreground_mask = np.where((mask == 1) | (mask == 3), 1, 0).astype(np.uint8)
    else:
        # 如果不使用边缘检测，直接使用颜色阈值（假设背景是接近灰色）
        lower_gray = np.array([100, 100, 100])
        upper_gray = np.array([160, 160, 160])
        mask = cv2.inRange(image_bgr, lower_gray, upper_gray)
        foreground_mask = np.where(mask == 0, 1, 0).astype(np.uint8)

    return foreground_mask


def calculate_iou(mask1, mask2):
    """
    计算两个二值掩码的 IoU。
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0.0  # 如果没有前景，IoU 为 0

    return intersection / union


def calculate_average_iou(multiview_image_path, pixel_image_path, num_views=6, image_size=(256, 256)):
    """
    计算 multiview_images_new.png 和 pixel_images.png 中每个子图的平均 IoU。
    """
    # 加载图像
    multiview_image = Image.open(multiview_image_path)
    pixel_image = Image.open(pixel_image_path)

    # 分辨率检查
    assert multiview_image.size == pixel_image.size, "两张图像的尺寸不一致"
    total_width, total_height = multiview_image.size
    assert total_width == num_views * image_size[0], "图像宽度不符合拼接的子图数量"

    # 初始化 IoU 列表
    ious = []

    # 对每个子图计算 IoU
    for i in range(num_views):
        # 计算子图的起始和结束位置
        left = i * image_size[0]
        right = left + image_size[0]
        top = 0
        bottom = image_size[1]

        # 裁剪出子图
        multiview_subimage = multiview_image.crop((left, top, right, bottom))
        pixel_subimage = pixel_image.crop((left, top, right, bottom))

        # 提取前景掩码
        multiview_mask = extract_foreground_mask(multiview_subimage)
        pixel_mask = extract_foreground_mask(pixel_subimage)

        # 计算 IoU
        iou = calculate_iou(multiview_mask, pixel_mask)
        ious.append(iou)

    # 计算平均 IoU
    average_iou = np.mean(ious)

    return average_iou, ious


if __name__ == "__main__":
    multiview_image_path = "multiview_images_new.png"
    pixel_image_path = "pixel_images.png"

    average_iou, ious = calculate_average_iou(multiview_image_path, pixel_image_path)

    print(f"Average IoU: {average_iou}")
    for idx, iou in enumerate(ious):
        print(f"IoU for view {idx + 1}: {iou}")
