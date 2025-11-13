import json


# 读取 JSON 文件
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


# 写入 JSON 文件
def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


# 将移除的键名和它们的 average_iou 追加到文件
def append_removed_keys(removed_keys, output_file_path):
    with open(output_file_path, 'a') as f:  # 使用 'a' 模式来追加内容
        for key, avg_iou in removed_keys:
            f.write(f"{key}: {avg_iou}\n")


# 处理数据
def process_json(data):
    removed_keys = []
    total_iou_sum = 0
    total_iou_count = 0

    # 复制一份数据以防止在遍历时修改原数据
    filtered_data = {}

    for key, value in data.items():
        # 跳过 overall_average_iou
        if key == "overall_average_iou":
            continue

        average_iou = value.get("average_iou", 0)

        # 如果 average_iou 小于 0.5，剔除该对象并记录键名和其 average_iou
        if average_iou < 0.7:
            removed_keys.append((key, average_iou))
        else:
            # 保留该对象并累加其 file_iou 的值
            filtered_data[key] = value
            for file_name, iou in value.get("file_iou", {}).items():
                total_iou_sum += iou
                total_iou_count += 1

    # 计算新的 overall_average_iou
    if total_iou_count > 0:
        new_overall_average_iou = total_iou_sum / total_iou_count
    else:
        new_overall_average_iou = 0

    # 更新 overall_average_iou
    filtered_data["overall_average_iou"] = new_overall_average_iou

    return filtered_data, removed_keys


# 主函数
def main(input_path, output_path, removed_keys_output_path):
    # 读取原始 JSON 数据
    data = load_json(input_path)

    # 处理数据并获取被移除的键名和 average_iou
    filtered_data, removed_keys = process_json(data)

    # 打印被移除的键名和 average_iou
    print("Removed keys (average_iou < 0.5):")
    for key, avg_iou in removed_keys:
        print(f"{key}: {avg_iou}")

    # 将移除的键名和 average_iou 追加到文件
    append_removed_keys(removed_keys, removed_keys_output_path)

    # 保存处理后的 JSON 数据到新的路径
    save_json(filtered_data, output_path)
    print(f"Processed data saved to {output_path}")
    print(f"Removed keys appended to {removed_keys_output_path}")


# 文件路径
input_json_path = r'E:\datasets\evaluate\5.3 yolo_sam_iou\data4_iou_results.json'  # 原始 JSON 文件路径
output_json_path = r'E:\datasets\evaluate\5.3 yolo_sam_iou\data4_iou_results_new.json'  # 处理后 JSON 文件保存路径
removed_keys_output_path = r'E:\datasets\evaluate\5.3 yolo_sam_iou\removed_keys.txt'  # 被移除的键名保存路径

# 执行主函数
main(input_json_path, output_json_path, removed_keys_output_path)
