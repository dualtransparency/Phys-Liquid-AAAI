import os
import shutil
import pandas as pd
import random

# 定义路径
train_data_dir = '/workspace/CRM/train_data/'
validate_data_dir = '/workspace/CRM/validate_data'
caption_csv_path = os.path.join(train_data_dir, 'caption.csv')

# 确保验证集目录存在
os.makedirs(validate_data_dir, exist_ok=True)

# 读取 caption.csv 文件
df = pd.read_csv(caption_csv_path)

# 随机选择 10% 的数据
validate_size = int(0.1 * len(df))
validate_ids = random.sample(df['id'].tolist(), validate_size)

# 移动文件夹并删除训练集中的数据
for validate_id in validate_ids:
    train_folder = os.path.join(train_data_dir, validate_id)
    validate_folder = os.path.join(validate_data_dir, validate_id)

    # 检查文件夹是否存在
    if os.path.exists(train_folder):
        # 移动文件夹到验证集
        shutil.move(train_folder, validate_folder)
    else:
        print(f"Warning: {train_folder} does not exist.")

# 更新 caption.csv，删除验证集对应的数据
df_train = df[~df['id'].isin(validate_ids)]
df_validate = df[df['id'].isin(validate_ids)]

# 将更新后的训练集 caption.csv 保存回原路径
df_train.to_csv(caption_csv_path, index=False)

# 将验证集的 caption.csv 保存到验证集目录
validate_caption_csv_path = os.path.join(validate_data_dir, 'caption.csv')
df_validate.to_csv(validate_caption_csv_path, index=False)

print(f"Moved {validate_size} samples to {validate_data_dir} and updated caption.csv.")
