import os
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# 定义根目录
root_dirs = [r'E:\datasets\MeshLiquid\data1',
             r'E:\datasets\MeshLiquid\data2',
             r'E:\datasets\MeshLiquid\data3',
             r'E:\datasets\MeshLiquid\data4']

# 获取所有文件夹的路径
folders = []
for root_dir in root_dirs:
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if os.path.isdir(folder_path) and folder_name.startswith('F00'):
            folders.append(folder_path)

# 确保提取100个文件夹
if len(folders) < 100:
    raise ValueError("文件夹数量不足100个")
else:
    selected_folders = random.sample(folders, 100)

# 校验提取的文件夹数量
assert len(selected_folders) == 100, "提取的文件夹数量不等于100"

# 初始化分类统计的字典
stats = {
    'color': defaultdict(int),
    'light': defaultdict(int),
    'scene': defaultdict(int),
    'rotation': defaultdict(int)
}

# 遍历选中的文件夹，进行分类统计
for folder in selected_folders:
    folder_name = os.path.basename(folder)

    # 提取颜色、光线、场景和旋转模式信息
    bottle_number = folder_name[1:5]
    color = folder_name[5]  # 颜色
    light = folder_name[6:8]  # 光线
    scene = folder_name[8:10]  # 场景
    rotation = folder_name[10:12]  # 旋转模式

    # 更新统计
    stats['color'][color] += 1
    stats['light'][light] += 1
    stats['scene'][scene] += 1
    stats['rotation'][rotation] += 1

# 打印统计结果
print("分类统计结果：")
for category, counts in stats.items():
    print(f"{category.capitalize()}统计:")
    for key, count in counts.items():
        print(f"  {key}: {count}")


# 定义绘制子图的函数
def plot_subplots(stats, save_path):
    sns.set(style="whitegrid")

    # 创建一个2x2的子图布局
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 绘制颜色分布
    sns.barplot(ax=axes[0, 0], x=list(stats['color'].keys()), y=list(stats['color'].values()), palette='Set2')
    axes[0, 0].set_title("液体颜色分布")
    axes[0, 0].set_xlabel("颜色")
    axes[0, 0].set_ylabel("数量")

    # 绘制光线分布
    sns.barplot(ax=axes[0, 1], x=list(stats['light'].keys()), y=list(stats['light'].values()), palette='Set2')
    axes[0, 1].set_title("光线分布")
    axes[0, 1].set_xlabel("光线")
    axes[0, 1].set_ylabel("数量")

    # 绘制场景分布
    sns.barplot(ax=axes[1, 0], x=list(stats['scene'].keys()), y=list(stats['scene'].values()), palette='Set2')
    axes[1, 0].set_title("场景分布")
    axes[1, 0].set_xlabel("场景")
    axes[1, 0].set_ylabel("数量")

    # 绘制旋转模式分布
    sns.barplot(ax=axes[1, 1], x=list(stats['rotation'].keys()), y=list(stats['rotation'].values()), palette='Set2')
    axes[1, 1].set_title("旋转模式分布")
    axes[1, 1].set_xlabel("旋转模式")
    axes[1, 1].set_ylabel("数量")

    # 调整布局，避免重叠
    plt.tight_layout()

    # 保存图表到本地文件
    plt.savefig(save_path)
    plt.show()


# 保存图表的路径
save_path = r'E:\datasets\MeshLiquid\folder_statistics.png'

# 调用函数绘制并保存图表
plot_subplots(stats, save_path)

print(f"图表已保存到 {save_path}")
