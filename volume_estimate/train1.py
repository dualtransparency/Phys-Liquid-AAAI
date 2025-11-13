import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
# from model3 import MultiViewViT
from model4 import MultiViewViT
from dataset import MultiViewDataset
import wandb

# 初始化wandb
wandb.init(project="multiview-vit-regression", entity="352453347-huazhong-university-of-science-and-technology")


# 定义训练函数，添加梯度累积
def train(model, dataloader, criterion, optimizer, device, accumulation_steps=8):
    model.train()
    running_loss = 0.0
    total_relative_error = 0.0
    optimizer.zero_grad()  # 在外部初始化梯度
    batch_idx = 0
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs.squeeze(), labels) / accumulation_steps  # 将损失除以累积步数
        loss.backward()  # 反向传播，累积梯度
        # 计算相对误差
        relative_error = torch.mean(torch.abs((outputs.squeeze() - labels) / labels)).item()
        running_loss += loss.item() * accumulation_steps  # 恢复原始损失
        total_relative_error += relative_error
        # 每 accumulation_steps 次更新一次权重
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()  # 更新权重
            optimizer.zero_grad()  # 清空梯度
    # 如果最后的 batch 数不足 accumulation_steps，也要更新梯度
    if (batch_idx + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    return running_loss / len(dataloader), total_relative_error / len(dataloader)



def validate(model, dataloader, criterion, device):
    model.eval()  # 设置模型为评估模式
    running_loss = 0.0
    total_relative_error = 0.0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            relative_error = torch.mean(torch.abs((outputs.squeeze() - labels) / labels)).item()
            running_loss += loss.item()
            total_relative_error += relative_error
    return running_loss / len(dataloader), total_relative_error / len(dataloader)


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """加载模型、优化器和调度器的状态"""
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path)

        # 加载模型的权重
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model weights loaded from '{checkpoint_path}'")

        # 加载优化器和调度器的状态（如果存在）
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer state loaded.")
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Scheduler state loaded.")

        # 返回保存的 epoch 和验证集最优误差
        return checkpoint['epoch'], checkpoint['best_val_error']
    else:
        print(f"No checkpoint found at '{checkpoint_path}'")
        return 0, float('inf')  # 如果没有找到checkpoint，返回epoch=0，best_val_error为无穷大


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MultiViewViT(freeze_vit=True, num_unfreeze_layers=6).to(device)
    # model = MultiViewCNN(num_views=6, feature_dim=512).to(device)

    # 定义多个数据集路径
    root_dirs = [
        '/workspace/crm_multiview_new/data1/',
        '/workspace/crm_multiview_new/data2/',
        '/workspace/crm_multiview_new/data3/',
        '/workspace/crm_multiview_new/data4/'
    ]

    # 定义图像预处理
    from torchvision import transforms

    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])

    # 创建多个数据集并拼接
    datasets = [MultiViewDataset(root_dir=root_dir, transform=transform, label_mode=4) for root_dir in root_dirs]
    combined_dataset = ConcatDataset(datasets)  # 使用 ConcatDataset 来组合多个数据集
    print("Dataset size:", len(combined_dataset))
    # 划分训练集和验证集（90% 训练集，10% 验证集）
    train_size = int(0.9 * len(combined_dataset))
    val_size = len(combined_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(combined_dataset, [train_size, val_size])

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    # criterion = nn.SmoothL1Loss()
    # optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    # 定义不同层的学习率
    optimizer = torch.optim.Adam([
        {'params': [param for name, param in model.vit.named_parameters() if param.requires_grad], 'lr': 1e-5},
        {'params': model.view_transformer.parameters(), 'lr': 1e-4},
        {'params': model.fc.parameters(), 'lr': 1e-4}
    ])

    # 定义学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, min_lr=1e-5,
                                                           verbose=True)

    # 定义训练参数
    num_epochs = 500  # 训练轮数
    checkpoint_dir = "/workspace/CRM/volume_estimate/multiviewvit_xyz_checkpoints_no_pos"  # 定义checkpoint目录
    os.makedirs(checkpoint_dir, exist_ok=True)  # 创建目录

    # 控制是否从头训练还是接续训练
    resume_training = False  # 设置为True则接续训练，否则从头开始
    checkpoint_path = os.path.join(checkpoint_dir, "model_epoch_116_val_error_0.2060.pth")  # 默认的checkpoint路径

    if resume_training:
        # 继续训练，加载checkpoint
        start_epoch, best_val_error = load_checkpoint(model, optimizer, scheduler, checkpoint_path)
    else:
        # 从头开始训练
        start_epoch = 0
        best_val_error = float('inf')
    # 开始训练
    for epoch in range(start_epoch, num_epochs):
        # 训练阶段
        train_loss, train_relative_error = train(model, train_loader, criterion, optimizer, device,
                                                 accumulation_steps=8)
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Relative Error: {train_relative_error:.4f}')

        # 验证阶段
        val_loss, val_relative_error = validate(model, val_loader, criterion, device)
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Val Loss: {val_loss:.4f}, Val Relative Error: {val_relative_error:.4f}')

        # 调整学习率
        scheduler.step(val_loss)  # 使用验证集损失来调整学习率

        # 记录到wandb
        wandb.log({
            'Train Loss': train_loss,
            'Train Relative Error': train_relative_error,
            'Val Loss': val_loss,
            'Val Relative Error': val_relative_error,
            'Epoch': epoch + 1,
            'Learning Rate': optimizer.param_groups[0]['lr']  # 记录当前学习率
        })

        # 保存模型（如果验证相对误差更低），并防止覆盖
        if val_relative_error < best_val_error and val_relative_error <= 0.3:
            best_val_error = val_relative_error
            # 动态生成文件名，包含 epoch 和验证相对误差
            checkpoint_path = os.path.join(checkpoint_dir,
                                           f"model_epoch_{epoch + 1}_val_error_{val_relative_error:.4f}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_error': best_val_error,
            }, checkpoint_path)
            print(f"Model saved at epoch {epoch + 1} with validation relative error {val_relative_error:.4f}")

    # 在最后一个epoch无论验证误差如何都保存模型
    final_checkpoint_path = os.path.join(checkpoint_dir, f"model_final_epoch_{num_epochs}.pth")
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_error': best_val_error,
    }, final_checkpoint_path)
    print(f"Model saved at the final epoch {num_epochs}.")

    print("Training finished.")
    wandb.finish()  # 结束wandb记录

