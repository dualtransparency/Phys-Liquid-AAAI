import torch
import torch.nn as nn


class MultiViewCNN(nn.Module):
    def __init__(self, num_views=6, feature_dim=512):
        super(MultiViewCNN, self).__init__()

        # 使用轻量级的CNN来提取每个视角的特征
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # 输入通道为3 (RGB)，输出通道为64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 下采样，特征图尺寸减半

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 下采样

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 下采样

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # 自适应平均池化，输出为 (512, 1, 1)
        )

        # Transformer Encoder 来融合多视角特征
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8)
        self.view_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # 回归的全连接层
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, images):
        """
        images: 形状为 (batch_size, num_views, 3, 224, 224) 的多视角图像输入
        """
        batch_size, num_views, _, _, _ = images.shape
        view_features = []

        # 对每个视角的图像使用CNN提取特征
        for i in range(num_views):
            img = images[:, i, :, :, :]  # 取出第 i 个视角的图像
            cnn_features = self.cnn(img)  # 通过CNN提取特征
            cnn_features = cnn_features.view(batch_size, -1)  # 展平为 (batch_size, feature_dim)
            view_features.append(cnn_features)

        # 将所有视角的特征堆叠为 (batch_size, num_views, feature_dim)
        view_features = torch.stack(view_features, dim=1)

        # 使用 Transformer encoder 进行多视角特征融合
        fused_features = self.view_transformer(view_features)

        # 对多视角特征进行平均池化 (batch_size, feature_dim)
        fused_features = fused_features.mean(dim=1)

        # 通过全连接层回归体积比值
        output = self.fc(fused_features)

        return output


# 测试网络结构
if __name__ == "__main__":
    model = MultiViewCNN(num_views=6, feature_dim=512)
    images = torch.randn(8, 6, 3, 224, 224)  # 模拟一个 batch_size = 8 的输入 (8个样本，每个样本6个视角)
    output = model(images)
    print(output.shape)  # 输出形状应该是 (8, 1)，即每个样本回归一个数值
