import torch
import torch.nn as nn
from transformers import ViTModel


class MultiViewViT(nn.Module):
    def __init__(self, num_views=6, vit_model_name='/workspace/model/vit-large-patch16-224', freeze_vit=True):
        super(MultiViewViT, self).__init__()
        # 使用预训练的 ViT Large 模型
        self.vit = ViTModel.from_pretrained(vit_model_name)

        # 冻结 ViT 的权重（默认冻结全部权重）
        if freeze_vit:
            for param in self.vit.parameters():
                param.requires_grad = False

        # 解冻 ViT 的最后几层 Transformer block
        for name, param in self.vit.named_parameters():
            if 'encoder.layer.11' in name or 'encoder.layer.10' in name:
                param.requires_grad = True

        # 特征向量的维度（ViT Large 的输出维度是 1024）
        self.feature_dim = self.vit.config.hidden_size
        # 初始化为1，形状为(num_views,)
        self.view_weights = nn.Parameter(torch.ones(num_views))
        # 自注意力机制进行特征融合
        self.attention1 = nn.MultiheadAttention(embed_dim=self.feature_dim, num_heads=8, batch_first=True)
        self.attention2 = nn.MultiheadAttention(embed_dim=self.feature_dim, num_heads=8, batch_first=True)

        # 优化后的全连接层
        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, images):
        """
        images: 形状为 (batch_size, num_views, 3, 224, 224) 的多视图图像输入
        """
        batch_size, num_views, _, _, _ = images.shape
        view_features = []

        # 对每个视角的图像进行特征提取
        for i in range(num_views):
            img = images[:, i, :, :, :]  # 取出第 i 个视角的图像
            # 通过 ViT 提取特征
            outputs = self.vit(img)
            # 取 [CLS] token 作为特征
            cls_token = outputs.last_hidden_state[:, 0, :]
            view_features.append(cls_token)

        # 将特征堆叠为 (batch_size, num_views, feature_dim)
        view_features = torch.stack(view_features, dim=1)

        # 使用自注意力机制进行特征融合
        fused_features, _ = self.attention1(view_features, view_features, view_features)
        fused_features, _ = self.attention2(fused_features, fused_features, fused_features)

        # 对视角权重进行归一化
        view_weights = torch.softmax(self.view_weights, dim=0)  # 归一化权重

        # 对视角特征加权求和 (batch_size, num_views, feature_dim) -> (batch_size, feature_dim)
        fused_features = torch.einsum('v,bvd->bd', view_weights, fused_features)  # 加权求和

        # 通过全连接层回归体积比值
        output = self.fc(fused_features)

        return output


# 测试网络
if __name__ == "__main__":
    # 初始化模型，默认冻结ViT的权重，解冻最后两层
    model = MultiViewViT(freeze_vit=True)
    # 假设有 batch_size = 4 的数据，每个样本有 6 张 224x224 的图像
    dummy_input = torch.randn(4, 6, 3, 224, 224)
    # 前向传播
    output = model(dummy_input)
    # 输出形状应为 (batch_size, 1)
    print(output.shape)  # 预期输出: torch.Size([4, 1])
