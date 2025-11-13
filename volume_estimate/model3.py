import torch
import torch.nn as nn
from transformers import ViTModel


class MultiViewViT(nn.Module):
    def __init__(self, num_views=6, vit_model_name='/workspace/model/vit-large-patch16-224', freeze_vit=True, num_unfreeze_layers=4):
        super(MultiViewViT, self).__init__()
        # 使用预训练的 ViT Large 模型
        self.vit = ViTModel.from_pretrained(vit_model_name)

        # 冻结 ViT 的权重（默认冻结全部权重）
        if freeze_vit:
            for param in self.vit.parameters():
                param.requires_grad = False

        # 解冻 ViT 的最后几层 Transformer block
        if freeze_vit and num_unfreeze_layers > 0:
            total_layers = 12  # ViT Large 有 12 个 Transformer blocks
            for name, param in self.vit.named_parameters():
                # 解冻最后 num_unfreeze_layers 层
                for i in range(total_layers - num_unfreeze_layers, total_layers):
                    if f'encoder.layer.{i}' in name:
                        param.requires_grad = True

        # 特征向量的维度（ViT Large 的输出维度是 1024）
        self.feature_dim = self.vit.config.hidden_size

        # 可学习的位置编码
        self.position_encoding = nn.Parameter(torch.randn(num_views, self.feature_dim))

        # Transformer Encoder 来融合多视角特征
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.feature_dim, nhead=8)
        self.view_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # 优化后的全连接层
        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
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

        # 添加位置编码 (batch_size, num_views, feature_dim)
        view_features = view_features + self.position_encoding

        # 使用 Transformer encoder 进行特征融合
        fused_features = self.view_transformer(view_features)

        # 对多视角特征进行平均池化 (batch_size, feature_dim)
        fused_features = fused_features.mean(dim=1)

        # 通过全连接层回归体积比值
        output = self.fc(fused_features)

        return output

# # 测试网络
# if __name__ == "__main__":
#     # 初始化模型，默认冻结ViT的权重，解冻最后两层
#     model = MultiViewViT(freeze_vit=True)
#     # 假设有 batch_size = 4 的数据，每个样本有 6 张 224x224 的图像
#     dummy_input = torch.randn(4, 6, 3, 224, 224)
#     # 前向传播
#     output = model(dummy_input)
#     # 输出形状应为 (batch_size, 1)
#     print(output.shape)  # 预期输出: torch.Size([4, 1])
