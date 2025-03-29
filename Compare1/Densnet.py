import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


class DenseNetFeatureExtractor(nn.Module):
    def __init__(self, model_type='densenet121', feature_dim=1024):
        super().__init__()
        # 加载预训练模型
        self.base_model = getattr(models, model_type)(pretrained=True)

        # 特征提取部分（保留原始特征层）
        self.features = self.base_model.features

        # 自适应池化层
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 自动计算最终特征维度
        self._feature_dim = self._get_feature_dim()
        self.fc = nn.Linear(self._feature_dim, feature_dim)

    def _get_feature_dim(self):
        """计算池化后的特征维度"""
        with torch.no_grad():
            test_input = torch.randn(1, 3, 640, 480)
            features = self.features(test_input)
            pooled = self.adaptive_pool(features)
            return pooled.view(1, -1).shape[1]  # 正确计算展平后的维度

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # 形状变为 [batch_size, feature_dim]
        return self.fc(x)


# 数据预处理保持不变
def create_densenet_transform():
    return transforms.Compose([
        transforms.Resize((640, 480)),
        transforms.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def get(path):

    image_path=path

    config = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'model_type': 'densenet121',
        'feature_dim': 9,
        'image_path': path
    }

    try:
        # 初始化模型
        model = DenseNetFeatureExtractor(
            model_type=config['model_type'],
            feature_dim=config['feature_dim']
        ).to(config['device']).eval()

        # 数据预处理
        transform = create_densenet_transform()
        img = Image.open(config['image_path']).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).to(config['device'])

        # 特征提取
        with torch.no_grad():
            features = model(input_tensor)

        print(f"成功提取特征，维度：{features.shape}")  # 正确输出：torch.Size([1, 1024])

    except Exception as e:
        print(f"错误发生：{str(e)}")

    return features.tolist()[0]

if __name__ == "__main__":

    # 处理示例图像
    image_path = "1.png"
    print(get(image_path))
