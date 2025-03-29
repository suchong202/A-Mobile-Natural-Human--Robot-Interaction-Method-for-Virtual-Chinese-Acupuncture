import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


class EfficientNetFeatureExtractor(nn.Module):
    def __init__(self, model_name='efficientnet_b0', feature_dim=1024):
        super().__init__()

        # 加载预训练模型
        self.base_model = getattr(models.efficientnet, model_name)(pretrained=True)

        # 特征提取部分（保留除分类层外的所有层）
        self.features = nn.Sequential(
            self.base_model.features,
            self.base_model.avgpool
        )

        # 动态计算特征维度
        self.feature_dim = self._calculate_feature_dim()
        self.fc = nn.Linear(self.feature_dim, feature_dim)

    def _calculate_feature_dim(self):
        """自动计算特征维度"""
        with torch.no_grad():
            test_input = torch.randn(1, 3, 640, 480)
            features = self.features(test_input)
            return features.view(1, -1).shape[1]

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 展平特征
        return self.fc(x)


def create_efficientnet_transform():
    return transforms.Compose([
        transforms.Resize((640, 480)),
        transforms.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def visualize_processing(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = create_efficientnet_transform()

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(img)
    ax[0].set_title(f'Original Image ({img.size[0]}x{img.size[1]})')

    # 处理后的张量可视化
    tensor_img = transform(img).numpy().transpose(1, 2, 0)
    tensor_img = np.clip(tensor_img, 0, 1)
    ax[1].imshow(tensor_img)
    ax[1].set_title('Processed Tensor')
    plt.show()


def visualize_feature_distribution(features):
    plt.figure(figsize=(10, 6))
    plt.hist(features.cpu().numpy().flatten(), bins=50, color='green', alpha=0.7)
    plt.title('EfficientNet Feature Distribution')
    plt.xlabel('Feature Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    #plt.show()


# 版本兼容组件 ---------------------------------------------------
class EfficientNetCompat(nn.Module):
    """兼容不同PyTorch版本"""

    def __init__(self, model_name='efficientnet_b0'):
        super().__init__()
        self.model = models.efficientnet.__dict__[model_name](pretrained=True)
        self.features = nn.Sequential(
            self.model.features,
            self.model.avgpool
        )

    def forward(self, x):
        return self.features(x)


# 多尺度特征支持 -------------------------------------------------
class MultiScaleEfficientNet(EfficientNetFeatureExtractor):
    def __init__(self, scales=[1.0, 0.5, 0.25], **kwargs):
        super().__init__(**kwargs)
        self.scales = scales

    def forward(self, x):
        features = []
        for scale in self.scales:
            scaled_x = nn.functional.interpolate(x, scale_factor=scale)
            feat = super().forward(scaled_x)
            features.append(feat)
        return torch.cat(features, dim=1)



def get(path):

    image_path=path
    config = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'model_name': 'efficientnet_b0',  # 可选b1-b7
        'feature_dim': 9,
        'image_path': path
    }

    try:
        # 初始化模型
        model = EfficientNetFeatureExtractor(
            model_name=config['model_name'],
            feature_dim=config['feature_dim']
        ).to(config['device']).eval()

        # 数据预处理
        transform = create_efficientnet_transform()
        img = Image.open(config['image_path'])
        input_tensor = transform(img).unsqueeze(0).to(config['device'])

        # 特征提取
        with torch.no_grad():
            features = model(input_tensor)

        # 结果展示
        print(f"特征维度: {features.shape}")  # 输出示例: torch.Size([1, 1024])
        visualize_feature_distribution(features)

        # 保存特征
        torch.save({
            'features': features.cpu(),
            'model_name': config['model_name']
        }, './Model/efficientnet_features.pth')

    except Exception as e:
        print(f"错误发生: {str(e)}")
        if 'img' in locals():
            print(f"图像模式: {img.mode}, 尺寸: {img.size}")

    return features.tolist()[0]

if __name__ == "__main__":

    # 处理示例图像
    image_path = "1.png"
    print(get(image_path))