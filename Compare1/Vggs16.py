import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


class VGG16FeatureExtractor(nn.Module):
    def __init__(self, feature_dim=1024):
        super().__init__()

        # 加载预训练VGG16
        self.base_model = models.vgg16(pretrained=True)

        # 特征提取部分（保留卷积层）
        self.features = nn.Sequential(
            *list(self.base_model.features.children()),
            nn.AdaptiveAvgPool2d((7, 7))  # 统一输出尺寸
        )

        # 自动计算特征维度
        self._feature_dim = 512 * 7 * 7  # VGG16最终特征维度
        self.fc = nn.Linear(self._feature_dim, feature_dim)

    def forward(self, x):
        # 输入验证
        if x.shape[1:] != (3, 640, 480):
            raise ValueError(f"输入尺寸需为[3,640,480]，当前为{x.shape[1:]}")

        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def create_transform():
    return transforms.Compose([
        transforms.Resize((640, 480)),
        transforms.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet标准化参数
            std=[0.229, 0.224, 0.225]
        )
    ])


def visualize_processing(image_path):
    """预处理可视化对比"""
    img = Image.open(image_path)
    transform = create_transform()

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # 原始图像
    ax[0].imshow(img)
    ax[0].set_title(f'Original ({img.size[0]}x{img.size[1]})')

    # 处理后的张量
    tensor_img = transform(img).numpy().transpose(1, 2, 0)
    tensor_img = np.clip(tensor_img, 0, 1)
    ax[1].imshow(tensor_img)
    ax[1].set_title('Processed Tensor')

    plt.show()


def visualize_feature_distribution(features):
    """特征分布可视化"""
    plt.figure(figsize=(10, 6))
    plt.hist(features.cpu().numpy().flatten(), bins=50, color='purple', alpha=0.7)
    plt.title('VGG16 Feature Distribution')
    plt.xlabel('Feature Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# 扩展功能 ---------------------------------------------------
class FeatureMapVisualizer:
    """中间特征图可视化"""

    def __init__(self, model):
        self.model = model
        self.activations = {}

        # 注册hook捕获各层输出
        for idx, layer in enumerate(model.features):
            if isinstance(layer, nn.Conv2d):
                layer.register_forward_hook(
                    lambda m, i, o, idx=idx: self._record(f'conv_{idx}', o)
                )

    def _record(self, name, output):
        self.activations[name] = output.detach().cpu().numpy()

    def visualize(self, layer_name, channel=0):
        """可视化指定卷积层的特征图"""
        if layer_name not in self.activations:
            raise ValueError(f"无效层名: {layer_name}")

        feat = self.activations[layer_name]
        plt.figure(figsize=(12, 6))
        plt.imshow(feat[0, channel], cmap='viridis')
        plt.title(f'{layer_name} - Channel {channel}')
        plt.colorbar()
        #plt.show()

def get(path):
    image_path=path

    config = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'feature_dim': 9,
        'image_path': path
    }

    try:
        # 初始化模型
        model = VGG16FeatureExtractor(feature_dim=config['feature_dim'])
        model = model.to(config['device']).eval()

        # 数据预处理
        transform = create_transform()
        img = Image.open(config['image_path'])
        input_tensor = transform(img).unsqueeze(0).to(config['device'])

        # 特征提取
        with torch.no_grad():
            features = model(input_tensor)

        print(f"特征维度: {features.shape}")  # 输出示例: torch.Size([1, 1024])
        #visualize_feature_distribution(features)

        # 保存特征
        torch.save({
            'features': features.cpu(),
            'config': config
        }, './Model/vgg16_features.pth')

    except Exception as e:
        print(f"错误发生: {str(e)}")
        if 'img' in locals():
            print(f"图像信息: 模式={img.mode}, 尺寸={img.size}")

    return features.tolist()[0]

if __name__ == "__main__":
    # 处理示例图像
    image_path = "1.png"
    print(get(image_path))