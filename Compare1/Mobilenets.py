import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class MobileNetFeatureExtractor(nn.Module):
    def __init__(self, model_type='mobilenet_v2', feature_dim=1024):
        super().__init__()

        # 加载预训练模型
        self.base_model = getattr(models, model_type)(pretrained=True)

        # 特征提取部分
        self.features = nn.Sequential(
            self.base_model.features,
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # 动态计算特征维度
        self._feature_dim = self._get_feature_dim()
        self.fc = nn.Linear(self._feature_dim, feature_dim)

    def _get_feature_dim(self):
        """自动计算特征维度"""
        with torch.no_grad():
            test_input = torch.randn(1, 3, 640, 480)
            features = self.features(test_input)
            return features.view(1, -1).shape[1]

    def forward(self, x):
        # 输入验证
        if x.shape[2:] != (640, 480):
            raise ValueError(f"输入尺寸需为640x480，当前为{x.shape[2:]}")
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def create_transform():
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
    img = Image.open(image_path)
    transform = create_transform()

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(img)
    ax[0].set_title(f'Original ({img.size[0]}x{img.size[1]})')

    # 处理后的张量可视化
    tensor_img = transform(img).numpy().transpose(1, 2, 0)
    tensor_img = np.clip(tensor_img, 0, 1)
    ax[1].imshow(tensor_img)
    ax[1].set_title('Normalized Tensor')
    plt.show()


def visualize_features(features):
    plt.figure(figsize=(10, 4))
    plt.plot(features.cpu().numpy().flatten())
    plt.title('Feature Vector Distribution')
    plt.xlabel('Dimension Index')
    plt.ylabel('Feature Value')
    plt.grid(True)
    plt.show()

# 扩展功能 ---------------------------------------------------
class MultiHeadMobileNet(MobileNetFeatureExtractor):
    """多任务特征提取"""

    def __init__(self, num_heads=3, **kwargs):
        super().__init__(**kwargs)
        self.heads = nn.ModuleList([
            nn.Linear(self._feature_dim, kwargs['feature_dim'])
            for _ in range(num_heads)
        ])

    def forward(self, x):
        base_features = super().forward(x)
        return [head(base_features) for head in self.heads]


class FeatureVisualizer:
    """中间特征可视化"""

    def __init__(self, model):
        self.model = model
        self.activations = {}

        # 注册hook捕获特征
        for name, layer in self.model.features.named_children():
            layer.register_forward_hook(
                lambda m, i, o, n=name: self._record(n, o)
            )

    def _record(self, name, output):
        self.activations[name] = output.detach()

    def show_layer(self, layer_name, channel=0):
        feat = self.activations.get(layer_name)
        if feat is None:
            raise ValueError(f"未找到层 {layer_name}")

        plt.imshow(feat[0, channel].cpu().numpy(), cmap='viridis')
        plt.title(f'{layer_name} - Channel {channel}')
        plt.colorbar()
        plt.show()

def get(path):
    image_path=path

    config = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'model_type': 'mobilenet_v2',
        'feature_dim': 9,
        'image_path': path
    }

    try:
        # 初始化模型
        model = MobileNetFeatureExtractor(
            model_type=config['model_type'],
            feature_dim=config['feature_dim']
        ).to(config['device']).eval()

        # 数据预处理
        transform = create_transform()
        img = Image.open(config['image_path'])
        input_tensor = transform(img).unsqueeze(0).to(config['device'])

        # 特征提取
        with torch.no_grad():
            features = model(input_tensor)

        print(f"特征维度: {features.shape}")  # 输出示例: torch.Size([1, 1024])
        #visualize_features(features)

        # 保存特征
        torch.save({
            'features': features.cpu(),
            'config': config
        }, './Model/mobilenet_features.pth')

    except Exception as e:
        print(f"错误发生: {str(e)}")
        if 'img' in locals():
            print(f"图像模式: {img.mode}, 尺寸: {img.size}")

    return features.tolist()[0]

if __name__ == "__main__":
    # 处理示例图像
    image_path = "1.png"
    print(get(image_path))