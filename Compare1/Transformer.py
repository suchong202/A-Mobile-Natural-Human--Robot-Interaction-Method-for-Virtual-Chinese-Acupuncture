import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import logging
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 配置日志记录
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class RobustFeatureExtractor(nn.Module):
    """鲁棒的特征提取模型"""

    def __init__(self, model_name='resnet50', feature_dim=1024):
        super().__init__()
        self.input_channels = 3  # 强制指定输入通道数

        # 加载预训练模型
        try:
            model_loader = getattr(models, model_name)
            self.base_model = model_loader(pretrained=True)
        except AttributeError:
            raise ValueError(f"不支持的模型: {model_name}")

        # 修改第一层卷积适配输入通道
        if hasattr(self.base_model, 'conv1'):
            original_conv1 = self.base_model.conv1
            self.base_model.conv1 = nn.Conv2d(
                self.input_channels,
                original_conv1.out_channels,
                kernel_size=original_conv1.kernel_size,
                stride=original_conv1.stride,
                padding=original_conv1.padding,
                bias=False
            )

        # 移除分类层
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        # 自动计算特征维度
        with torch.no_grad():
            test_input = torch.randn(1, self.input_channels, 640, 480)
            self.feature_dim = self.features(test_input).view(1, -1).shape[1]

        # 特征降维层
        self.fc = nn.Linear(self.feature_dim, feature_dim)

        # 参数初始化
        nn.init.kaiming_normal_(self.fc.weight)

    @staticmethod
    def validate_input(x, expected_channels=3):
        """输入验证装饰器"""
        if x.shape[1] != expected_channels:
            raise ValueError(
                f"通道数不匹配！预期{expected_channels}通道，实际{x.shape[1]}通道\n"
                "可能原因：\n"
                "1. 图像未正确转换为RGB\n"
                "2. 预处理流程错误\n"
                "3. 模型定义与输入不匹配"
            )

    def forward(self, x):
        self.validate_input(x, self.input_channels)
        features = self.features(x)
        features = features.view(features.size(0), -1)
        return self.fc(features)


def safe_image_loader(image_path: str) -> Image.Image:
    """安全加载图像并处理各种模式"""
    try:
        img = Image.open(image_path)
        logging.info(f"原始图像模式: {img.mode}, 尺寸: {img.size}")

        # 处理特殊模式
        if img.mode in ('RGBA', 'LA'):
            # 创建白色背景并合并
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1])  # 使用alpha通道作为mask
            return background
        elif img.mode == 'P':
            return img.convert('RGB')
        elif img.mode == 'L':
            return img.convert('RGB')
        elif img.mode == 'CMYK':
            return img.convert('RGB')
        return img.convert('RGB')
    except Exception as e:
        logging.error(f"图像加载失败: {str(e)}")
        raise


def create_robust_transform() -> transforms.Compose:
    """创建鲁棒的预处理流程"""
    return transforms.Compose([
        transforms.Resize((640, 480)),
        transforms.Lambda(lambda x: x.convert('RGB')),  # 二次确认通道转换
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.Lambda(lambda x: x[:3])  # 确保只保留前3个通道
    ])


def visualize_process(image_path: str):
    """可视化预处理过程"""
    try:
        original_img = safe_image_loader(image_path)
        transform = create_robust_transform()
        tensor_img = transform(original_img)

        plt.figure(figsize=(12, 6))

        # 原始图像
        plt.subplot(1, 3, 1)
        plt.imshow(original_img)
        plt.title(f'Original ({original_img.size[0]}x{original_img.size[1]})')
        plt.axis('off')

        # 通道分布
        plt.subplot(1, 3, 2)
        channels = ['Red', 'Green', 'Blue']
        for i in range(3):
            plt.hist(tensor_img[i].numpy().flatten(), bins=50, alpha=0.5, label=channels[i])
        plt.title('Channel Distribution')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.legend()

        # 处理后的张量
        plt.subplot(1, 3, 3)
        display_img = tensor_img.numpy().transpose(1, 2, 0)
        display_img = np.clip(display_img, 0, 1)
        plt.imshow(display_img)
        plt.title('Processed Tensor')
        plt.axis('off')

        plt.tight_layout()
        #plt.show()

    except Exception as e:
        logging.error(f"可视化失败: {str(e)}")


def get(path):
    image_path=path

    # 配置参数
    config = {
        'model_name': 'resnet50',
        'feature_dim': 9,
        'image_path': path,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }

    try:
        logging.info("初始化模型...")
        model = RobustFeatureExtractor(
            model_name=config['model_name'],
            feature_dim=config['feature_dim']
        ).to(config['device']).eval()

        logging.info("加载和预处理图像...")
        img = safe_image_loader(config['image_path'])
        transform = create_robust_transform()
        input_tensor = transform(img).unsqueeze(0).to(config['device'])

        logging.info(f"输入张量形状: {input_tensor.shape}")
        logging.info(f"输入范围: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")

        # 可视化预处理
        visualize_process(config['image_path'])

        logging.info("提取特征...")
        with torch.no_grad():
            features = model(input_tensor)

        logging.info(f"特征维度: {features.shape}")
        logging.info(f"特征范围: [{features.min():.3f}, {features.max():.3f}]")

        # 保存结果
        torch.save({
            'features': features.cpu(),
            'config': config
        }, './Model/features.pth')

        logging.info("处理完成！")

    except Exception as e:
        logging.error(f"主流程错误: {str(e)}", exc_info=True)

    return features.tolist()[0]


if __name__ == "__main__":
    # 处理示例图像
    image_path = "1.png"
    print(get(image_path))