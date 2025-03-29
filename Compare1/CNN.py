import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


# 增强型CNN模型（显式处理通道问题）
class RobustCNN(nn.Module):
    def __init__(self, feature_dim=1024):
        super().__init__()
        # 输入通道显式验证
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),  # 严格限定输入3通道
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, feature_dim)

    def forward(self, x):
        # 添加通道数验证
        if x.shape[1] != 3:
            raise ValueError(f"预期3通道输入，实际得到{x.shape[1]}通道")
        return self.fc(self.adaptive_pool(self.features(x)).view(x.size(0), -1))


# 强化的数据预处理
def create_transform():
    return transforms.Compose([
        transforms.Resize((640, 480)),
        transforms.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),  # 强制转换RGB
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


# 修正的可视化函数
def visualize_processing(image_path):
    orig_img = Image.open(image_path).convert('RGB')  # 显式转换
    transform = create_transform()

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(orig_img)
    ax[0].set_title(f'Original Image ({orig_img.size[0]}x{orig_img.size[1]})')

    processed = transform(orig_img).numpy().transpose(1, 2, 0)  # 正确转换维度
    processed = np.clip(processed, 0, 1)  # 限制数值范围
    ax[1].imshow(processed)
    ax[1].set_title('Processed Tensor')
    plt.show()

def get(path):
    image_path=path


    # 配置参数
    config = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'feature_dim': 9,
        'checkpoint_path': './Model/fixed_model.pth'
    }

    # 初始化模型
    model = RobustCNN(config['feature_dim']).to(config['device'])



    try:
        # 加载并转换图像
        image = Image.open(image_path).convert('RGB')  # 双重保险
        input_tensor = create_transform()(image).unsqueeze(0).to(config['device'])

        # 提取特征
        with torch.no_grad():
            features = model(input_tensor)

        #print(f"成功提取特征，形状：{features.shape}")
        torch.save(model.state_dict(), config['checkpoint_path'])

    except Exception as e:
        print(f"错误发生：{str(e)}")
        # 显示图像信息辅助调试
        debug_img = Image.open(image_path)
        print(f"图像模式：{debug_img.mode}, 尺寸：{debug_img.size}")


    return features.tolist()[0]


if __name__ == "__main__":
    # 处理示例图像
    image_path = "1.png"
    print(get(image_path))