import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt


class AlexNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载预训练AlexNet
        original_model = models.alexnet(pretrained=True)

        # 特征提取部分（移除分类层）
        self.features = nn.Sequential(
            *list(original_model.features.children()),  # 包含所有卷积层
            original_model.avgpool  # 全局平均池化
        )

        # 自定义分类层（输出9维特征）
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 9)  # 关键修改：输出维度设为9
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# 数据预处理管道
def create_alexnet_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),  # AlexNet标准输入尺寸
        transforms.Lambda(lambda x: x.convert('RGB')),  # 强制转换RGB
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


# 可视化特征分布
def visualize_features(features):
    plt.figure(figsize=(10, 4))
    plt.bar(range(9), features.squeeze().detach().numpy())
    plt.title("9维特征向量分布")
    plt.xlabel("特征维度")
    plt.ylabel("特征值")
    plt.grid(True)
    plt.show()


def get(path):
    image_path=path

    # 初始化模型
    model = AlexNetFeatureExtractor()
    model.eval()

    # 数据预处理
    transform = create_alexnet_transform()
    image = Image.open(image_path)
    input_tensor = transform(image).unsqueeze(0)  # [1, 3, 224, 224]

    # 特征提取
    with torch.no_grad():
        features = model(input_tensor)

    #print("特征向量形状:", features.shape)  # 输出: torch.Size([1, 9])
    #visualize_features(features)
    return features.tolist()[0]

if __name__ == "__main__":
    # 处理示例图像
    image_path = "1.png"
    print(get(image_path))