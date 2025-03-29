import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


class BNInceptionFeatureExtractor(nn.Module):
    def __init__(self, feature_dim=9):
        super().__init__()
        # 正确加载Inception v3的方式
        base_model = models.inception_v3(pretrained=True, aux_logits=True)  # 必须保持aux_logits=True

        # 禁用辅助分类器
        base_model.aux_logits = False
        del base_model.AuxLogits  # 删除辅助分类器

        # 特征提取层
        self.features = nn.Sequential(
            base_model.Conv2d_1a_3x3,
            base_model.Conv2d_2a_3x3,
            base_model.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            base_model.Conv2d_3b_1x1,
            base_model.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            base_model.Mixed_5b,
            base_model.Mixed_5c,
            base_model.Mixed_5d,
            base_model.Mixed_6a,
            base_model.Mixed_6b,
            base_model.Mixed_6c,
            base_model.Mixed_6d,
            base_model.Mixed_6e,
            base_model.Mixed_7a,
            base_model.Mixed_7b,
            base_model.Mixed_7c,
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # 替换分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, feature_dim)  # 输出9维特征
        )

        # 参数初始化
        nn.init.kaiming_normal_(self.classifier[1].weight)

    def forward(self, x):
        # 显式处理辅助分类器问题
        if self.training:
            raise NotImplementedError("本模型仅用于推理模式")
        x = self.features(x)
        return self.classifier(x)


# 数据预处理（保持不变）
def create_transform():
    return transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.Lambda(lambda x: x.convert('RGB')),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def get(path):

    image_path=path

    # 初始化模型
    model = BNInceptionFeatureExtractor(feature_dim=9)
    model.eval()

    # 加载图像
    image = Image.open(image_path)
    transform = create_transform()
    input_tensor = transform(image).unsqueeze(0)  # [1, 3, 299, 299]

    # 特征提取
    with torch.no_grad():
        features = model(input_tensor)

    print("特征向量形状:", features.shape)  # 输出: torch.Size([1, 9])

    return features.tolist()[0]

if __name__ == "__main__":

    # 处理示例图像
    image_path = "1.png"
    print(get(image_path))