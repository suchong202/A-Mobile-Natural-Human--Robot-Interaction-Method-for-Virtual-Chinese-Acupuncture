import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# 定义增强版LSTM特征提取器
class OptimizedLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=512, num_layers=2):
        super().__init__()
        # 空间压缩层
        self.compress = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # 尺寸减半
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, stride=2, padding=1)  # 再次减半
        )

        # 增强型LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,  # 启用双向LSTM
            dropout=0.3
        )

        # 自适应特征融合
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # 空间压缩 [b,3,640,480] -> [b,3,160,120]
        x = self.compress(x)

        # 转换为序列 [b,160*120,3]
        batch_size, _, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(batch_size, h * w, 3)

        # 双向LSTM处理
        outputs, _ = self.lstm(x)  # [b, seq_len, hidden*2]

        # 注意力机制融合特征
        attn_weights = torch.softmax(self.attention(outputs), dim=1)
        features = torch.sum(attn_weights * outputs, dim=1)

        return features


# 改进的数据预处理流程
def create_transform(target_size=(640, 480)):
    return transforms.Compose([
        transforms.Resize(target_size),  # 保持原始分辨率
        transforms.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet标准化
                             std=[0.229, 0.224, 0.225])
    ])


# 可视化处理过程
def visualize_processing(image_path):
    orig_img = Image.open(image_path)
    transform = create_transform()

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(orig_img)
    ax[0].set_title('Original Image (640x480)')

    processed = transform(orig_img).permute(1, 2, 0).numpy()
    ax[1].imshow(processed)
    ax[1].set_title('Processed Image')
    plt.show()

def get(path):

    image_path=path

    # 配置参数
    config = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'target_size': (640, 480),
        'hidden_size': 9,
        'checkpoint_path': './Model/model.pth'
    }

    # 初始化模型
    model = OptimizedLSTM(
        hidden_size=config['hidden_size'],
        num_layers=2
    ).to(config['device'])

    # 可视化处理过程
    #visualize_processing(image_path)

    # 数据预处理
    transform = create_transform()
    image = Image.open(image_path)
    input_tensor = transform(image).unsqueeze(0).to(config['device'])

    # 特征提取
    with torch.no_grad():
        features = model(input_tensor)

    # 保存/加载模型示例
    torch.save(model.state_dict(), config['checkpoint_path'])
    #print(f"特征向量形状：{features.shape} 值范围：[{features.min():.3f}, {features.max():.3f}]")

    return features.tolist()[0]

if __name__ == "__main__":


        # 处理示例图像
        image_path = "1.png"
        print(get(image_path))
