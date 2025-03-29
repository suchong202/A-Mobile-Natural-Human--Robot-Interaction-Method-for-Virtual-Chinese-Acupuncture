import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, confusion_matrix)


# 加载图片数据和标签
def load_data(data_dir):
    data = []
    labels = []
    for label in ['0', '1', '2', '3']:
        folder = os.path.join(data_dir, label)
        for filename in os.listdir(folder):
            if filename.endswith('.png'):
                img_path = os.path.join(folder, filename)
                img = Image.open(img_path).convert('L')  # 转为灰度图
                img_array = np.array(img).flatten()  # 展平为1D数组
                data.append(img_array)
                labels.append(int(label) - 1)  # 标签转换为0-3
    return np.array(data), np.array(labels)


# 定义评估指标计算函数
def calculate_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    # 计算特异性
    cm = confusion_matrix(y_true, y_pred)
    specificity = []
    for i in range(cm.shape[0]):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        specificity.append(tn / (tn + fp))
    specificity_avg = np.mean(specificity)

    return precision, recall, specificity_avg, f1


# 主程序
if __name__ == "__main__":
    # 加载数据
    X, y = load_data('Pic2')

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # 数据归一化
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train.astype(np.float32))
    X_test = scaler.transform(X_test.astype(np.float32))

    # 构建DBN模型（两个RBM层 + 分类器）
    rbm1 = BernoulliRBM(n_components=256, learning_rate=0.06,
                        n_iter=20, random_state=42, verbose=True)
    rbm2 = BernoulliRBM(n_components=128, learning_rate=0.05,
                        n_iter=20, random_state=42, verbose=True)

    # 逐层训练
    print("Training first RBM layer...")
    rbm1.fit(X_train)
    X_train = rbm1.transform(X_train)
    X_test = rbm1.transform(X_test)

    print("Training second RBM layer...")
    rbm2.fit(X_train)
    X_train = rbm2.transform(X_train)
    X_test = rbm2.transform(X_test)

    # 训练分类器
    print("Training classifier...")
    classifier = LogisticRegression(max_iter=1000, multi_class='multinomial')
    classifier.fit(X_train, y_train)

    # 预测和评估
    y_pred = classifier.predict(X_test)
    precision, recall, specificity, f1 = calculate_metrics(y_test, y_pred)

    # 输出结果
    print("\nEvaluation Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1 Score:  {f1:.4f}")