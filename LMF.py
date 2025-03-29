import numpy as np
from sklearn.decomposition import TruncatedSVD


def Lmf(X1,X2):
    # 将这两个矩阵进行水平拼接
    X = np.hstack((X1, X2))

    # 使用TruncatedSVD进行低秩分解
    # 选择目标低秩，例如10
    svd = TruncatedSVD(n_components=9)
    X_fused = svd.fit_transform(X)

    #print("原始拼接矩阵形状:", X.shape)
    #print("低秩融合后矩阵形状:", X_fused)

    return  X_fused

if __name__ == '__main__':

    # 假设我们有两个特征矩阵 X1 和 X2
    np.random.seed(0)
    X1 = np.random.rand(10, 9)  # 100个样本，50个特征
    X2 = np.random.rand(10, 9)  # 100个样本，50个特征
    print(X1)
    print(Lmf(X1,X2))