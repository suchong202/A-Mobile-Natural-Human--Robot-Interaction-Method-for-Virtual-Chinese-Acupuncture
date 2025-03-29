import numpy as np
from scipy.fft import fft, ifft


def count_sketch_projection(matrix, hash_size, seed=42):
    """
    简化版的Count Sketch投影函数，对矩阵的每一列进行投影。
    实际实现中，应使用哈希函数和随机符号来投影。
    这里仅用一个简单的随机投影来模拟。
    """
    np.random.seed(seed)
    random_matrix = np.random.randn(hash_size, matrix.shape[0])
    return np.dot(random_matrix, matrix)


def mcb_matrix_fusion(feature_matrix1, feature_matrix2, hash_size):
    """
    简化的MCB特征矩阵融合函数。
    feature_matrix1, feature_matrix2: 输入的两个特征矩阵。
    hash_size: Count Sketch投影后的维度。
    """
    # 确保两个矩阵的列数相同（即特征数量相同）
    assert len(feature_matrix1[0]) == len(feature_matrix2[1]), "特征矩阵的列数必须相同"

    # Step 1: 对每个特征（列）进行Count Sketch投影
    projected_matrix1 = np.zeros((hash_size, len(feature_matrix1[0])), dtype=np.complex128)
    projected_matrix2 = np.zeros((hash_size, len(feature_matrix1[0])), dtype=np.complex128)

    for i in range(len(feature_matrix1[0])):
        projected_matrix1[:, i] = fft(count_sketch_projection(feature_matrix1[:, i], hash_size))
        projected_matrix2[:, i] = fft(count_sketch_projection(feature_matrix2[:, i], hash_size))

    # Step 2: 在频域中进行元素级乘法（模拟外积）
    fused_matrix_fft = projected_matrix1 * projected_matrix2

    # Step 3: 进行逆FFT回到时域
    fused_matrix = np.zeros((hash_size, len(feature_matrix1[0])), dtype=np.float64)
    for i in range(fused_matrix_fft.shape[1]):
        fused_matrix[:, i] = np.real(ifft(fused_matrix_fft[:, i]))

    # Step 4: （可选）进行归一化或其他处理
    # 这里我们直接返回融合后的特征矩阵
    return fused_matrix


def Mcb(x1,x2):
    x1 =np.array(x1)
    x2= np.array(x2)
    # 设置Count Sketch投影后的维度
    hash_size =len(x1)

    # 进行MCB特征矩阵融合
    fused_feature_matrix = mcb_matrix_fusion(x1, x2, hash_size)

    print("融合后的特征矩阵:")
    print(fused_feature_matrix)
    return fused_feature_matrix

if __name__ == '__main__':

    # 示例特征矩阵
    x1 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    x2 = np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])

    Mcb(x1,x2)