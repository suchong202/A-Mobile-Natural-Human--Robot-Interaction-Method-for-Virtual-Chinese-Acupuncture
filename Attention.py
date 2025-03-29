import torch
import torch.nn as nn
import numpy as np

class CustomAttention(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super(CustomAttention, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        assert self.head_dim * num_heads == feature_dim, "feature_dim must be divisible by num_heads"
        # rgb-RGB模态的Q/K/V投影
        self.query_r = nn.Linear(feature_dim, feature_dim, bias=False)
        self.key_r = nn.Linear(feature_dim, feature_dim, bias=False)
        self.value_r = nn.Linear(feature_dim, feature_dim, bias=False)
        # skeleton-骨骼模态的Q/K/V投影
        self.query_s = nn.Linear(feature_dim, feature_dim, bias=False)
        self.key_s = nn.Linear(feature_dim, feature_dim, bias=False)
        self.value_s = nn.Linear(feature_dim, feature_dim, bias=False)
        # 最终融合层
        self.fc_out = nn.Linear(feature_dim, feature_dim, bias=False)


    def forward(self, r, s, alpha=1.0):
        r_t, s_t = r, s     #保留原始输入用于残差连接
        r, s = r.unsqueeze(1), s.unsqueeze(1)      #添加序列维度（8,1,512）-单时间步序列
        B, N, _ = r.shape       #B=8,N=1

        Q_r = self.query_r(r).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)     #将512维度特征切分成8头*64维（8,8,1,64）
        K_r = self.key_r(r).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V_r = self.value_r(r).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        Q_s = self.query_s(r).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)     #（r）改为（s）
        K_s = self.key_s(r).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V_s = self.value_s(r).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        #RGB→骨骼注意力
        energy_r2s = torch.matmul(Q_r, K_s.permute(0, 1, 3, 2)) / (self.feature_dim ** 0.5)
        energy_s2r = torch.matmul(Q_s, K_r.permute(0, 1, 3, 2)) / (self.feature_dim ** 0.5)
        #骨骼→RGB注意力
        attention_r2s = torch.softmax(energy_r2s, dim=-1)
        attention_s2r = torch.softmax(energy_s2r, dim=-1)

        if alpha > 0:
            alpha = np.random.beta(alpha, alpha)    #Beta分布随机权重
        else:
            alpha = 1.0

        #加权融合两种注意力结果
        out = (alpha * torch.matmul(attention_r2s, V_r) + (1-alpha)* torch.matmul(attention_s2r, V_s)).permute(0, 2, 1, 3).contiguous()
        out = out.view(B, N, self.feature_dim)
        out = self.fc_out(out)
        out = out.squeeze()
        return out + s_t + r_t      #融合结果与原始特征相加


def attention(f1,f2):

    feature_dim = len(f1[0])
    num_heads = 1
    f1 = torch.tensor(f1).float()
    f2 = torch.tensor(f2).float()
    #print(f1)

    # Concatenate f1 and f2
    a = CustomAttention(feature_dim, num_heads)
    b = a(f1, f2)


    return b.tolist()

if __name__ == '__main__':



    f1=[[1.1,3.1,4.1,6.1,2.1,4.1,7.1,8.1,4.1]]
    f2=[[1.1,3.1,4.1,6.1,2.1,4.1,7.1,8.1,4.1]]

    print(attention(f1, f2))




