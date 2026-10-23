import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionFusion(nn.Module):
    # attention
    def __init__(self, feature_dim, feature_dim2):
        super(AttentionFusion, self).__init__()
        self.feature_dim = feature_dim
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim2, feature_dim)
        self.value = nn.Linear(feature_dim2, feature_dim)

    def forward(self, sequence, feature):
        feature_expanded = feature.unsqueeze(1).expand(-1, sequence.size(1), -1)

        Q = self.query(sequence)
        K = self.key(feature_expanded)
        V = self.value(feature_expanded)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.feature_dim ** 0.5)
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        fused_data = sequence + attention_output
        return fused_data.permute(0, 2, 1)
    

class AttentionFusion1(nn.Module):
    # plus
    def __init__(self, feature_dim, feature_dim2):
        super(AttentionFusion, self).__init__()
        self.feature_dim = feature_dim
        self.g_pre_net = nn.Conv1d(feature_dim, feature_dim2, kernel_size=3, padding=1)
        self.g_aft_net = nn.Conv1d(feature_dim2, feature_dim, kernel_size=3, padding=1)
    
    def forward(self, sequence, feature):
        feature_expanded = feature.unsqueeze(1).expand(-1, sequence.size(1), -1)
        sequence = self.g_pre_net(sequence.permute(0,2,1))
        fused_data = sequence + feature_expanded.permute(0,2,1)
        fused_data = self.g_aft_net(fused_data)
        return fused_data


if __name__ == '__main__':

    attention_fusion = AttentionFusion(feature_dim=100, feature_dim2=9)
    a = torch.randint(0, 20, ([32, 50, 100]))
    b = torch.randint(0, 20, ([32, 9]))
    fused_data = attention_fusion(sequence=a.to(torch.float), feature=b.to(torch.float))
    print(fused_data.shape)