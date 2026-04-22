import torch.nn as nn


class LinearProjection(nn.Module):
    """논문 HyperKVGeneratorFixed의 linear_K, linear_V와 일치.

    hidden_dim → num_kv * d_model 로 flatten 후 view(B, num_kv, d_model).
    """

    def __init__(self, hidden_dim, d_model, num_kv):
        super().__init__()
        self.num_kv = num_kv
        self.d_model = d_model
        self.linear_K = nn.Linear(hidden_dim, num_kv * d_model)
        self.linear_V = nn.Linear(hidden_dim, num_kv * d_model)

    def forward(self, h):
        B = h.size(0)
        K = self.linear_K(h).view(B, self.num_kv, self.d_model)
        V = self.linear_V(h).view(B, self.num_kv, self.d_model)
        return K, V
