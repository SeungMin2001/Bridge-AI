import torch
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

    def _project(self, linear, hidden):
        out = linear(hidden)
        if hidden.dim() == 2:
            batch = hidden.size(0)
            return out.view(batch, self.num_kv, self.d_model)

        if hidden.dim() == 3:
            batch, slots, _ = hidden.shape
            out = out.view(batch, slots, self.num_kv, self.d_model)
            if slots == self.num_kv:
                # Each pooled slot owns the matching output block. This keeps the
                # old num_kv*d projection shape while letting slots specialize.
                idx = torch.arange(slots, device=hidden.device)
                return out[:, idx, idx, :]
            return out.mean(dim=1)

        raise ValueError(f"Unsupported hidden rank for LinearProjection: {hidden.dim()}")

    def forward(self, h_k, h_v=None):
        if h_v is None:
            h_v = h_k
        K = self._project(self.linear_K, h_k)
        V = self._project(self.linear_V, h_v)
        return K, V
