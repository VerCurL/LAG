import torch
import torch.nn as nn
from .flatten import build_flattener


class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, activation_id):
        super(MLPLayer, self).__init__()
        self._size = [input_dim] + list(map(int, hidden_size.split(' ')))
        self._hidden_layers = len(self._size) - 1
        active_func = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(), nn.ELU()][activation_id]

        fc_h = []
        for j in range(len(self._size) - 1):
            in_dim = self._size[j]
            out_dim = self._size[j + 1]
            is_last = (j == len(self._size) - 2)

            fc_h.append(nn.Linear(in_dim, out_dim))

            # 最后一层如果输出是 1，就不要再做激活和 LayerNorm
            if is_last and out_dim == 1:
                pass
            else:
                fc_h.append(active_func)
                fc_h.append(nn.LayerNorm(out_dim))

        self.fc = nn.Sequential(*fc_h)

    def forward(self, x: torch.Tensor):
        x = self.fc(x)
        return x

    @property
    def output_size(self) -> int:
        return self._size[-1]


# Feature extraction module
class MLPBase(nn.Module):
    def __init__(self, obs_space, hidden_size, activation_id, use_feature_normalization):
        super(MLPBase, self).__init__()
        self._hidden_size = hidden_size
        self._activation_id = activation_id
        self._use_feature_normalization = use_feature_normalization

        self.obs_flattener = build_flattener(obs_space)
        input_dim = self.obs_flattener.size
        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(input_dim)
        self.mlp = MLPLayer(input_dim, self._hidden_size, self._activation_id)

    def forward(self, x: torch.Tensor):
        if self._use_feature_normalization:
            x = self.feature_norm(x)
        x = self.mlp(x)
        return x

    @property
    def output_size(self) -> int:
        return self.mlp.output_size
