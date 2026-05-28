import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import MLPLayer
from .flatten import build_flattener
from .gru import GRULayer


class AeroTAFGRULayer(nn.Module):
    """
    Minimal temporal AeroTAF:
    first encode spatial interaction at each timestep, then apply GRU over
    the time dimension for each agent independently.
    """

    def __init__(
        self,
        agent_num: int,
        head_num: int,
        KQ_input_dim: int,
        V_input_dim: int,
        activation_id,
        KQ_hidden_size: str,
        V_hidden_size: str,
        output_hidden_size: str,
        gru_hidden_size: int,
        gru_num_layers: int,
    ):
        super(AeroTAFGRULayer, self).__init__()
        self.agent_num = agent_num
        self.head_num = head_num

        self._KQ_hidden_size = [KQ_input_dim] + list(map(int, KQ_hidden_size.split(" ")))
        self._V_hidden_size = [V_input_dim] + list(map(int, V_hidden_size.split(" ")))
        self._output_hidden_size = [gru_hidden_size] + list(map(int, output_hidden_size.split(" "))) + [1]

        self.K_module = MLPLayer(KQ_input_dim, KQ_hidden_size, activation_id)
        self.Q_module = MLPLayer(KQ_input_dim, KQ_hidden_size, activation_id)
        self.V_module = MLPLayer(V_input_dim, V_hidden_size, activation_id)
        self.attn_output_module = nn.Linear(self._V_hidden_size[-1], V_input_dim)
        self.norm = nn.LayerNorm(V_input_dim)

        self.rnn = GRULayer(
            input_size=V_input_dim,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
        )

        self.threat_output_module = MLPLayer(
            gru_hidden_size,
            output_hidden_size + " " + str(self._output_hidden_size[-1]),
            activation_id,
        )
        self.attack_output_module = MLPLayer(
            gru_hidden_size,
            output_hidden_size + " " + str(self._output_hidden_size[-1]),
            activation_id,
        )

        self.record_info = {}

    def _spatial_encode(self, s_t: torch.Tensor, a_t: torch.Tensor):
        """
        s_t: [B, N, obs_dim]
        a_t: [B, N, act_dim]
        returns:
            x: [B, N, V_input_dim]
        """
        x = torch.cat((s_t, a_t), dim=-1)
        batch_size, _, x_dim = x.shape

        KQ_dim = self._KQ_hidden_size[-1]
        V_dim = self._V_hidden_size[-1]

        s_flat = s_t.reshape(batch_size * self.agent_num, s_t.shape[-1])
        x_flat = x.reshape(batch_size * self.agent_num, x_dim)

        K = self.K_module(s_flat).view(batch_size, self.agent_num, KQ_dim)
        Q = self.Q_module(s_flat).view(batch_size, self.agent_num, KQ_dim)
        V = self.V_module(x_flat).view(batch_size, self.agent_num, V_dim)

        KQ_head_dim = KQ_dim // self.head_num
        V_head_dim = V_dim // self.head_num

        K = K.view(batch_size, self.agent_num, self.head_num, KQ_head_dim).transpose(1, 2)
        Q = Q.view(batch_size, self.agent_num, self.head_num, KQ_head_dim).transpose(1, 2)
        V = V.view(batch_size, self.agent_num, self.head_num, V_head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(KQ_head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        heads = torch.matmul(attn_weights, V)
        heads = heads.transpose(1, 2).contiguous().view(batch_size, self.agent_num, V_dim)

        attn_output = self.attn_output_module(heads)
        x = self.norm(x + attn_output)

        return x

    def forward(self, s: torch.Tensor, a: torch.Tensor, rnn_states: torch.Tensor, masks: torch.Tensor):
        """
        Inputs:
            s:           [B * T * N, obs_dim]
            a:           [B * T * N, act_dim]
            rnn_states:  [B * N, L, H]
            masks:       [B * T * N, 1]

        Returns:
            threat_output: [B, N, 1]
            attack_output: [B, N, 1]
            rnn_states:    [B * N, L, H]
        """
        batch_agent_size = rnn_states.size(0)
        total_agent_steps = s.size(0)

        if batch_agent_size % self.agent_num != 0:
            raise ValueError(
                "AeroTAF_GRU rnn_states size mismatch: expected rnn_states to be [B * N, L, H]."
            )

        batch_size = batch_agent_size // self.agent_num
        seq_len = total_agent_steps // batch_agent_size

        s = s.view(batch_size, seq_len, self.agent_num, s.shape[-1])
        a = a.view(batch_size, seq_len, self.agent_num, a.shape[-1])

        spatial_outputs = []
        for t in range(seq_len):
            spatial_outputs.append(self._spatial_encode(s[:, t], a[:, t]))

        # spatial_outputs: [B, T, N, dim]
        x = torch.stack(spatial_outputs, dim=1)

        # For GRULayer:
        # rnn_states: [B * N, L, rnn_hidden_size]
        # x: [T * (B * N), dim]
        feature_dim = x.shape[-1]
        x = x.transpose(1, 0).contiguous()                  # [T, B, N, dim]
        x = x.view(seq_len, batch_size * self.agent_num, feature_dim)
        x = x.view(seq_len * batch_size * self.agent_num, feature_dim)

        # masks: [B * T * N, 1] -> [T * B * N, 1]
        masks_dim = masks.shape[-1]
        masks = masks.view(batch_size, seq_len, self.agent_num, masks_dim)
        masks = masks.transpose(1, 0).contiguous()
        masks = masks.view(seq_len, batch_size * self.agent_num, masks_dim)
        masks = masks.view(seq_len * batch_size * self.agent_num, masks_dim)

        rnn_output, rnn_states = self.rnn(x, rnn_states, masks)

        # rnn_output: [T * (B * N), dim] -> [T, B, N, dim]
        rnn_output = rnn_output.view(seq_len, batch_size, self.agent_num, -1)

        # output: [T, B, N, 1] -> [T, B, 1] -> [B, T, 1] -> [B * T, 1]
        threat_output = self.threat_output_module(rnn_output).mean(dim=2).transpose(1, 0).contiguous().view(batch_size * seq_len, -1)
        attack_output = self.attack_output_module(rnn_output).mean(dim=2).transpose(1, 0).contiguous().view(batch_size * seq_len, -1)
        return threat_output, attack_output, rnn_states

    @property
    def output_size(self):
        return self._output_hidden_size[-1]

    def get_info(self):
        return self.record_info


class AeroTAFGRUBase(nn.Module):
    def __init__(
        self,
        obs_space,
        act_space,
        agent_num: int,
        head_num: int,
        KQ_hidden_size: str,
        V_hidden_size: str,
        output_hidden_size: str,
        gru_hidden_size: int,
        gru_num_layers: int,
        activation_id,
        use_feature_normalization,
    ):
        super().__init__()
        self._use_feature_normalization = use_feature_normalization
        self.obs_flattener = build_flattener(obs_space)
        self.act_flattener = build_flattener(act_space)
        obs_input_dim = self.obs_flattener.size
        act_input_dim = self.act_flattener.size

        if self._use_feature_normalization:
            self.obs_feature_norm = nn.LayerNorm(obs_input_dim)
            self.act_feature_norm = nn.LayerNorm(act_input_dim)

        self.AeroTAF = AeroTAFGRULayer(
            agent_num=agent_num,
            head_num=head_num,
            KQ_input_dim=obs_input_dim,
            V_input_dim=obs_input_dim + act_input_dim,
            activation_id=activation_id,
            KQ_hidden_size=KQ_hidden_size,
            V_hidden_size=V_hidden_size,
            output_hidden_size=output_hidden_size,
            gru_hidden_size=gru_hidden_size,
            gru_num_layers=gru_num_layers,
        )

    def forward(self, s, a, rnn_states, masks):
        if self._use_feature_normalization:
            s = self.obs_feature_norm(s)
            a = self.act_feature_norm(a)
        threat_output, attack_output, rnn_states = self.AeroTAF(s, a, rnn_states, masks)
        return threat_output, attack_output, rnn_states

    @property
    def output_size(self):
        return self.AeroTAF.output_size

    @property
    def record_info(self):
        return self.AeroTAF.get_info()
