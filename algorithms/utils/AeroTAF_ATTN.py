import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .flatten import build_flattener
from .mlp import MLPLayer


class AeroTAFATTNLayer(nn.Module):
    """
    Minimal temporal-attention AeroTAF:
    first encode spatial interaction at each timestep, then apply causal
    attention over the time dimension for each agent independently.

    Training:
        forward_sequence(s, a, seq_len=None)
        - accepts a variable-length sequence for one episode, or multiple
          same-length sequences batched together when seq_len is provided.

    Inference:
        forward_step(s, a, kv_cache=None, time_index=None, max_cache_len=None)
        - accepts one timestep and updates KV cache for online inference.
    """

    def __init__(
        self,
        agent_num: int,
        head_num: int,
        time_head_num: int,
        KQ_input_dim: int,
        V_input_dim: int,
        activation_id,
        KQ_hidden_size: str,
        V_hidden_size: str,
        attn_output_hidden_size: str,
        field_output_hidden_size: str,
    ):
        super(AeroTAFATTNLayer, self).__init__()
        self.agent_num = agent_num
        self.head_num = head_num
        self.time_head_num = time_head_num

        self._KQ_hidden_size = [KQ_input_dim] + list(map(int, KQ_hidden_size.split(" ")))
        self._V_hidden_size = [V_input_dim] + list(map(int, V_hidden_size.split(" ")))
        self._attn_output_hidden_size = [V_input_dim] + list(map(int, attn_output_hidden_size.split(" "))) + [V_input_dim]
        self._field_output_hidden_size = [V_input_dim] + list(map(int, field_output_hidden_size.split(" "))) + [1]

        self._time_kq_dim = self._KQ_hidden_size[-1]
        self._time_v_dim = self._V_hidden_size[-1]
        self._time_kq_head_dim = self._time_kq_dim // self.time_head_num
        self._time_v_head_dim = self._time_v_dim // self.time_head_num
        self._rope_dim = self._time_kq_head_dim if self._time_kq_head_dim % 2 == 0 else self._time_kq_head_dim - 1

        self.K_module = MLPLayer(KQ_input_dim, KQ_hidden_size, activation_id)
        self.Q_module = MLPLayer(KQ_input_dim, KQ_hidden_size, activation_id)
        self.V_module = MLPLayer(V_input_dim, V_hidden_size, activation_id)
        self.attn_output_module = nn.Linear(self._V_hidden_size[-1], V_input_dim)
        self.norm = nn.LayerNorm(V_input_dim)

        self.time_K_module = MLPLayer(V_input_dim, KQ_hidden_size, activation_id)
        self.time_Q_module = MLPLayer(V_input_dim, KQ_hidden_size, activation_id)
        self.time_V_module = MLPLayer(V_input_dim, V_hidden_size, activation_id)
        self.time_attn_output_module = nn.Linear(self._V_hidden_size[-1], V_input_dim)
        self.time_attn_norm = nn.LayerNorm(V_input_dim)
        self.time_ffn_module = MLPLayer(
            V_input_dim,
            attn_output_hidden_size + " " + str(V_input_dim),
            activation_id,
        )
        self.time_ffn_norm = nn.LayerNorm(V_input_dim)

        self.threat_output_module = MLPLayer(
            V_input_dim,
            field_output_hidden_size + " " + str(self._field_output_hidden_size[-1]),
            activation_id,
        )
        self.attack_output_module = MLPLayer(
            V_input_dim,
            field_output_hidden_size + " " + str(self._field_output_hidden_size[-1]),
            activation_id,
        )

        if self._rope_dim > 0:
            inv_freq = 1.0 / (
                10000
                ** (torch.arange(0, self._rope_dim, 2, dtype=torch.float32) / self._rope_dim)
            )
        else:
            inv_freq = torch.empty(0, dtype=torch.float32)
        self.register_buffer("rope_inv_freq", inv_freq, persistent=False)

        self.record_info = {}

    @staticmethod
    def _rotate_half(x: torch.Tensor):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)

    def _apply_rope(self, x: torch.Tensor, positions: torch.Tensor):
        """
        x: [B * N, H, T, D]
        positions: [T]
        """
        if self._rope_dim <= 0:
            return x

        rope_x = x[..., : self._rope_dim]
        pass_x = x[..., self._rope_dim :]

        sinusoid_inp = torch.outer(positions.to(dtype=self.rope_inv_freq.dtype), self.rope_inv_freq)
        sin = torch.repeat_interleave(sinusoid_inp.sin(), 2, dim=-1).to(dtype=x.dtype)
        cos = torch.repeat_interleave(sinusoid_inp.cos(), 2, dim=-1).to(dtype=x.dtype)
        sin = sin.unsqueeze(0).unsqueeze(0)
        cos = cos.unsqueeze(0).unsqueeze(0)

        rope_x = rope_x * cos + self._rotate_half(rope_x) * sin
        if pass_x.numel() > 0:
            return torch.cat((rope_x, pass_x), dim=-1)
        return rope_x

    def _project_temporal_qkv(self, x: torch.Tensor):
        """
        x: [B * N, T, dim]
        returns:
            Q: [B * N, H, T, Dk]
            K: [B * N, H, T, Dk]
            V: [B * N, H, T, Dv]
        """
        batch_agent_size, seq_len, feature_dim = x.shape
        flat_x = x.reshape(batch_agent_size * seq_len, feature_dim)

        K = self.time_K_module(flat_x).view(batch_agent_size, seq_len, self._time_kq_dim)
        Q = self.time_Q_module(flat_x).view(batch_agent_size, seq_len, self._time_kq_dim)
        V = self.time_V_module(flat_x).view(batch_agent_size, seq_len, self._time_v_dim)

        K = K.view(batch_agent_size, seq_len, self.time_head_num, self._time_kq_head_dim).transpose(1, 2)
        Q = Q.view(batch_agent_size, seq_len, self.time_head_num, self._time_kq_head_dim).transpose(1, 2)
        V = V.view(batch_agent_size, seq_len, self.time_head_num, self._time_v_head_dim).transpose(1, 2)
        return Q, K, V

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

    def _temporal_encode(self, x: torch.Tensor, time_offset: int = 0):
        """
        x: [B, T, N, dim]
        returns:
            z: [B, T, N, dim]
        """
        batch_size, seq_len, _, feature_dim = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size * self.agent_num, seq_len, feature_dim)

        Q, K, V = self._project_temporal_qkv(x)
        positions = torch.arange(time_offset, time_offset + seq_len, device=x.device, dtype=torch.long)
        Q = self._apply_rope(Q, positions)
        K = self._apply_rope(K, positions)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self._time_kq_head_dim)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=scores.device),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        heads = torch.matmul(attn_weights, V)
        heads = heads.transpose(1, 2).contiguous().view(batch_size * self.agent_num, seq_len, self._time_v_dim)

        attn_output = self.time_attn_output_module(heads)
        y = self.time_attn_norm(x + attn_output)

        z_delta = self.time_ffn_module(y.reshape(batch_size * self.agent_num * seq_len, feature_dim))
        z_delta = z_delta.view(batch_size * self.agent_num, seq_len, feature_dim)
        z = self.time_ffn_norm(y + z_delta)

        z = z.view(batch_size, self.agent_num, seq_len, feature_dim).permute(0, 2, 1, 3).contiguous()
        return z

    def _output_heads(self, z: torch.Tensor):
        """
        z: [B, T, N, dim]
        returns:
            temporal_output: [B * T * N, dim]
            threat_output:   [B * T, 1]
            attack_output:   [B * T, 1]
        """
        batch_size, seq_len, _, feature_dim = z.shape
        temporal_output = z.reshape(batch_size * seq_len * self.agent_num, feature_dim)

        pooled_input = z.reshape(batch_size * seq_len, self.agent_num, feature_dim)
        pooled_flat = pooled_input.reshape(batch_size * seq_len * self.agent_num, feature_dim)

        threat_output = self.threat_output_module(pooled_flat).view(batch_size * seq_len, self.agent_num, -1).mean(dim=1)
        attack_output = self.attack_output_module(pooled_flat).view(batch_size * seq_len, self.agent_num, -1).mean(dim=1)
        return temporal_output, threat_output, attack_output

    def forward_sequence(self, s: torch.Tensor, a: torch.Tensor, seq_len: int = None, time_offset: int = 0):
        """
        Inputs:
            s: [B * T * N, obs_dim]
            a: [B * T * N, act_dim]
            seq_len:
                None  -> treat input as one episode, infer T from total size
                int   -> treat input as B episodes with the same T
        Returns:
            temporal_output: [B * T * N, dim]
            threat_output:   [B * T, 1]
            attack_output:   [B * T, 1]
        """
        total_agent_steps = s.size(0)
        if total_agent_steps % self.agent_num != 0:
            raise ValueError("AeroTAF_ATTN input mismatch: expected s/a to be [B * T * N, dim].")

        if seq_len is None:
            batch_size = 1
            seq_len = total_agent_steps // self.agent_num
        else:
            denominator = seq_len * self.agent_num
            if denominator <= 0 or total_agent_steps % denominator != 0:
                raise ValueError(
                    "AeroTAF_ATTN input mismatch: expected s/a to be [B * T * N, dim] with the provided seq_len."
                )
            batch_size = total_agent_steps // denominator

        s = s.view(batch_size, seq_len, self.agent_num, s.shape[-1])
        a = a.view(batch_size, seq_len, self.agent_num, a.shape[-1])

        spatial_outputs = []
        for t in range(seq_len):
            spatial_outputs.append(self._spatial_encode(s[:, t], a[:, t]))
        x = torch.stack(spatial_outputs, dim=1)

        z = self._temporal_encode(x, time_offset=time_offset)
        return self._output_heads(z)

    def forward_step(
        self,
        s: torch.Tensor,
        a: torch.Tensor,
        kv_cache: dict = None,
        time_index: int = None,
        max_cache_len: int = None,
    ):
        """
        Inputs:
            s: [B * N, obs_dim]
            a: [B * N, act_dim]
            kv_cache:
                {
                    "k": [B * N, H, L, Dk],
                    "v": [B * N, H, L, Dv],
                    "next_time_index": int,
                }
        Returns:
            temporal_output: [B * N, dim]
            threat_output:   [B, 1]
            attack_output:   [B, 1]
            kv_cache: updated cache
        """
        total_agents = s.size(0)
        if total_agents % self.agent_num != 0:
            raise ValueError("AeroTAF_ATTN step input mismatch: expected s/a to be [B * N, dim].")

        batch_size = total_agents // self.agent_num
        s = s.view(batch_size, self.agent_num, s.shape[-1])
        a = a.view(batch_size, self.agent_num, a.shape[-1])

        x_t = self._spatial_encode(s, a)
        x_t = x_t.view(batch_size * self.agent_num, 1, x_t.shape[-1])

        Q, K, V = self._project_temporal_qkv(x_t)

        if kv_cache is None:
            kv_cache = {}
        if time_index is None:
            time_index = int(kv_cache.get("next_time_index", 0))

        positions = torch.tensor([time_index], device=x_t.device, dtype=torch.long)
        Q = self._apply_rope(Q, positions)
        K = self._apply_rope(K, positions)

        cached_k = kv_cache.get("k")
        cached_v = kv_cache.get("v")
        if cached_k is not None:
            K_all = torch.cat((cached_k, K), dim=2)
            V_all = torch.cat((cached_v, V), dim=2)
        else:
            K_all = K
            V_all = V

        if max_cache_len is not None and max_cache_len > 0:
            K_all = K_all[:, :, -max_cache_len:, :]
            V_all = V_all[:, :, -max_cache_len:, :]

        scores = torch.matmul(Q, K_all.transpose(-2, -1)) / math.sqrt(self._time_kq_head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        heads = torch.matmul(attn_weights, V_all)
        heads = heads.transpose(1, 2).contiguous().view(batch_size * self.agent_num, 1, self._time_v_dim)

        attn_output = self.time_attn_output_module(heads)
        y = self.time_attn_norm(x_t + attn_output)

        z_delta = self.time_ffn_module(y.reshape(batch_size * self.agent_num, y.shape[-1]))
        z_delta = z_delta.view(batch_size * self.agent_num, 1, y.shape[-1])
        z_t = self.time_ffn_norm(y + z_delta)
        z_t = z_t.view(batch_size, 1, self.agent_num, z_t.shape[-1])

        temporal_output, threat_output, attack_output = self._output_heads(z_t)
        new_cache = {
            "k": K_all.detach(),
            "v": V_all.detach(),
            "next_time_index": time_index + 1,
        }
        return temporal_output, threat_output, attack_output, new_cache

    def forward(self, s: torch.Tensor, a: torch.Tensor, seq_len: int = None, time_offset: int = 0):
        return self.forward_sequence(s, a, seq_len=seq_len, time_offset=time_offset)

    @property
    def output_size(self):
        return self._field_output_hidden_size[-1]

    def get_info(self):
        return self.record_info


class AeroTAFATTNBase(nn.Module):
    def __init__(
        self,
        obs_space,
        act_space,
        agent_num: int,
        head_num: int,
        time_head_num: int,
        KQ_hidden_size: str = "",
        V_hidden_size: str = "",
        attn_output_hidden_size: str = "",
        field_output_hidden_size: str = "",
        activation_id=1,
        use_feature_normalization=False,
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

        self.AeroTAF = AeroTAFATTNLayer(
            agent_num=agent_num,
            head_num=head_num,
            time_head_num=time_head_num,
            KQ_input_dim=obs_input_dim,
            V_input_dim=obs_input_dim + act_input_dim,
            activation_id=activation_id,
            KQ_hidden_size=KQ_hidden_size,
            V_hidden_size=V_hidden_size,
            attn_output_hidden_size=attn_output_hidden_size,
            field_output_hidden_size=field_output_hidden_size,
        )

    def forward(self, s, a, seq_len=None, time_offset=0):
        if self._use_feature_normalization:
            s = self.obs_feature_norm(s)
            a = self.act_feature_norm(a)
        temporal_output, threat_output, attack_output = self.AeroTAF(
            s,
            a,
            seq_len=seq_len,
            time_offset=time_offset,
        )
        return temporal_output, threat_output, attack_output

    def forward_step(self, s, a, kv_cache=None, time_index=None, max_cache_len=None):
        if self._use_feature_normalization:
            s = self.obs_feature_norm(s)
            a = self.act_feature_norm(a)
        return self.AeroTAF.forward_step(
            s,
            a,
            kv_cache=kv_cache,
            time_index=time_index,
            max_cache_len=max_cache_len,
        )

    @property
    def output_size(self):
        return self.AeroTAF.output_size

    @property
    def record_info(self):
        return self.AeroTAF.get_info()
