import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .flatten import build_flattener
from .mlp import MLPLayer


class AeroTAFATNNFastLayer(nn.Module):
    """
    Fast temporal-attention AeroTAF.

    This variant is for many-to-one window training/inference. Spatial
    interaction is still encoded for every timestep, but temporal attention
    only builds Q from the last timestep while K/V come from the whole window.
    The temporal attention score therefore has shape [B * N, H, 1, T] instead
    of [B * N, H, T, T].
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
        super(AeroTAFATNNFastLayer, self).__init__()
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
        x: [B * N, H, L, D]
        positions: [L]
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

    def _spatial_encode(self, s_t: torch.Tensor, a_t: torch.Tensor):
        """
        s_t: [B, N, obs_dim]
        a_t: [B, N, act_dim]
        returns:
            x: [B, N, obs_dim + act_dim]
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

    def _spatial_encode_variants(self, s_t: torch.Tensor, a_variants: torch.Tensor):
        """Encode many current-action variants while sharing state-only Q/K."""
        batch_size, variant_count, _, action_dim = a_variants.shape
        state_dim = s_t.shape[-1]
        s_variants = s_t[:, None].expand(-1, variant_count, -1, -1)
        x = torch.cat((s_variants, a_variants), dim=-1)

        kq_dim = self._KQ_hidden_size[-1]
        v_dim = self._V_hidden_size[-1]
        kq_head_dim = kq_dim // self.head_num
        v_head_dim = v_dim // self.head_num

        s_flat = s_t.reshape(batch_size * self.agent_num, state_dim)
        K = self.K_module(s_flat).view(batch_size, self.agent_num, kq_dim)
        Q = self.Q_module(s_flat).view(batch_size, self.agent_num, kq_dim)
        K = K.view(batch_size, self.agent_num, self.head_num, kq_head_dim).transpose(1, 2)
        Q = Q.view(batch_size, self.agent_num, self.head_num, kq_head_dim).transpose(1, 2)
        weights = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(kq_head_dim), dim=-1)

        V = self.V_module(x.reshape(batch_size * variant_count * self.agent_num, state_dim + action_dim))
        V = V.view(batch_size, variant_count, self.agent_num, self.head_num, v_head_dim)
        V = V.permute(0, 1, 3, 2, 4)
        heads = torch.matmul(weights[:, None], V)
        heads = heads.permute(0, 1, 3, 2, 4).contiguous().view(
            batch_size, variant_count, self.agent_num, v_dim
        )
        return self.norm(x + self.attn_output_module(heads))

    def _project_temporal_kv(self, x: torch.Tensor):
        """
        x: [B * N, T, dim]
        returns:
            K: [B * N, H, T, Dk]
            V: [B * N, H, T, Dv]
        """
        batch_agent_size, seq_len, feature_dim = x.shape
        flat_x = x.reshape(batch_agent_size * seq_len, feature_dim)

        K = self.time_K_module(flat_x).view(batch_agent_size, seq_len, self._time_kq_dim)
        V = self.time_V_module(flat_x).view(batch_agent_size, seq_len, self._time_v_dim)

        K = K.view(batch_agent_size, seq_len, self.time_head_num, self._time_kq_head_dim).transpose(1, 2)
        V = V.view(batch_agent_size, seq_len, self.time_head_num, self._time_v_head_dim).transpose(1, 2)
        return K, V

    def _project_temporal_q_last(self, x_last: torch.Tensor):
        """
        x_last: [B * N, 1, dim]
        returns:
            Q: [B * N, H, 1, Dk]
        """
        batch_agent_size, _, feature_dim = x_last.shape
        Q = self.time_Q_module(x_last.reshape(batch_agent_size, feature_dim))
        Q = Q.view(batch_agent_size, 1, self.time_head_num, self._time_kq_head_dim).transpose(1, 2)
        return Q

    def _temporal_encode_last(self, x: torch.Tensor, time_offset: int = 0):
        """
        x: [B, T, N, dim]
        returns:
            z_last: [B, 1, N, dim]
        """
        batch_size, seq_len, _, feature_dim = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size * self.agent_num, seq_len, feature_dim)
        x_last = x[:, -1:, :]

        K, V = self._project_temporal_kv(x)
        Q = self._project_temporal_q_last(x_last)

        key_positions = torch.arange(time_offset, time_offset + seq_len, device=x.device, dtype=torch.long)
        query_positions = torch.tensor([time_offset + seq_len - 1], device=x.device, dtype=torch.long)
        K = self._apply_rope(K, key_positions)
        Q = self._apply_rope(Q, query_positions)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self._time_kq_head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        heads = torch.matmul(attn_weights, V)
        heads = heads.transpose(1, 2).contiguous().view(batch_size * self.agent_num, 1, self._time_v_dim)

        attn_output = self.time_attn_output_module(heads)
        y = self.time_attn_norm(x_last + attn_output)

        z_delta = self.time_ffn_module(y.reshape(batch_size * self.agent_num, feature_dim))
        z_delta = z_delta.view(batch_size * self.agent_num, 1, feature_dim)
        z = self.time_ffn_norm(y + z_delta)

        z = z.view(batch_size, self.agent_num, 1, feature_dim).permute(0, 2, 1, 3).contiguous()
        return z

    def _output_heads(self, z_last: torch.Tensor):
        """
        z_last: [B, 1, N, dim]
        returns:
            temporal_output: [B * N, dim]
            threat_output:   [B, 1]
            attack_output:   [B, 1]
        """
        batch_size, seq_len, _, feature_dim = z_last.shape
        temporal_output = z_last.reshape(batch_size * seq_len * self.agent_num, feature_dim)
        pooled_flat = z_last.reshape(batch_size * seq_len * self.agent_num, feature_dim)

        threat_output = self.threat_output_module(pooled_flat).view(batch_size * seq_len, self.agent_num, -1).mean(dim=1)
        attack_output = self.attack_output_module(pooled_flat).view(batch_size * seq_len, self.agent_num, -1).mean(dim=1)
        return temporal_output, threat_output, attack_output

    def build_trajectory_cache(self, s: torch.Tensor, a: torch.Tensor):
        """Project factual trajectory features once for cached sliding-window inference."""
        if s.ndim != 4 or a.ndim != 4:
            raise ValueError("cached AeroTAF inputs must be [env, time, agent, feature]")
        n_envs, time_steps, agent_num, _ = s.shape
        if agent_num != self.agent_num or a.shape[:3] != s.shape[:3]:
            raise ValueError("cached AeroTAF state/action shapes do not match")

        spatial = self._spatial_encode(
            s.reshape(n_envs * time_steps, agent_num, s.shape[-1]),
            a.reshape(n_envs * time_steps, agent_num, a.shape[-1]),
        ).view(n_envs, time_steps, agent_num, -1)
        flat = spatial.reshape(n_envs * time_steps * agent_num, spatial.shape[-1])
        temporal_k = self.time_K_module(flat).view(
            n_envs,
            time_steps,
            agent_num,
            self.time_head_num,
            self._time_kq_head_dim,
        )
        temporal_v = self.time_V_module(flat).view(
            n_envs,
            time_steps,
            agent_num,
            self.time_head_num,
            self._time_v_head_dim,
        )
        return {
            "states": s,
            "actions": a,
            "spatial": spatial,
            "temporal_k": temporal_k,
            "temporal_v": temporal_v,
        }

    def _cached_history(self, cache, env_indices, time_indices, segment_starts, history_windows):
        window = int(history_windows)
        offsets = torch.arange(window - 1, -1, -1, device=time_indices.device)
        history_indices = time_indices[:, None] - offsets[None]
        starts = segment_starts[env_indices, time_indices]
        valid = history_indices >= starts[:, None]
        safe_indices = torch.maximum(history_indices, starts[:, None])

        K = cache["temporal_k"][env_indices[:, None], safe_indices]
        V = cache["temporal_v"][env_indices[:, None], safe_indices]
        K = K.permute(0, 2, 3, 1, 4).contiguous()
        V = V.permute(0, 2, 3, 1, 4).contiguous()

        batch_size = env_indices.numel()
        positions = torch.arange(window, device=time_indices.device)
        K = self._apply_rope(
            K.view(batch_size * self.agent_num, self.time_head_num, window, self._time_kq_head_dim),
            positions,
        ).view(batch_size, self.agent_num, self.time_head_num, window, self._time_kq_head_dim)
        return K, V, valid

    def _finish_cached_temporal(self, x_last, attended):
        leading_shape = x_last.shape[:-2]
        feature_dim = x_last.shape[-1]
        flat_size = math.prod(leading_shape) * self.agent_num
        x_flat = x_last.reshape(flat_size, feature_dim)
        attended_flat = attended.reshape(flat_size, self._time_v_dim)

        y = self.time_attn_norm(x_flat + self.time_attn_output_module(attended_flat))
        z = self.time_ffn_norm(y + self.time_ffn_module(y))
        z = z.view(*leading_shape, self.agent_num, feature_dim)

        pooled = z.reshape(flat_size, feature_dim)
        threat = self.threat_output_module(pooled).view(*leading_shape, self.agent_num, -1).mean(dim=-2)
        attack = self.attack_output_module(pooled).view(*leading_shape, self.agent_num, -1).mean(dim=-2)
        return threat, attack

    def predict_cached(
        self,
        cache,
        env_indices: torch.Tensor,
        time_indices: torch.Tensor,
        segment_starts: torch.Tensor,
        history_windows: int,
        action_variants: torch.Tensor,
    ):
        """Predict endpoint variants while sharing all factual history projections."""
        env_indices = env_indices.to(dtype=torch.long)
        time_indices = time_indices.to(dtype=torch.long)
        batch_size, variant_count = action_variants.shape[:2]
        if batch_size != env_indices.numel() or action_variants.shape[2] != self.agent_num:
            raise ValueError("cached AeroTAF endpoint batch does not match action variants")

        history_k, history_v, valid = self._cached_history(
            cache, env_indices, time_indices, segment_starts, history_windows
        )
        current_states = cache["states"][env_indices, time_indices]
        x_last = self._spatial_encode_variants(current_states, action_variants)
        flat = x_last.reshape(batch_size * variant_count * self.agent_num, x_last.shape[-1])

        query = self.time_Q_module(flat).view(
            batch_size, variant_count, self.agent_num, self.time_head_num, self._time_kq_head_dim
        )
        endpoint_k = self.time_K_module(flat).view_as(query)
        endpoint_v = self.time_V_module(flat).view(
            batch_size, variant_count, self.agent_num, self.time_head_num, self._time_v_head_dim
        )

        current_position = torch.tensor([int(history_windows) - 1], device=flat.device)
        query = self._apply_rope(
            query.reshape(-1, self.time_head_num, 1, self._time_kq_head_dim), current_position
        ).reshape_as(query)
        endpoint_k = self._apply_rope(
            endpoint_k.reshape(-1, self.time_head_num, 1, self._time_kq_head_dim), current_position
        ).reshape_as(endpoint_k)

        scores = torch.einsum("bvnhd,bnhld->bvnhl", query, history_k)
        scores = scores / math.sqrt(self._time_kq_head_dim)
        scores[..., -1] = (query * endpoint_k).sum(dim=-1) / math.sqrt(self._time_kq_head_dim)
        scores = scores.masked_fill(~valid[:, None, None, None, :], torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1)

        attended = torch.einsum("bvnhl,bnhld->bvnhd", weights, history_v)
        factual_endpoint_v = history_v[..., -1, :][:, None]
        attended = attended + weights[..., -1, None] * (endpoint_v - factual_endpoint_v)
        attended = attended.reshape(batch_size, variant_count, self.agent_num, self._time_v_dim)
        return self._finish_cached_temporal(x_last, attended)

    def forward_sequence(self, s: torch.Tensor, a: torch.Tensor, seq_len: int = None, time_offset: int = 0):
        """
        Inputs:
            s: [B * T * N, obs_dim]
            a: [B * T * N, act_dim]
            seq_len:
                None -> treat input as one episode/window
                int  -> treat input as B same-length windows
        Returns only the last timestep:
            temporal_output: [B * N, dim]
            threat_output:   [B, 1]
            attack_output:   [B, 1]
        """
        total_agent_steps = s.size(0)
        if total_agent_steps % self.agent_num != 0:
            raise ValueError("AeroTAF_ATNN_Fast input mismatch: expected s/a to be [B * T * N, dim].")

        if seq_len is None:
            batch_size = 1
            seq_len = total_agent_steps // self.agent_num
        else:
            denominator = seq_len * self.agent_num
            if denominator <= 0 or total_agent_steps % denominator != 0:
                raise ValueError(
                    "AeroTAF_ATNN_Fast input mismatch: expected s/a to be [B * T * N, dim] with the provided seq_len."
                )
            batch_size = total_agent_steps // denominator

        s = s.view(batch_size, seq_len, self.agent_num, s.shape[-1])
        a = a.view(batch_size, seq_len, self.agent_num, a.shape[-1])

        # Spatial attention is independent across timesteps. Folding B and T
        # avoids launching the same modules once per history position.
        spatial = self._spatial_encode(
            s.reshape(batch_size * seq_len, self.agent_num, s.shape[-1]),
            a.reshape(batch_size * seq_len, self.agent_num, a.shape[-1]),
        )
        x = spatial.view(batch_size, seq_len, self.agent_num, spatial.shape[-1])

        z_last = self._temporal_encode_last(x, time_offset=time_offset)
        return self._output_heads(z_last)

    def forward(self, s: torch.Tensor, a: torch.Tensor, seq_len: int = None, time_offset: int = 0):
        return self.forward_sequence(s, a, seq_len=seq_len, time_offset=time_offset)

    @property
    def output_size(self):
        return self._field_output_hidden_size[-1]

    def get_info(self):
        return self.record_info


class AeroTAFATNNFastBase(nn.Module):
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

        self.AeroTAF = AeroTAFATNNFastLayer(
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

    def build_trajectory_cache(self, s, a):
        if self._use_feature_normalization:
            s = self.obs_feature_norm(s)
            a = self.act_feature_norm(a)
        return self.AeroTAF.build_trajectory_cache(s, a)

    def predict_cached(
        self,
        cache,
        env_indices,
        time_indices,
        segment_starts,
        history_windows,
        action_variants,
    ):
        if self._use_feature_normalization:
            action_variants = self.act_feature_norm(action_variants)
        return self.AeroTAF.predict_cached(
            cache,
            env_indices,
            time_indices,
            segment_starts,
            history_windows,
            action_variants,
        )

    @property
    def output_size(self):
        return self.AeroTAF.output_size

    @property
    def record_info(self):
        return self.AeroTAF.get_info()


AeroTAFATTNFastLayer = AeroTAFATNNFastLayer
AeroTAFATTNFastBase = AeroTAFATNNFastBase
