import torch
import numpy as np
from typing import Union, List
from abc import ABC, abstractmethod
from .utils import get_shape_from_space


class Buffer(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def insert(self, **kwargs):
        pass

    @abstractmethod
    def after_update(self):
        pass

    @abstractmethod
    def clear(self):
        pass


class ReplayBuffer(Buffer):

    @staticmethod
    def _flatten(T: int, N: int, M: int, x: np.ndarray):
        return x.reshape(T * N * M, *x.shape[3:])

    @staticmethod
    def _cast(x: np.ndarray):
        # T: buffer_size, n: n_rollout_threads, m: num_agents
        # size: (T, n, m, dim) -> (n, m, T, dim)
        return x.transpose(1, 2, 0, *range(3, x.ndim))

    def __init__(self, args, num_agents, obs_space, act_space):
        # buffer config
        self.buffer_size = args.buffer_size
        self.n_rollout_threads = args.n_rollout_threads
        self.num_agents = num_agents
        self.gamma = args.gamma
        self.use_proper_time_limits = args.use_proper_time_limits
        self.use_gae = args.use_gae
        self.gae_lambda = args.gae_lambda
        # rnn config
        self.recurrent_hidden_size_actor = args.recurrent_hidden_size_actor
        self.recurrent_hidden_size_critic = args.recurrent_hidden_size_critic
        self.recurrent_hidden_layers = args.recurrent_hidden_layers

        obs_shape = get_shape_from_space(obs_space)
        act_shape = get_shape_from_space(act_space)

        # (o_0, a_0, r_0, d_1, o_1, ... , d_T, o_T)
        self.obs = np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_rollout_threads, self.num_agents, *act_shape), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        # NOTE: masks[t] = 1 - dones[t-1], which represents whether obs[t] is a terminal state
        self.masks = np.ones((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        # NOTE: bad_masks[t] = 'bad_transition' in info[t-1], which indicates whether obs[t] a true terminal state or time limit end state
        self.bad_masks = np.ones((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

        # pi(a)
        self.action_log_probs = np.zeros((self.buffer_size, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        # V(o), R(o) while advantage = returns - value_preds
        self.value_preds = np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        # rnn
        self.rnn_states_actor = np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents,
                                          self.recurrent_hidden_layers, self.recurrent_hidden_size_actor),
                                         dtype=np.float32)
        self.rnn_states_critic = np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents,
                                           self.recurrent_hidden_layers, self.recurrent_hidden_size_critic),
                                          dtype=np.float32)
        self.step = 0
        self._chunk_index_cache = {}

    def _get_chunk_layout(self, data_chunk_length: int):
        layout = self._chunk_index_cache.get(data_chunk_length)
        if layout is not None:
            return layout

        per_thread_chunks = self.buffer_size * self.num_agents // data_chunk_length
        chunk_ids = np.arange(per_thread_chunks, dtype=np.int64)
        start_indices = chunk_ids * data_chunk_length
        time_starts = start_indices % self.buffer_size
        agent_starts = start_indices // self.buffer_size

        layout = {
            # 每个线程一共能切出多少个 chunk
            "per_thread_chunks": per_thread_chunks,
            # 每个 chunk 对应的时间窗索引，size: (per_thread_chunks_num, L)
            "time_indices": (time_starts[:, None] + np.arange(data_chunk_length, dtype=np.int64)[None, :]) % self.buffer_size,
            # 每个 chunk 的起始时刻，用来取 RNN 初始状态，size: (per_thread_chunks_num,)
            "state_time_indices": time_starts,
            # 每个 chunk 对应的 agent 重排顺序，保证“当前定位 agent 放在最前面”，size: (per_thread_chunks_num, M)
            "agent_indices": (agent_starts[:, None] + np.arange(self.num_agents, dtype=np.int64)[None, :]) % self.num_agents,
        }
        self._chunk_index_cache[data_chunk_length] = layout
        return layout

    @staticmethod
    def _gather_sequence_batch(x: np.ndarray, env_indices: np.ndarray, agent_indices: np.ndarray, time_indices: np.ndarray):
        # env_indices: (N,)     agent_indices: (N, M)       time_indices: (N, L)
        # size: (n, M, T, -1) -[]-> (N, L, M, -1) -T-> (L, N, M, -1)
        sample = x[env_indices[:, None, None], agent_indices[:, None, :], time_indices[:, :, None]]
        return sample.transpose(1, 0, 2, *range(3, sample.ndim))

    @staticmethod
    def _gather_state_batch(x: np.ndarray, env_indices: np.ndarray, agent_indices: np.ndarray, time_indices: np.ndarray):
        # env_indices: (N,)     agent_indices: (N, M)       time_indices: (N,)
        # size: (n, M, T, -1) -[]-> (N, M, -1)
        return x[env_indices[:, None], agent_indices, time_indices[:, None]]

    @property
    def advantages(self) -> np.ndarray:
        advantages = self.returns[:-1] - self.value_preds[:-1]  # type: np.ndarray
        return (advantages - advantages.mean()) / (advantages.std() + 1e-5)

    def insert(self,
               obs: np.ndarray,
               actions: np.ndarray,
               rewards: np.ndarray,
               masks: np.ndarray,
               action_log_probs: np.ndarray,
               value_preds: np.ndarray,
               rnn_states_actor: np.ndarray,
               rnn_states_critic: np.ndarray,
               bad_masks: Union[np.ndarray, None] = None,
               **kwargs):
        """Insert numpy data.
        Args:
            obs:                o_{t+1}
            actions:            a_{t}
            rewards:            r_{t}
            masks:              mask[t+1] = 1 - done_{t}
            action_log_probs:   log_prob(a_{t})
            value_preds:        value(o_{t})
            rnn_states_actor:   ha_{t+1}
            rnn_states_critic:  hc_{t+1}
        """
        self.obs[self.step + 1] = obs.copy()
        self.actions[self.step] = actions.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rnn_states_actor[self.step + 1] = rnn_states_actor.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()

        self.step = (self.step + 1) % self.buffer_size

    def after_update(self):
        """Copy last timestep data to first index. Called after update to model."""
        self.obs[0] = self.obs[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.rnn_states_actor[0] = self.rnn_states_actor[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()

    def clear(self):
        self.step = 0
        self.obs = np.zeros_like(self.obs, dtype=np.float32)
        self.actions = np.zeros_like(self.actions, dtype=np.float32)
        self.rewards = np.zeros_like(self.rewards, dtype=np.float32)
        self.masks = np.ones_like(self.masks, dtype=np.float32)
        self.bad_masks = np.ones_like(self.bad_masks, dtype=np.float32)
        self.action_log_probs = np.zeros_like(self.action_log_probs, dtype=np.float32)
        self.value_preds = np.zeros_like(self.value_preds, dtype=np.float32)
        self.returns = np.zeros_like(self.returns, dtype=np.float32)
        self.rnn_states_actor = np.zeros_like(self.rnn_states_actor)
        self.rnn_states_critic = np.zeros_like(self.rnn_states_critic)

    def compute_returns(self, next_value: np.ndarray):
        """
        Compute returns either as discounted sum of rewards, or using GAE.

        Args:
            next_value(np.ndarray): value predictions for the step after the last episode step.
        """
        if self.use_proper_time_limits:
            if self.use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    td_delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                    gae = td_delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                    gae = gae * self.bad_masks[step + 1]
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]) \
                        * self.bad_masks[step + 1] + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if self.use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    td_delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                    gae = td_delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]

    @staticmethod
    def recurrent_generator(buffer: Union[Buffer, List[Buffer]], num_mini_batch: int, data_chunk_length: int):
        """
        A recurrent generator that yields training data for chunked RNN training arranged in mini batches.
        This generator shuffles the data by sequences.

        Args:
            buffers (Buffer or List[Buffer])
            num_mini_batch (int): number of minibatches to split the batch into.
            data_chunk_length (int): length of sequence chunks with which to train RNN.

        Returns:
            (obs_batch, actions_batch, masks_batch, old_action_log_probs_batch, advantages_batch, \
                returns_batch, value_preds_batch, rnn_states_actor_batch, rnn_states_critic_batch)
        """
        buffer = [buffer] if isinstance(buffer, ReplayBuffer) else buffer  # type: List[ReplayBuffer]
        n_rollout_threads = buffer[0].n_rollout_threads
        buffer_size = buffer[0].buffer_size
        num_agents = buffer[0].num_agents
        assert all([b.n_rollout_threads == n_rollout_threads for b in buffer]) \
            and all([b.buffer_size == buffer_size for b in buffer]) \
            and all([b.num_agents == num_agents for b in buffer]) \
            and all([isinstance(b, ReplayBuffer) for b in buffer]), \
            "Input buffers must has the same type and shape"
        buffer_size = buffer_size * len(buffer)

        assert n_rollout_threads * buffer_size >= data_chunk_length, (
            "PPO requires the number of processes ({}) * buffer size ({}) * num_agents ({})"
            "to be greater than or equal to the number of "
            "data chunk length ({}).".format(n_rollout_threads, buffer_size, num_agents, data_chunk_length))

        # Transpose and reshape parallel data into sequential data
        obs = np.vstack([ReplayBuffer._cast(buf.obs[:-1]) for buf in buffer])
        actions = np.vstack([ReplayBuffer._cast(buf.actions) for buf in buffer])
        masks = np.vstack([ReplayBuffer._cast(buf.masks[:-1]) for buf in buffer])
        old_action_log_probs = np.vstack([ReplayBuffer._cast(buf.action_log_probs) for buf in buffer])
        advantages = np.vstack([ReplayBuffer._cast(buf.advantages) for buf in buffer])
        returns = np.vstack([ReplayBuffer._cast(buf.returns[:-1]) for buf in buffer])
        value_preds = np.vstack([ReplayBuffer._cast(buf.value_preds[:-1]) for buf in buffer])
        rnn_states_actor = np.vstack([ReplayBuffer._cast(buf.rnn_states_actor[:-1]) for buf in buffer])
        rnn_states_critic = np.vstack([ReplayBuffer._cast(buf.rnn_states_critic[:-1]) for buf in buffer])

        # Get mini-batch size and shuffle chunk data
        data_chunks = n_rollout_threads * buffer_size // data_chunk_length
        mini_batch_size = data_chunks // num_mini_batch
        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        for indices in sampler:
            obs_batch = []
            actions_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            advantages_batch = []
            returns_batch = []
            value_preds_batch = []
            rnn_states_actor_batch = []
            rnn_states_critic_batch = []

            for index in indices:

                ind = index * data_chunk_length
                # size [T+1, N, Dim] => [T, N, Dim] => [N, T, Dim] => [N * T, Dim] => [L, Dim]
                obs_batch.append(obs[ind:ind + data_chunk_length])
                actions_batch.append(actions[ind:ind + data_chunk_length])
                masks_batch.append(masks[ind:ind + data_chunk_length])
                old_action_log_probs_batch.append(old_action_log_probs[ind:ind + data_chunk_length])
                advantages_batch.append(advantages[ind:ind + data_chunk_length])
                returns_batch.append(returns[ind:ind + data_chunk_length])
                value_preds_batch.append(value_preds[ind:ind + data_chunk_length])
                # size [T+1, N, Dim] => [T, N, Dim] => [N, T, Dim] => [N * T, Dim] => [1, Dim]
                rnn_states_actor_batch.append(rnn_states_actor[ind])
                rnn_states_critic_batch.append(rnn_states_critic[ind])

            L, N = data_chunk_length, mini_batch_size

            # These are all from_numpys of size (L, N, Dim)
            obs_batch = np.stack(obs_batch, axis=1)
            actions_batch = np.stack(actions_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, axis=1)
            advantages_batch = np.stack(advantages_batch, axis=1)
            returns_batch = np.stack(returns_batch, axis=1)
            value_preds_batch = np.stack(value_preds_batch, axis=1)

            # States is just a (N, -1) from_numpy
            rnn_states_actor_batch = np.stack(rnn_states_actor_batch).reshape(N, *buffer[0].rnn_states_actor.shape[3:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(N, *buffer[0].rnn_states_critic.shape[3:])

            # Flatten the (L, N, ...) from_numpys to (L * N, ...)
            obs_batch = ReplayBuffer._flatten(L, N, obs_batch)
            actions_batch = ReplayBuffer._flatten(L, N, actions_batch)
            masks_batch = ReplayBuffer._flatten(L, N, masks_batch)
            old_action_log_probs_batch = ReplayBuffer._flatten(L, N, old_action_log_probs_batch)
            advantages_batch = ReplayBuffer._flatten(L, N, advantages_batch)
            returns_batch = ReplayBuffer._flatten(L, N, returns_batch)
            value_preds_batch = ReplayBuffer._flatten(L, N, value_preds_batch)

            yield obs_batch, actions_batch, masks_batch, old_action_log_probs_batch, advantages_batch, \
                returns_batch, value_preds_batch, rnn_states_actor_batch, rnn_states_critic_batch


class SharedReplayBuffer(ReplayBuffer):

    def __init__(self, args, num_agents, obs_space, share_obs_space, act_space):
        # env config
        self.num_agents = num_agents
        self.n_rollout_threads = args.n_rollout_threads
        # buffer config
        self.gamma = args.gamma
        self.buffer_size = args.buffer_size
        self.use_proper_time_limits = args.use_proper_time_limits
        self.use_gae = args.use_gae
        self.gae_lambda = args.gae_lambda
        # rnn config
        self.recurrent_hidden_size_actor = args.recurrent_hidden_size_actor
        self.recurrent_hidden_size_critic = args.recurrent_hidden_size_critic
        self.recurrent_hidden_layers = args.recurrent_hidden_layers

        obs_shape = get_shape_from_space(obs_space)
        share_obs_shape = get_shape_from_space(share_obs_space)
        act_shape = get_shape_from_space(act_space)

        # (o_0, s_0, a_0, r_0, d_0, ..., o_T, s_T)
        self.obs = np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, *obs_shape), dtype=np.float32)
        self.share_obs = np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, *share_obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_rollout_threads, self.num_agents, *act_shape), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        # NOTE: masks[t] = 1 - dones[t-1], which represents whether obs[t] is a terminal state .... same for all agents
        self.masks = np.ones((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        self.bad_masks = np.ones_like(self.masks)
        # NOTE: active_masks[t, :, i] represents whether agent[i] is alive in obs[t] .... differ in different agents
        self.active_masks = np.ones_like(self.masks)
        # pi(a)
        self.action_log_probs = np.zeros((self.buffer_size, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        # V(o), R(o) while advantage = returns - value_preds
        self.value_preds = np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        # rnn
        self.rnn_states_actor = np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents,
                                          self.recurrent_hidden_layers, self.recurrent_hidden_size_actor),
                                         dtype=np.float32)
        self.rnn_states_critic = np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents,
                                           self.recurrent_hidden_layers, self.recurrent_hidden_size_critic),
                                          dtype=np.float32)
        self.step = 0
        self._chunk_index_cache = {}

    def insert(self,
               obs: np.ndarray,
               share_obs: np.ndarray,
               actions: np.ndarray,
               rewards: np.ndarray,
               masks: np.ndarray,
               action_log_probs: np.ndarray,
               value_preds: np.ndarray,
               rnn_states_actor: np.ndarray,
               rnn_states_critic: np.ndarray,
               bad_masks: Union[np.ndarray, None] = None,
               active_masks: Union[np.ndarray, None] = None,
               available_actions: Union[np.ndarray, None] = None):
        """Insert numpy data.
        Args:
            obs:                o_{t+1}
            share_obs:          s_{t+1}
            actions:            a_{t}
            rewards:            r_{t}
            masks:              1 - done_{t}
            action_log_probs:   log_prob(a_{t})
            value_preds:        value(o_{t})
            rnn_states_actor:   ha_{t+1}
            rnn_states_critic:  hc_{t+1}
            active_masks:       1 - agent_done_{t}
        """
        self.share_obs[self.step + 1] = share_obs.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()
        if available_actions is not None:
            pass
        return super().insert(obs, actions, rewards, masks, action_log_probs, value_preds, rnn_states_actor, rnn_states_critic)

    def after_update(self):
        self.active_masks[0] = self.active_masks[-1].copy()
        self.share_obs[0] = self.share_obs[-1].copy()
        return super().after_update()

    def recurrent_generator(self, advantages: np.ndarray, num_mini_batch: int, data_chunk_length: int):
        """
        A recurrent generator that yields training data for chunked RNN training arranged in mini batches.
        This generator shuffles the data by sequences.

        Args:
            advantages (np.ndarray): advantage estimates.
            num_mini_batch (int): number of minibatches to split the batch into.
            data_chunk_length (int): length of sequence chunks with which to train RNN.

        Returns:
            (obs_batch, share_obs_batch, actions_batch, masks_batch, active_masks_batch, \
                old_action_log_probs_batch, advantages_batch, returns_batch, value_preds_batch, \
                rnn_states_actor_batch, rnn_states_critic_batch)
        """
        assert self.n_rollout_threads * self.buffer_size >= data_chunk_length, (
            "PPO requires the number of processes ({}) * buffer size ({}) "
            "to be greater than or equal to the number of data chunk length ({}).".format(
                self.n_rollout_threads, self.buffer_size, data_chunk_length))

        # Transpose and reshape parallel data into sequential data
        # size: [T, n, m, dim] -> [n, m, T, dim]
        obs = self._cast(self.obs[:-1])
        share_obs = self._cast(self.share_obs[:-1])
        actions = self._cast(self.actions)
        masks = self._cast(self.masks[:-1])
        active_masks = self._cast(self.active_masks[:-1])
        old_action_log_probs = self._cast(self.action_log_probs)
        advantages = self._cast(advantages)
        returns = self._cast(self.returns[:-1])
        value_preds = self._cast(self.value_preds[:-1])
        rnn_states_actor = self._cast(self.rnn_states_actor[:-1])
        rnn_states_critic = self._cast(self.rnn_states_critic[:-1])
        # size: (T, n, m, dim) -sum-> (T, n, 1, dim) -broadcast-> (T, n, m, dim)
        rewards = np.broadcast_to(self.rewards.sum(axis=2, keepdims=True), self.rewards.shape)
        rewards = self._cast(rewards)
        layout = self._get_chunk_layout(data_chunk_length)
        per_thread_chunks = layout["per_thread_chunks"]

        # Get mini-batch size and shuffle chunk data
        data_chunks = self.n_rollout_threads * per_thread_chunks
        mini_batch_size = data_chunks // num_mini_batch
        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        for indices in sampler:
            env_indices = indices // per_thread_chunks  # 记录indices中的chunk都在第几个线程，size: (N,)
            local_indices = indices % per_thread_chunks  # 记录indices中的chunk在所在线程中的索引，size: (N,)
            agent_indices = layout["agent_indices"][local_indices]  # 按照在当前线程中的索引获取agents的顺序，size: (N, m)
            time_indices = layout["time_indices"][local_indices]  # 按照在当前线程中的索引获取时间序列，size: (N, L)
            state_time_indices = layout["state_time_indices"][local_indices]  # 按照在当前线程中的索引获取时间序列开头，size: (N,)

            L, N, M = data_chunk_length, mini_batch_size, self.num_agents

            # These are all from_numpys of size (L, N, M, -1)
            obs_batch = self._gather_sequence_batch(obs, env_indices, agent_indices, time_indices)
            share_obs_batch = self._gather_sequence_batch(share_obs, env_indices, agent_indices, time_indices)
            actions_batch = self._gather_sequence_batch(actions, env_indices, agent_indices, time_indices)
            masks_batch = self._gather_sequence_batch(masks, env_indices, agent_indices, time_indices)
            active_masks_batch = self._gather_sequence_batch(active_masks, env_indices, agent_indices, time_indices)
            old_action_log_probs_batch = self._gather_sequence_batch(old_action_log_probs, env_indices, agent_indices,
                                                                     time_indices)
            advantages_batch = self._gather_sequence_batch(advantages, env_indices, agent_indices, time_indices)
            returns_batch = self._gather_sequence_batch(returns, env_indices, agent_indices, time_indices)
            value_preds_batch = self._gather_sequence_batch(value_preds, env_indices, agent_indices, time_indices)
            rewards_batch = self._gather_sequence_batch(rewards, env_indices, agent_indices, time_indices)

            # States is just a (N, M, -1) from_numpy
            rnn_states_actor_batch = self._gather_state_batch(
                rnn_states_actor, env_indices, agent_indices, state_time_indices
            ).reshape(N * M, *self.rnn_states_actor.shape[3:])
            rnn_states_critic_batch = self._gather_state_batch(
                rnn_states_critic, env_indices, agent_indices, state_time_indices
            ).reshape(N * M, *self.rnn_states_critic.shape[3:])

            # Flatten the (L, N, M, ...) from_numpys to (L * N * M, ...)
            obs_batch = self._flatten(L, N, M, obs_batch)
            share_obs_batch = self._flatten(L, N, M, share_obs_batch)
            actions_batch = self._flatten(L, N, M, actions_batch)
            masks_batch = self._flatten(L, N, M, masks_batch)
            active_masks_batch = self._flatten(L, N, M, active_masks_batch)
            old_action_log_probs_batch = self._flatten(L, N, M, old_action_log_probs_batch)
            advantages_batch = self._flatten(L, N, M, advantages_batch)
            returns_batch = self._flatten(L, N, M, returns_batch)
            value_preds_batch = self._flatten(L, N, M, value_preds_batch)
            rewards_batch = self._flatten(L, N, M, rewards_batch)

            yield obs_batch, share_obs_batch, actions_batch, masks_batch, active_masks_batch, \
                old_action_log_probs_batch, advantages_batch, returns_batch, value_preds_batch, \
                rnn_states_actor_batch, rnn_states_critic_batch, rewards_batch
