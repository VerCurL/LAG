
"""执行层 PPO。每个训练片段保留采样时的 GRU 状态和时间顺序。"""

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch.distributions import Categorical

from laglayer3.execution.observation import OBSERVATION_SCHEMA, ROUTE_FEATURES, STATE_DIM
from laglayer3.policies.model.execution import ExecutionRouteNetwork

CHECKPOINT_SCHEMA = "laglayer3.execution_ppo.v1"


def model_kwargs(env, config):
    return dict(route_points=env.window_size, route_features=ROUTE_FEATURES,
                state_dim=STATE_DIM, hidden_dim=config["hidden_dim"])


def distributions(logits):
    return tuple(Categorical(logits=head) for head in logits)


@torch.no_grad()
def evaluate_action(model, observation, hidden, device, deterministic=False):
    route, state = (torch.as_tensor(x, device=device)[None, None] for x in observation)
    logits, value, hidden = model.actor_critic(route, state, hidden)
    heads = distributions(logits)
    actions = torch.stack([
        head.probs.argmax(dim=-1) if deterministic else head.sample() for head in heads
    ], dim=-1)
    log_prob = sum(head.log_prob(actions[..., index]) for index, head in enumerate(heads))
    return actions[0, 0].cpu().numpy(), log_prob.item(), value.item(), hidden


@dataclass
class Transition:
    observation: tuple
    hidden: torch.Tensor
    action: np.ndarray
    log_prob: float
    value: float
    reward: float
    next_value: float
    episode_end: bool


def advantages_for(transitions, gamma, gae_lambda):
    advantages = np.empty(len(transitions), dtype=np.float32)
    running = 0.0
    for index in reversed(range(len(transitions))):
        step = transitions[index]
        delta = step.reward + gamma * step.next_value - step.value
        running = delta + gamma * gae_lambda * running * (not step.episode_end)
        advantages[index] = running
    returns = advantages + np.asarray([step.value for step in transitions], dtype=np.float32)
    return (advantages - advantages.mean()) / max(float(advantages.std()), 1e-8), returns


class ExecutionPPO:
    def __init__(self, env, config, seed, device):
        self.env, self.config, self.device = env, config, torch.device(device)
        self.random = np.random.default_rng(seed)
        self.model_kwargs = model_kwargs(env, config)
        torch.manual_seed(seed)
        self.model = ExecutionRouteNetwork(**self.model_kwargs).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["learning_rate"])
        for key in ("rollout_steps", "update_epochs", "chunk_length", "minibatch_chunks"):
            value = config[key]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("{} 必须是正整数".format(key))
        self.observation = env.reset()
        self.hidden = self.zero_hidden()
        self.episode_return = 0.0
        self.total_steps = 0

    def zero_hidden(self):
        return torch.zeros((1, 1, self.model_kwargs["hidden_dim"]), device=self.device)

    def collect(self):
        transitions = []
        episodes = []
        for _ in range(self.config["rollout_steps"]):
            action, log_prob, value, hidden = evaluate_action(
                self.model, self.observation, self.hidden, self.device
            )
            observation, reward, terminated, truncated, metrics = self.env.step(action)
            self.episode_return += reward
            if terminated:
                next_value = 0.0
            else:
                next_value = self.bootstrap(observation, hidden)
            done = terminated or truncated
            transitions.append(Transition(self.observation, self.hidden.cpu(), action, log_prob,
                                          value, reward, next_value, done))
            if done:
                episodes.append(dict(metrics, episode_return=self.episode_return))
                self.observation, self.hidden = self.env.reset(), self.zero_hidden()
                self.episode_return = 0.0
            else:
                self.observation, self.hidden = observation, hidden
        self.total_steps += len(transitions)
        return transitions, episodes

    @torch.no_grad()
    def bootstrap(self, observation, hidden):
        route, state = (torch.as_tensor(x, device=self.device)[None, None] for x in observation)
        _, value, _ = self.model.actor_critic(route, state, hidden)
        return value.item()

    def update(self, transitions):
        config = self.config
        advantages, returns = advantages_for(transitions, config["gamma"], config["gae_lambda"])
        # GRU 片段不会跨过 reset，末尾填充通过掩码排除。
        chunks = []
        start = 0
        for index, step in enumerate(transitions, 1):
            if step.episode_end or index - start == config["chunk_length"] or index == len(transitions):
                chunks.append(range(start, index))
                start = index
        losses = []
        width = config["minibatch_chunks"]
        for _ in range(config["update_epochs"]):
            order = self.random.permutation(len(chunks))
            for offset in range(0, len(order), width):
                batch = [chunks[index] for index in order[offset:offset + width]]
                losses.append(self.update_batch(batch, transitions, advantages, returns))
        return {key: float(np.mean([row[key] for row in losses])) for key in losses[0]}

    def update_batch(self, batch, transitions, advantages, returns):
        length = max(map(len, batch))
        indices = np.zeros((len(batch), length), dtype=np.int64)
        valid = np.zeros_like(indices, dtype=bool)
        for row, chunk in enumerate(batch):
            indices[row, :len(chunk)] = chunk
            valid[row, :len(chunk)] = True
        steps = [transitions[index] for index in indices.ravel()]
        def pack(values, dtype=None):
            tensor = torch.as_tensor(np.asarray(values), device=self.device, dtype=dtype)
            return tensor.reshape(*indices.shape, *tensor.shape[1:])
        routes = pack([step.observation[0] for step in steps])
        states = pack([step.observation[1] for step in steps])
        actions = pack([step.action for step in steps])
        old_log = pack([step.log_prob for step in steps], torch.float32)
        hidden = torch.cat([transitions[chunk.start].hidden for chunk in batch], dim=1).to(self.device)
        mask = torch.as_tensor(valid, device=self.device)
        logits, value, _ = self.model.actor_critic(routes, states, hidden)
        heads = distributions(tuple(head[mask] for head in logits))
        log_prob = sum(head.log_prob(actions[mask][:, index]) for index, head in enumerate(heads))
        entropy = sum(head.entropy() for head in heads).mean()
        advantage = torch.as_tensor(advantages[indices[valid]], device=self.device)
        target = torch.as_tensor(returns[indices[valid]], device=self.device)
        ratio = (log_prob - old_log[mask]).exp()
        clip = self.config["clip"]
        actor_loss = -torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage).mean()
        value_loss = (value[mask] - target).square().mean()
        loss = actor_loss + self.config["value_coef"] * value_loss - self.config["entropy_coef"] * entropy
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["max_grad_norm"],
                                       error_if_nonfinite=True)
        self.optimizer.step()
        return dict(actor_loss=actor_loss.item(), value_loss=value_loss.item(), entropy=entropy.item())

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + ".tmp")
        torch.save(dict(schema=CHECKPOINT_SCHEMA, observation_schema=OBSERVATION_SCHEMA,
                        model_kwargs=self.model_kwargs, model=self.model.state_dict(),
                        optimizer=self.optimizer.state_dict(), steps=self.total_steps), temp)
        temp.replace(path)


def load_execution_model(path, env, config, device):
    data = torch.load(path, map_location=device, weights_only=True)
    expected = model_kwargs(env, config)
    if (data.get("schema"), data.get("observation_schema"), data.get("model_kwargs")) != (
        CHECKPOINT_SCHEMA, OBSERVATION_SCHEMA, expected
    ):
        raise ValueError("执行层检查点与模型或观测配置不匹配")
    model = ExecutionRouteNetwork(**expected).to(device)
    model.load_state_dict(data["model"])
    return model.eval()
