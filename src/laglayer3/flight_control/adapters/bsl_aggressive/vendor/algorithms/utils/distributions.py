import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import init

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

# Standardize distribution interfaces


# Categorical
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        # Single: [1] => [] => [] => [1, 1] => [1] => [1]
        # Batch: [N]/[N, 1] => [N] => [N] => [N, 1] => [N] => [N, 1]
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.squeeze(-1).unsqueeze(-1).size())
            .sum(-1, keepdim=True)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)

    def entropy(self):
        return super().entropy().unsqueeze(-1)


# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1, keepdim=True)

    def mode(self):
        return self.mean


class FixedSquashedNormal:
    """Diagonal Normal followed by tanh and an optional affine box transform.

    PPO stores the bounded action. ``log_probs`` therefore inverts the
    transform and applies the exact change-of-variables correction so rollout
    and update-time log probabilities describe the same random variable.
    """

    def __init__(self, action_mean, action_std, action_low, action_high, eps=1e-6):
        self.base_dist = torch.distributions.Normal(action_mean, action_std)
        self.action_scale = (action_high - action_low) * 0.5
        self.action_bias = (action_high + action_low) * 0.5
        self.eps = float(eps)

    def _squash(self, raw_actions):
        return self.action_bias + self.action_scale * torch.tanh(raw_actions)

    def _unsquash(self, actions):
        normalized = (actions - self.action_bias) / self.action_scale
        normalized = normalized.clamp(-1.0 + self.eps, 1.0 - self.eps)
        raw_actions = 0.5 * (torch.log1p(normalized) - torch.log1p(-normalized))
        return raw_actions, normalized

    def _log_abs_det_jacobian(self, raw_actions):
        # log(1 - tanh(z)^2), written in a numerically stable form, plus
        # the affine scale from [-1, 1] to the Box action bounds.
        tanh_log_det = 2.0 * (math.log(2.0) - raw_actions - F.softplus(-2.0 * raw_actions))
        return torch.log(self.action_scale) + tanh_log_det

    def sample(self):
        return self._squash(self.base_dist.sample())

    def rsample(self):
        return self._squash(self.base_dist.rsample())

    def log_probs(self, actions):
        raw_actions, _ = self._unsquash(actions)
        log_prob = self.base_dist.log_prob(raw_actions) - self._log_abs_det_jacobian(raw_actions)
        return log_prob.sum(-1, keepdim=True)

    def entropy(self):
        # A tanh-transformed Normal has no closed-form entropy. Use one
        # reparameterized Monte Carlo sample so the entropy term remains
        # differentiable with respect to both mean and log standard deviation.
        raw_actions = self.base_dist.rsample()
        log_prob = self.base_dist.log_prob(raw_actions) - self._log_abs_det_jacobian(raw_actions)
        return -log_prob.sum(-1, keepdim=True)

    def mode(self):
        return self._squash(self.base_dist.mean)


# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):
    def log_probs(self, actions):
        # Single: [K] => [K] => [1]
        # Batch: [N, K] => [N, K] => [N, 1]
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1, keepdim=True)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, gain=0.01):
        super(Categorical, self).__init__()

        def init_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain)

        self.logits_net = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.logits_net(x)
        return FixedCategorical(logits=x)

    @property
    def output_size(self) -> int:
        return 1


class DiagGaussian(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_outputs,
        gain=0.01,
        initial_log_std=0.0,
        squash_actions=True,
        action_low=None,
        action_high=None,
    ):
        super(DiagGaussian, self).__init__()

        def init_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain)

        self.mu_net = init_(nn.Linear(num_inputs, num_outputs))
        self.log_std = nn.Parameter(torch.full((num_outputs,), float(initial_log_std)))
        self._num_outputs = num_outputs
        self.squash_actions = bool(squash_actions)
        if action_low is None:
            action_low = torch.full((num_outputs,), -1.0, dtype=torch.float32)
        else:
            action_low = torch.as_tensor(action_low, dtype=torch.float32).reshape(num_outputs)
        if action_high is None:
            action_high = torch.full((num_outputs,), 1.0, dtype=torch.float32)
        else:
            action_high = torch.as_tensor(action_high, dtype=torch.float32).reshape(num_outputs)
        if not torch.all(torch.isfinite(action_low)) or not torch.all(torch.isfinite(action_high)):
            raise ValueError("Squashed Gaussian requires finite Box action bounds")
        if not torch.all(action_high > action_low):
            raise ValueError("Each Box action high bound must be greater than its low bound")
        # Non-persistent buffers move with the module without changing existing
        # actor checkpoint keys.
        self.register_buffer("action_low", action_low, persistent=False)
        self.register_buffer("action_high", action_high, persistent=False)

    def forward(self, x):
        action_mean = self.mu_net(x)
        action_std = self.log_std.exp().expand_as(action_mean)
        if self.squash_actions:
            return FixedSquashedNormal(
                action_mean,
                action_std,
                self.action_low,
                self.action_high,
            )
        return FixedNormal(action_mean, action_std)

    def get_action_std(self):
        """Return the pre-tanh Normal standard deviation for diagnostics."""
        return self.log_std.exp()

    @property
    def output_size(self) -> int:
        return self._num_outputs

class BetaShootBernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs, gain=0.01):
        super(BetaShootBernoulli, self).__init__()

        def init_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain)

        self.net = init_(nn.Linear(num_inputs, num_outputs))
        self._num_outputs = num_outputs
        self.constraint = nn.Softplus()

    def forward(self, x, **kwargs):
        x = self.net(x)
        x = self.constraint(x) # contrain alpha, beta >=0
        x = 100 - self.constraint(100-x) # constrain alpha, beta <=100
        alpha = 1 + x[:, 0].unsqueeze(-1)
        beta = 1 + x[:, 1].unsqueeze(-1)
        alpha_0 = kwargs['alpha0']
        beta_0 = kwargs['beta0']
        # print(f"{alpha}, {beta}, {alpha_0}, {beta_0}")
        p = (alpha + alpha_0) / (alpha + alpha_0 + beta + beta_0)
        return FixedBernoulli(p)

    @property
    def output_size(self) -> int:
        return self._num_outputs

class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs, gain=0.01):
        super(Bernoulli, self).__init__()

        def init_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain)

        self.logits_net = init_(nn.Linear(num_inputs, num_outputs))
        self._num_outputs = num_outputs

    def forward(self, x):
        x = self.logits_net(x)
        return FixedBernoulli(logits=x)

    @property
    def output_size(self) -> int:
        return self._num_outputs
