"""Isolated C46 compact PPO candidates; no flight or training is started here.

Both candidates consume the same 40D, already scaled physical observation.
Actor and critic have independent feature extractors.  Raw observations remain
in the PPO buffer; every rollout/update evaluates the encoder afresh.
"""

from copy import deepcopy
from numbers import Integral

import numpy as np
import torch
from gymnasium import spaces
from torch import nn

from ...algorithms.ppo.ppo_actor import PPOActor
from ...algorithms.ppo.ppo_critic import PPOCritic
from ...algorithms.ppo.ppo_policy import PPOPolicy
from ...algorithms.utils.act import ACTLayer
from ...algorithms.utils.gru import GRULayer
from ...algorithms.utils.mlp import MLPLayer
from ...config import get_config
from ...scripts.experiments.c46_compact_observation import (
    CONTINUOUS_INDICES, DIRECT_INDICES, FEATURE_NAMES, OBSERVATION_DIM,
    SCHEMA as OBSERVATION_SCHEMA,
)


SCHEMA = "c46-compact-policy-v1"
HIDDEN_SIZE = 64


def _identity(architecture, seed):
    if architecture not in ("A", "B"):
        raise ValueError("architecture must be exactly 'A' or 'B'")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**31:
        raise ValueError("seed must be an integer in [0, 2**31)")
    return architecture, int(seed)


def _seeded(seed, factory):
    # Modules are initialized on CPU and then moved to the requested device.
    # Preserve caller RNG state; A/B shared downstream blocks use identical seeds.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return factory()


def make_compact_args(**overrides):
    """Return ordinary repository PPO args with the fixed candidate network."""
    args = get_config().parse_args([])
    for name, value in overrides.items():
        if not hasattr(args, name):
            raise ValueError("unknown PPO argument: " + name)
        setattr(args, name, value)
    args.hidden_size = "64 64"
    args.act_hidden_size = "64"
    args.activation_id = 0  # Tanh in both candidates.
    args.use_feature_normalization = False  # Physical scaling precedes encoding.
    args.use_recurrent_policy = True
    args.recurrent_hidden_size = HIDDEN_SIZE
    args.recurrent_hidden_layers = 1
    args.use_prior = False
    args.use_squashed_action = True
    args.initial_action_std = 0.1
    args.initial_log_std = None
    return args


def _validated_args(args):
    args = make_compact_args() if args is None else deepcopy(args)
    required = dict(hidden_size="64 64", act_hidden_size="64", activation_id=0,
                    use_feature_normalization=False, use_recurrent_policy=True,
                    recurrent_hidden_size=64, recurrent_hidden_layers=1,
                    use_prior=False, use_squashed_action=True,
                    initial_action_std=0.1, initial_log_std=None)
    for name, value in required.items():
        if getattr(args, name, object()) != value:
            raise ValueError("C46 requires {}={!r}".format(name, value))
    return args


class CompactEncoder(nn.Module):
    """A small MLP, or a bottleneck preserving 14 selected raw channels."""

    def __init__(self, architecture):
        super().__init__()
        _identity(architecture, 0)
        self.architecture = architecture
        if architecture == "A":
            self.mlp = MLPLayer(OBSERVATION_DIM, "64 64", 0)
        else:
            self.register_buffer("continuous_indices", torch.tensor(CONTINUOUS_INDICES), persistent=False)
            self.register_buffer("direct_indices", torch.tensor(DIRECT_INDICES), persistent=False)
            self.bottleneck = MLPLayer(len(CONTINUOUS_INDICES), "32 16", 0)
            self.align = MLPLayer(16 + len(DIRECT_INDICES), "64", 0)

    def _check(self, obs):
        if not isinstance(obs, torch.Tensor) or obs.ndim != 2 or obs.shape[-1] != OBSERVATION_DIM:
            raise ValueError("C46 encoder requires a [batch,40] tensor")
        if not torch.isfinite(obs).all():
            raise ValueError("C46 observation must be finite")

    def joined_features(self, obs):
        """Expose the B bottleneck plus unchanged bypass for independent tests."""
        self._check(obs)
        if self.architecture != "B":
            raise ValueError("joined_features is only defined for B")
        encoded = self.bottleneck(obs.index_select(1, self.continuous_indices))
        direct = obs.index_select(1, self.direct_indices)
        return torch.cat((encoded, direct), dim=-1)

    def forward(self, obs):
        if self.architecture == "A":
            self._check(obs)
            return self.mlp(obs)
        return self.align(self.joined_features(obs))

    @property
    def output_size(self):
        return HIDDEN_SIZE


def _set_network_attributes(module, args, device):
    module.hidden_size = args.hidden_size
    module.act_hidden_size = args.act_hidden_size
    module.activation_id = args.activation_id
    module.use_feature_normalization = args.use_feature_normalization
    module.use_recurrent_policy = True
    module.recurrent_hidden_size = HIDDEN_SIZE
    module.recurrent_hidden_layers = 1
    module.tpdv = dict(dtype=torch.float32, device=torch.device(device))


class CompactActor(PPOActor):
    def __init__(self, args, act_space, architecture, seed, device):
        nn.Module.__init__(self)
        _set_network_attributes(self, args, device)
        self.gain = args.gain
        self.initial_log_std = None
        self.initial_action_std = 0.1
        self.use_squashed_action = True
        self.use_prior = False
        self.base = _seeded(seed + 11 + (1000 if architecture == "B" else 0),
                            lambda: CompactEncoder(architecture))
        self.rnn = _seeded(seed + 101, lambda: GRULayer(64, 64, 1))
        self.act = _seeded(seed + 201, lambda: ACTLayer(
            act_space, 64, "64", 0, args.gain,
            initial_action_std=0.1, use_squashed_action=True))
        self.to(device)


class CompactCritic(PPOCritic):
    def __init__(self, args, architecture, seed, device):
        nn.Module.__init__(self)
        _set_network_attributes(self, args, device)
        self.base = _seeded(seed + 21 + (1000 if architecture == "B" else 0),
                            lambda: CompactEncoder(architecture))
        self.rnn = _seeded(seed + 301, lambda: GRULayer(64, 64, 1))
        self.mlp, self.value_out = _seeded(seed + 401, lambda: (
            MLPLayer(64, "64", 0), nn.Linear(64, 1)))
        self.to(device)


class CompactPolicy(PPOPolicy):
    def __init__(self, architecture, seed, device="cpu", args=None):
        self.architecture, self.seed = _identity(architecture, seed)
        self.args = _validated_args(args)
        self.device = torch.device(device)
        limit = np.finfo(np.float32).max
        self.obs_space = spaces.Box(-limit, limit, shape=(OBSERVATION_DIM,), dtype=np.float32)
        self.act_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.lr = self.args.lr
        self.actor_lr = self.lr if getattr(self.args, "actor_lr", None) is None else self.args.actor_lr
        self.critic_lr = self.lr if getattr(self.args, "critic_lr", None) is None else self.args.critic_lr
        self.actor = CompactActor(self.args, self.act_space, self.architecture, self.seed, self.device)
        self.critic = CompactCritic(self.args, self.architecture, self.seed, self.device)
        self.optimizer = torch.optim.Adam([
            dict(params=self.actor.parameters(), lr=self.actor_lr),
            dict(params=self.critic.parameters(), lr=self.critic_lr),
        ], lr=self.lr)

    def metadata(self):
        """Network identity only; formal training config must be registered separately."""
        actor_count = sum(p.numel() for p in self.actor.parameters())
        critic_count = sum(p.numel() for p in self.critic.parameters())
        return dict(schema=SCHEMA, architecture=self.architecture, initialization_seed=self.seed,
                    observation_schema=OBSERVATION_SCHEMA,
                    observation_features=list(FEATURE_NAMES), observation_dim=OBSERVATION_DIM,
                    continuous_indices=list(CONTINUOUS_INDICES),
                    direct_indices=list(DIRECT_INDICES), recurrent_hidden_size=64,
                    recurrent_hidden_layers=1, head_width=64, activation="tanh",
                    initial_action_std=0.1, action_gain=float(self.args.gain),
                    squashed_action=True, action_dim=4,
                    actor_parameters=actor_count, critic_parameters=critic_count,
                    total_parameters=actor_count + critic_count)

    def checkpoint_payload(self):
        """Evaluation checkpoint, not an optimizer/RNG training-resume package."""
        return dict(metadata=self.metadata(), actor=deepcopy(self.actor.state_dict()),
                    critic=deepcopy(self.critic.state_dict()))

    def load_checkpoint_payload(self, payload):
        if not isinstance(payload, dict) or set(payload) != {"metadata", "actor", "critic"}:
            raise ValueError("C46 checkpoint requires metadata, actor, and critic")
        if payload["metadata"] != self.metadata():
            raise ValueError("C46 checkpoint identity mismatch")
        # Validate both modules before changing either one.
        for name in ("actor", "critic"):
            state = payload[name]
            expected = getattr(self, name).state_dict()
            if not isinstance(state, dict) or set(state) != set(expected):
                raise ValueError("C46 checkpoint state keys mismatch: " + name)
            for key, wanted in expected.items():
                got = state[key]
                if (not isinstance(got, torch.Tensor) or got.shape != wanted.shape
                        or got.dtype != wanted.dtype or not torch.isfinite(got).all()):
                    raise ValueError("C46 checkpoint invalid tensor: " + name + "." + key)
        self.actor.load_state_dict(payload["actor"], strict=True)
        self.critic.load_state_dict(payload["critic"], strict=True)

    def copy(self):
        # Match the base policy's fresh-instance semantics, preserving C46 type.
        return CompactPolicy(self.architecture, self.seed, self.device, self.args)


def create_compact_policy(architecture, seed, device="cpu", args=None):
    return CompactPolicy(architecture, seed, device=device, args=args)
