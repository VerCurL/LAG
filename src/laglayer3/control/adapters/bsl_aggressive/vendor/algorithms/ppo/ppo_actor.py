import torch
import torch.nn as nn

from ..utils.mlp import MLPBase
from ..utils.gru import GRULayer
from ..utils.act import ACTLayer
from ..utils.utils import check


class PPOActor(nn.Module):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        super(PPOActor, self).__init__()
        # network config
        self.gain = args.gain
        self.initial_log_std = getattr(args, "initial_log_std", None)
        self.initial_action_std = getattr(args, "initial_action_std", 0.5)
        self.use_squashed_action = getattr(args, "use_squashed_action", True)
        self.hidden_size = args.hidden_size
        self.act_hidden_size = args.act_hidden_size
        self.activation_id = args.activation_id
        self.use_feature_normalization = args.use_feature_normalization
        self.use_recurrent_policy = args.use_recurrent_policy
        self.recurrent_hidden_size = args.recurrent_hidden_size
        self.recurrent_hidden_layers = args.recurrent_hidden_layers
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.use_prior = args.use_prior
        # (1) feature extraction module
        self.base = MLPBase(obs_space, self.hidden_size, self.activation_id, self.use_feature_normalization)
        # (2) rnn module
        input_size = self.base.output_size
        if self.use_recurrent_policy:
            self.rnn = GRULayer(input_size, self.recurrent_hidden_size, self.recurrent_hidden_layers)
            input_size = self.rnn.output_size
        # (3) act module
        self.act = ACTLayer(
            act_space,
            input_size,
            self.act_hidden_size,
            self.activation_id,
            self.gain,
            initial_log_std=self.initial_log_std,
            initial_action_std=self.initial_action_std,
            use_squashed_action=self.use_squashed_action,
        )

        self.to(device)

    def forward(self, obs, rnn_states, masks, deterministic=False):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if self.use_prior:
            # prior knowledage for controling shoot missile
            attack_angle = torch.rad2deg(obs[:, 11]) # unit degree
            distance = obs[:, 13] * 10000 # unit m
            alpha0 = torch.full(size=(obs.shape[0],1), fill_value=3).to(**self.tpdv)
            beta0 = torch.full(size=(obs.shape[0],1), fill_value=10).to(**self.tpdv)
            alpha0[distance<=12000] = 6
            alpha0[distance<=8000] = 10
            beta0[attack_angle<=45] = 6
            beta0[attack_angle<=22.5] = 3

        actor_features = self.base(obs)

        if self.use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self.use_prior:
            actions, action_log_probs = self.act(actor_features, deterministic, alpha0=alpha0, beta0=beta0)
        else:
            actions, action_log_probs = self.act(actor_features, deterministic)

        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, active_masks=None):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if self.use_prior:
            # prior knowledage for controling shoot missile
            attack_angle = torch.rad2deg(obs[:, 11]) # unit degree
            distance = obs[:, 13] * 10000 # unit m
            alpha0 = torch.full(size=(obs.shape[0], 1), fill_value=3).to(**self.tpdv)
            beta0 = torch.full(size=(obs.shape[0], 1), fill_value=10).to(**self.tpdv)
            alpha0[distance<=12000] = 6
            alpha0[distance<=8000] = 10
            beta0[attack_angle<=45] = 6
            beta0[attack_angle<=22.5] = 3

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self.use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self.use_prior:
            action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features, action, active_masks, alpha0=alpha0, beta0=beta0)
        else:
            action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features, action, active_masks)

        return action_log_probs, dist_entropy

    def get_action_distribution(self, obs, rnn_states, masks):
        """Return the current continuous distribution at the supplied state."""
        if self.use_prior:
            raise ValueError("Teacher KL is not supported for prior-guided policies")
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        actor_features = self.base(obs)
        if self.use_recurrent_policy:
            actor_features, _ = self.rnn(actor_features, rnn_states, masks)
        return self.act.get_continuous_distribution(actor_features)

    def get_action_std(self):
        return self.act.get_action_std()
