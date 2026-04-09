import torch
import torch.nn as nn

from ..utils.mlp import MLPBase
from ..utils.gru import GRULayer
from ..utils.act import ACTLayer
from ..utils.pcan_v2 import PCANBase
from ..utils.utils import check


class PPOActor(nn.Module):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        super(PPOActor, self).__init__()
        # network config
        self.gain = args.gain
        self.num_agents = args.num_agents
        self.hidden_size = args.hidden_size_actor
        self.act_hidden_size = args.act_hidden_size_actor
        self.activation_id = args.activation_id
        self.use_feature_normalization = args.use_feature_normalization
        self.use_recurrent_policy = args.use_recurrent_policy
        self.recurrent_hidden_size = args.recurrent_hidden_size_actor
        self.recurrent_hidden_layers = args.recurrent_hidden_layers
        self.num_heads = args.num_heads
        self.KQ_hidden_size = args.KQ_hidden_size
        self.V_hidden_size = args.V_hidden_size
        self.PCANOut_hidden_size = args.PCANOut_hidden_size
        self.tpdv = dict(dtype=torch.float32, device=device)
        # (1) feature extraction module
        self.base = MLPBase(obs_space, self.hidden_size, self.activation_id, self.use_feature_normalization)
        # (2) rnn module
        input_size = self.base.output_size
        if self.use_recurrent_policy:
            self.rnn = GRULayer(input_size, self.recurrent_hidden_size, self.recurrent_hidden_layers)
            input_size = self.rnn.output_size
        # (3) act module
        self.act = ACTLayer(act_space, input_size, self.act_hidden_size, self.activation_id, self.gain)
        # (4) pcan module
        self.pcan = PCANBase(obs_space, self.num_agents, self.num_heads, input_size,
                             self.KQ_hidden_size, self.V_hidden_size, self.PCANOut_hidden_size,
                             self.activation_id, self.use_feature_normalization)

        self.to(device)

    def forward(self, obs, rnn_states, masks, deterministic=False):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self.use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actions, action_log_probs = self.act(actor_features, deterministic)

        _, credit = self.pcan(actor_features, obs)

        return actions, action_log_probs, rnn_states, actor_features, credit

    def evaluate_actions(self, obs, rnn_states, action, masks, active_masks=None):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self.use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features, action, active_masks)

        return action_log_probs, dist_entropy

    def evaluate_actor_features(self, obs, rnn_states, masks):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        actor_features = self.base(obs)
        if self.use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        return actor_features

    def evaluate_pcan(self, obs, rnn_states, masks):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        actor_features = self.base(obs)
        if self.use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actor_features.detach()
        rewards_pred, credit = self.pcan(actor_features, obs)
        return rewards_pred, credit, self.pcan.record_info
