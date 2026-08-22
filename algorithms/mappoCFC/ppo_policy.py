from pathlib import Path

import torch

from algorithms.mappo.ppo_actor import PPOActor
from algorithms.mappo.ppo_critic import PPOCritic
from .ppo_AeroTAF import PPOAeroTAF


class PPOAeroTAFPolicy:
    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.args = args
        self.device = device
        self.obs_space = obs_space
        self.cent_obs_space = cent_obs_space
        self.act_space = act_space

        self.AeroTAF = PPOAeroTAF(args, obs_space, act_space, device)
        self.actor = PPOActor(args, obs_space, act_space, device)
        self.critic = PPOCritic(args, cent_obs_space, device)

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.actor.parameters()},
                {"params": self.critic.parameters()},
            ],
            lr=args.lr,
        )
        self.AeroTAF_optimizer = torch.optim.Adam(
            self.AeroTAF.parameters(),
            lr=args.AeroTAF_lr,
            weight_decay=args.AeroTAF_weight_decay,
        )
        if args.AeroTAF_pretrained_model:
            self.load_pretrained_aerotaf(args.AeroTAF_pretrained_model)

    def load_pretrained_aerotaf(self, checkpoint_path):
        path = Path(checkpoint_path).expanduser()
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self.AeroTAF.load_state_dict(state_dict)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks):
        actions, action_log_probs, rnn_states_actor = self.actor(obs, rnn_states_actor, masks)
        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks):
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(
        self,
        cent_obs,
        obs,
        rnn_states_actor,
        rnn_states_critic,
        action,
        masks,
        active_masks=None,
    ):
        action_log_probs, dist_entropy = self.actor.evaluate_actions(
            obs, rnn_states_actor, action, masks, active_masks
        )
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy

    def evaluate_AeroTAF(
        self,
        obs,
        actions,
        seq_len,
        time_offset=0,
        valid_mask=None,
    ):
        return self.AeroTAF(
            obs,
            actions,
            seq_len=seq_len,
            time_offset=time_offset,
            valid_mask=valid_mask,
        )

    def build_AeroTAF_trajectory_cache(self, obs, actions):
        return self.AeroTAF.build_trajectory_cache(obs, actions)

    def evaluate_AeroTAF_cached(
        self,
        cache,
        env_indices,
        time_indices,
        segment_starts,
        history_windows,
        action_variants,
    ):
        return self.AeroTAF.predict_cached(
            cache,
            env_indices,
            time_indices,
            segment_starts,
            history_windows,
            action_variants,
        )

    def act(self, obs, rnn_states_actor, masks, deterministic=False):
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, deterministic)
        return actions, rnn_states_actor

    def prep_training(self):
        self.AeroTAF.train()
        self.actor.train()
        self.critic.train()

    def prep_rollout(self):
        self.AeroTAF.eval()
        self.actor.eval()
        self.critic.eval()

    def copy(self):
        return PPOAeroTAFPolicy(
            self.args,
            self.obs_space,
            self.cent_obs_space,
            self.act_space,
            self.device,
        )
