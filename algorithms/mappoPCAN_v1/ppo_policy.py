import torch
from .ppo_actor import PPOActor
from .ppo_critic import PPOCritic
from .ppo_pcan import PPOPCAN


class PPOPCANPolicy:
    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):

        self.args = args
        self.device = device
        # optimizer config
        self.lr = args.lr

        self.obs_space = obs_space
        self.cent_obs_space = cent_obs_space
        self.act_space = act_space

        self.pcan = PPOPCAN(args, self.obs_space, self.act_space, self.device)
        self.actor = PPOActor(args, self.obs_space, self.act_space, self.device)
        self.critic = PPOCritic(args, self.cent_obs_space, self.device)

        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters()},
            {'params': self.critic.parameters()}
        ], lr=self.lr)

        self.pcan_optimizer = torch.optim.Adam(
            self.pcan.parameters(),
            lr=self.lr,
        )

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks):
        """
        Returns:
            values, actions, action_log_probs, rnn_states_actor, rnn_states_critic
        """
        actions, action_log_probs, rnn_states_actor = self.actor(obs, rnn_states_actor, masks)
        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks):
        """
        Returns:
            values
        """
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks, active_masks=None):
        """
        Returns:
            values, action_log_probs, dist_entropy
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs, rnn_states_actor, action, masks, active_masks)
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy

    def evaluate_pcan(self, obs, actions):
        """
        Returns:
            threat_output, attack_output, pcan_record_info
        """
        threat_output, attack_output, pcan_record_info = self.pcan(obs, actions)
        return threat_output, attack_output, pcan_record_info

    def act(self, obs, rnn_states_actor, masks, deterministic=False):
        """
        Returns:
            actions, rnn_states_actor
        """
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, deterministic)
        return actions, rnn_states_actor

    def prep_training(self):
        self.pcan.train()
        self.actor.train()
        self.critic.train()

    def prep_rollout(self):
        self.pcan.eval()
        self.actor.eval()
        self.critic.eval()

    def copy(self):
        return PPOPCANPolicy(self.args, self.obs_space, self.act_space, self.device)
