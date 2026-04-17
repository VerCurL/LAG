import torch
from .ppo_actor import PPOActor
from .ppo_critic import PPOCritic


class PPOPCANPolicy:
    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):

        self.args = args
        self.device = device
        # optimizer config
        self.lr = args.lr

        self.obs_space = obs_space
        self.cent_obs_space = cent_obs_space
        self.act_space = act_space

        self.actor = PPOActor(args, self.obs_space, self.act_space, self.device)
        self.critic = PPOCritic(args, self.cent_obs_space, self.device)

        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters()},
            {'params': self.critic.parameters()}
        ], lr=self.lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks):
        """
        Returns:
            values, actions, action_log_probs, rnn_states_actor, rnn_states_critic
        """
        actions, action_log_probs, rnn_states_actor, actor_features, credit = self.actor(obs, rnn_states_actor, masks)
        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic, actor_features, credit

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

    def act(self, obs, rnn_states_actor, masks, deterministic=False):
        """
        Returns:
            actions, rnn_states_actor
        """
        actions, _, rnn_states_actor, _, credit = self.actor(obs, rnn_states_actor, masks, deterministic)
        return actions, rnn_states_actor, credit

    def pcan(self, obs, rnn_state_actor, masks):
        """
        Returns:
            rewards_pred, credit, pcan_record_info
        """
        target_pred, credit, pcan_record_info = self.actor.evaluate_pcan(obs, rnn_state_actor, masks)
        return target_pred, credit, pcan_record_info

    def prep_training(self):
        self.actor.train()
        self.critic.train()

    def prep_rollout(self):
        self.actor.eval()
        self.critic.eval()

    def copy(self):
        return PPOPCANPolicy(self.args, self.obs_space, self.act_space, self.device)
