import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from config import get_config
from algorithms.mappoMoE.ppo_actor import PPOMoEActor
from envs.JSBSim.envs import MultipleCombatEnv


def _t2n(x):
    return x.detach().cpu().numpy()
# ---------- 1. 构造 actor 并加载模型 ----------
device = torch.device("cuda")
class Args_MAPPOMoE:
    def __init__(self) -> None:
        self.gain = 0.01
        self.hidden_size = '128 128'
        self.expert_hidden_size = '32 32'
        self.num_general_experts = 2
        self.num_special_experts = 6
        self.top_k = 2
        self.act_hidden_size = '128 128'
        self.activation_id = 1
        self.use_feature_normalization = False
        self.use_recurrent_policy = True
        self.recurrent_hidden_size = 128
        self.recurrent_hidden_layers = 1
        self.tpdv = dict(dtype=torch.float32, device=torch.device('cpu'))
        self.use_prior = True

num_agents = 8
ego_policy_index = 222
enm_policy_index = 222
ego_run_dir = "D:/FastProjects/ModelFlight/LAGMoE/scripts/results/MultipleCombat/4v4/ShootMissile/HierarchySelfplay/mappoMoE/128-128-32{2-6-2}/run-test"
enm_run_dir = "D:/FastProjects/ModelFlight/LAGMoE/scripts/results/MultipleCombat/4v4/ShootMissile/HierarchySelfplay/mappoMoE/128-128-32{2-6-2}/run-test"
env = MultipleCombatEnv(config_name="4v4/ShootMissile/HierarchySelfplay", policy_type="fkr", fix_position=True)

args = Args_MAPPOMoE()
ego_actor = PPOMoEActor(args, env.observation_space, env.action_space, device=device)
enm_actor = PPOMoEActor(args, env.observation_space, env.action_space, device=device)
ego_actor.eval()
enm_actor.eval()
ego_actor.load_state_dict(torch.load(ego_run_dir + f"/actor_{ego_policy_index}.pt"))
enm_actor.load_state_dict(torch.load(enm_run_dir + f"/actor_{enm_policy_index}.pt"))

# ---------- 2. 初始化敌我机的状态空间 ----------
obs, _ = env.reset()
ego_rnn_states = np.zeros((1, 1, 128), dtype=np.float32)
masks = np.ones((num_agents // 2, 1))
enm_obs = obs[num_agents // 2:, :]
ego_obs = obs[:num_agents // 2, :]
enm_rnn_states = np.zeros_like(ego_rnn_states, dtype=np.float32)

# ---------- 3. 开始对战 ----------
all_experts_out = []
while True:
    # 敌我方获取动作空间
    ego_actions, _, ego_rnn_states, experts_out = ego_actor(ego_obs, ego_rnn_states, masks, deterministic=True)
    ego_actions = _t2n(ego_actions)
    ego_rnn_states = _t2n(ego_rnn_states)
    enm_actions, _, enm_rnn_states, _ = enm_actor(enm_obs, enm_rnn_states, masks, deterministic=True)
    enm_actions = _t2n(enm_actions)
    enm_rnn_states = _t2n(enm_rnn_states)
    actions = np.concatenate((ego_actions, enm_actions), axis=0)

    # 单步更新并计算奖励等单步指标
    obs, _, _, dones, infos = env.step(actions)

    # 如果该回合达到了终止条件则终止
    if dones.all():
        print(infos)
        break
    print(f"step:{env.current_step}")

    # 更新敌我方观测值
    enm_obs = obs[num_agents // 2:, ...]
    ego_obs = obs[:num_agents // 2, ...]

    # 记录所有的experts_out
    all_experts_out.append(experts_out.detach().cpu())

# experts_out: [batch, total_experts, dim]
all_experts_out = torch.cat(all_experts_out, dim=0)
experts_repr = all_experts_out.mean(dim=0)
experts_norm = F.normalize(experts_repr, dim=1)
cos_sim_matrix = experts_norm @ experts_norm.T
plt.imshow(cos_sim_matrix.cpu(), vmin=-1, vmax=1)
plt.colorbar()
plt.title("Cosine Similarity Between Experts")
plt.xlabel("Expert Index")
plt.ylabel("Expert Index")
plt.tight_layout()
plt.show()
