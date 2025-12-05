import numpy as np
import torch
from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, MultipleCombatEnv
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from envs.JSBSim.core.catalog import Catalog as c
from algorithms.ppo.ppo_actor import PPOActor
from algorithms.ppoMoE.ppo_actor import PPOMoEActor
import time
import logging

logging.basicConfig(level=logging.DEBUG)


class Args_PPO:
    def __init__(self) -> None:
        self.gain = 0.01
        self.hidden_size = '128 128'
        self.act_hidden_size = '128 128'
        self.activation_id = 1
        self.use_feature_normalization = False
        self.use_recurrent_policy = True
        self.recurrent_hidden_size = 128
        self.recurrent_hidden_layers = 1
        self.tpdv = dict(dtype=torch.float32, device=torch.device('cpu'))
        self.use_prior = True

class Args_PPOMoE:
    def __init__(self) -> None:
        self.gain = 0.01
        self.hidden_size = '256 256'
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

def _t2n(x):
    return x.detach().cpu().numpy()


num_agents = 2
render = True
ego_policy_index = 505
enm_policy_index = 579
episode_rewards = 0
ego_run_dir = "D:/FastProjects/ModelFlight/LAGMoE/scripts/results/SingleCombat/1v1/ShootMissile/HierarchySelfplay/ppoMoE/v1/run-test"
enm_run_dir = "D:/FastProjects/ModelFlight/LAGMoE/scripts/results/SingleCombat/1v1/ShootMissile/HierarchySelfplay/ppo/v1/run-test"

env = SingleCombatEnv(config_name="1v1/ShootMissile/HierarchySelfplay", policy_type="default")
env.seed(0)
args_ppo = Args_PPO()
args_ppoMoE = Args_PPOMoE()

ego_policy = PPOMoEActor(args_ppoMoE, env.observation_space, env.action_space, device=torch.device("cuda"))
enm_policy = PPOActor(args_ppo, env.observation_space, env.action_space, device=torch.device("cuda"))
ego_policy.eval()
enm_policy.eval()
ego_policy.load_state_dict(torch.load(ego_run_dir + f"/actor_{ego_policy_index}.pt"))
enm_policy.load_state_dict(torch.load(enm_run_dir + f"/actor_{enm_policy_index}.pt"))

# 开始测试
num_render = 1
experiment_file_name = "../gaming_result/PPOMoE.vs.PPO/1v1" + "/[" + str(ego_policy_index) + "," + str(enm_policy_index) + "]/"
print("Start render")
for i in range(num_render):
    print(f"=======The {i + 1} game begins!=======")

    # 设置飞行文件记录
    obs = env.reset()
    experiment_name = experiment_file_name + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
    if render:
        env.render(mode='txt', filepath=f'{experiment_name}.txt.acmi')

    # 初始化敌我机的状态空间
    ego_rnn_states = np.zeros((1, 1, 128), dtype=np.float32)
    masks = np.ones((num_agents // 2, 1))
    enm_obs = obs[num_agents // 2:, :]
    ego_obs = obs[:num_agents // 2, :]
    enm_rnn_states = np.zeros_like(ego_rnn_states, dtype=np.float32)

    # 开始对战
    while True:
        # 敌我方获取动作空间
        ego_actions, _, ego_rnn_states = ego_policy(ego_obs, ego_rnn_states, masks, deterministic=True)
        ego_actions = _t2n(ego_actions)
        ego_rnn_states = _t2n(ego_rnn_states)
        enm_actions, _, enm_rnn_states = enm_policy(enm_obs, enm_rnn_states, masks, deterministic=True)
        enm_actions = _t2n(enm_actions)
        enm_rnn_states = _t2n(enm_rnn_states)
        actions = np.concatenate((ego_actions, enm_actions), axis=0)

        # 单步更新并计算奖励等单步指标
        obs, rewards, dones, infos = env.step(actions)
        rewards = rewards[:num_agents // 2, ...]
        episode_rewards += rewards

        # 记录单步轨迹
        if render:
            env.render(mode='txt', filepath=f'{experiment_name}.txt.acmi')

        # 如果该回合达到了终止条件则终止
        if dones.all():
            print(infos)
            break

        bloods = [env.agents[agent_id].bloods for agent_id in env.agents.keys()]
        print(f"step:{env.current_step}, bloods:{bloods}")
        enm_obs = obs[num_agents // 2:, ...]
        ego_obs = obs[:num_agents // 2, ...]

    print(episode_rewards)
    print(f"=======The {i + 1} game ends!=======")