import numpy as np
import torch
from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, MultipleCombatEnv
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from envs.JSBSim.core.catalog import Catalog as c
from algorithms.ppo.ppo_actor import PPOActor
import time
import logging
logging.basicConfig(level=logging.DEBUG)

class Args:
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
    
def _t2n(x):
    return x.detach().cpu().numpy()

# 配置要测试的模型
num_agents = 4
render = True
ego_policy_index = 1040
enm_policy_index = 311
episode_rewards = 0
ego_run_dir = "../scripts/results/MultipleCombat/2v2/ShootMissile/HierarchySelfplay/mappo/v1/run-v2/files"
enm_run_dir = "../scripts/results/MultipleCombat/2v2/ShootMissile/HierarchySelfplay/mappo/v1/run-hyx/files"

fix_position = False
env = MultipleCombatEnv("2v2/ShootMissile/HierarchySelfplay", fix_position=fix_position)
env.seed(0)
args = Args()

# 加载模型
ego_policy = PPOActor(args, env.observation_space, env.action_space, device=torch.device("cuda"))
enm_policy = PPOActor(args, env.observation_space, env.action_space, device=torch.device("cuda"))
ego_policy.eval()
enm_policy.eval()
ego_policy.load_state_dict(torch.load(ego_run_dir + f"/actor_{ego_policy_index}.pt"))
enm_policy.load_state_dict(torch.load(enm_run_dir + f"/actor_{enm_policy_index}.pt"))

# 开始测试
num_render = 1000
ego_version = ego_run_dir.split('/')[-2].split('-')[-1]
enm_version = enm_run_dir.split('/')[-2].split('-')[-1]
experiment_file_name = "gaming_result/" + ego_version + ".vs." + enm_version + "/[" + str(ego_policy_index) + "," + str(enm_policy_index) + "]/"
print("Start render")
for i in range(num_render):
    if not fix_position:
        print(f"=======The {i + 1} game begins!=======")
    obs, _ = env.reset()
    experiment_name = experiment_file_name + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time())) + ("_fix" if fix_position else "")
    if render:
        env.render(mode='txt', filepath=f'{experiment_name}.txt.acmi')
    ego_rnn_states = np.zeros((1, 1, 128), dtype=np.float32)
    masks = np.ones((num_agents // 2, 1))
    enm_obs =  obs[num_agents // 2:, :]
    ego_obs =  obs[:num_agents // 2, :]
    enm_rnn_states = np.zeros_like(ego_rnn_states, dtype=np.float32)
    while True:
        start = time.time()
        ego_actions, _, ego_rnn_states = ego_policy(ego_obs, ego_rnn_states, masks, deterministic=True)
        end = time.time()
        # print(f"NN forward time: {end-start}")
        ego_actions = _t2n(ego_actions)
        ego_rnn_states = _t2n(ego_rnn_states)
        enm_actions, _, enm_rnn_states = enm_policy(enm_obs, enm_rnn_states, masks, deterministic=True)
        enm_actions = _t2n(enm_actions)
        enm_rnn_states = _t2n(enm_rnn_states)
        actions = np.concatenate((ego_actions, enm_actions), axis=0)
        # Obser reward and next obs
        start = time.time()
        obs, _, rewards, dones, infos = env.step(actions)
        end = time.time()
        # print(f"Env step time: {end-start}")
        rewards = rewards[:num_agents // 2, ...]
        episode_rewards += rewards
        if render:
            env.render(mode='txt', filepath=f'{experiment_name}.txt.acmi')
        if dones.all():
            print(infos)
            break
        bloods = [env.agents[agent_id].bloods for agent_id in env.agents.keys()]
        print(f"step:{env.current_step}, bloods:{bloods}")
        enm_obs =  obs[num_agents // 2:, ...]
        ego_obs =  obs[:num_agents // 2, ...]

    print(episode_rewards)
    if fix_position:
        break
    env._create_records = False
    print(f"=======The {i + 1} game ends!=======")