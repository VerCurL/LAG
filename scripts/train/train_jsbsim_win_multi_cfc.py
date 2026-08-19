#!/usr/bin/env python
import sys
import os
import traceback
import wandb
import socket
import torch
import random
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
import setproctitle
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
from config import get_config
from utils.logger import AlgorithmsLogger
from runner.share_jsbsim_runner import ShareJSBSimRunner
from envs.JSBSim.envs import MultipleCombatEnv
from envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv
from runner.tacview import Tacview

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = MultipleCombatEnv(all_args.scenario_name, all_args.policy_type, all_args.algorithm_name)
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = MultipleCombatEnv(all_args.scenario_name, all_args.policy_type, all_args.algorithm_name)
            env.seed(all_args.seed * 50000 + rank * 1000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    group = parser.add_argument_group("JSBSim Env parameters")
    group.add_argument('--scenario-name', type=str, default='singlecombat_simple',
                       help="Which scenario to run on")
    group.add_argument('--render-mode', type=str, default='txt',
                       help="txt or real_time")
    all_args = parser.parse_known_args(args)[0]
    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # seed
    np.random.seed(all_args.seed)
    random.seed(all_args.seed)
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        logging.info("choose to use gpu...")
        device = torch.device("cuda:0")  # use cude mask to control using which GPU
        torch.set_num_threads(all_args.n_training_threads)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    else:
        logging.info("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/results") \
        / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # model dir
    if all_args.model_dir is not None:
        all_args.model_dir = str(run_dir) + all_args.model_dir

    # wandb
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         notes=socket.gethostname(),
                         name=f"{all_args.experiment_name}_seed{all_args.seed}",
                         group=all_args.scenario_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        curr_run = f"run-{timestamp}"
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + str(all_args.env_name)
                              + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # logger init
    logger = AlgorithmsLogger(str(run_dir / "logs"), filename="training_log.csv", algorithm_name=all_args.algorithm_name)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None

    render_mode = all_args.render_mode

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "logger": logger,
        "device": device,
        "run_dir": run_dir,
        "render_mode": render_mode
    }

    # run experiments
    runner = ShareJSBSimRunner(config)

    try:
        runner.run()
    except BaseException:
        traceback.print_exc()
    finally:
        # post process
        envs.close()

        if all_args.use_wandb:
            run.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # 2v2参数配置
    main([
        '--env-name', 'MultipleCombat',         # 仿真环境名称
        '--scenario-name', '4v4/ShootMissile/HierarchySelfplay',    # 训练结构记录名称
        '--algorithm-name', 'mappoCFC',         # 算法名称
        '--experiment-name', 'A128-C512-H4',          # 网络参数说明
        '--seed', '1',                          # 随机种子
        '--policy-type', 'fkr',                 # 训练策略名称
        '--n-training-threads', '1',
        '--n-rollout-threads', '32',            # 每轮并行收集32局用于事件划分和AeroTAF训练
        '--cuda',                               # 是否使用cuda
        '--log-interval', '1',
        '--save-interval', '1',
        # '--model-dir', '/stage-1-Approach',     # 起始训练模型的路径
        '--use-selfplay',                       # 训练策略,这里默认自博弈
        '--selfplay-algorithm', 'fsp',          # 自博弈采用的算法
        '--n-choose-opponents', '1',
        '--use-eval',
        '--n-eval-rollout-threads', '1',
        '--eval-interval', '1',
        '--eval-episodes', '1',
        '--num-mini-batch', '5',
        '--buffer-size', '3000',
        '--num-env-steps', '1e8',
        '--lr', '3e-4',
        '--gamma', '0.99',
        '--ppo-epoch', '4',
        '--clip-params', '0.2',
        '--max-grad-norm', '2',
        '--entropy-coef', '1e-3',
        '--hidden-size-actor', '128 128',
        '--hidden-size-critic', '512 512',

        '--KQ-hidden-size', '128 128',
        '--V-hidden-size', '128 128',
        '--AeroTAF-spatial-head-num', '4',
        '--AeroTAF-time-head-num', '4',
        '--AeroTAF-attn-output-hidden-size', '64 32',
        '--AeroTAF-field-output-hidden-size', '64 32',
        '--AeroTAF-history-windows', '100',
        '--AeroTAF-kstep', '100',
        '--AeroTAF-field-gamma', '0.95',
        '--AeroTAF-epoch', '1',
        '--AeroTAF-mini-batch-size', '256',
        '--AeroTAF-inference-batch-size', '512',
        '--AeroTAF-stable-sample-ratio', '0.05',
        '--AeroTAF-lr', '3e-5',
        '--AeroTAF-weight-decay', '1e-4',
        '--CFC-counterfactual-actions', 'previous',
        '--CFC-softmax-tau', '0.2',
        '--CFC-reward-blend', '1.0',
        '--use-feature-normalization',

        '--act-hidden-size-actor', '128 128',
        '--act-hidden-size-critic', '512 512',
        '--recurrent-hidden-size-actor', '128',
        '--recurrent-hidden-size-critic', '512',
        '--recurrent-hidden-layers', '1',
        '--data-chunk-length', '8',
        '--user-name', 'fkr',                       # 名称
        '--enable-flight-recorder',                 # 是否记录飞机数据
        '--flight-recorder-agent-id', '',           # 记录的飞机id
        '--flight-recorder-plot',
        '--flight-recorder-plot-agent-ids', 'A0100,A0200'       # 指定要绘制图像的飞机id
    ])
