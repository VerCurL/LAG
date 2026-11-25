# LAGLayer3

LAGLayer3 是一个面向 4v4 空战的分层控制框架。当前版本包含领域对象、相对航迹表示、Provider 接口、首脑和执行层网络模板、本地 JSBSIM 适配器，以及迁移后的 bsl_aggressive 飞控。

## 从零创建环境

项目使用 Conda 管理 Python 环境。下面命令会创建名为 LagLayer3 的环境：

~~~powershell
conda create -n LagLayer3 python=3.8 -y
conda activate LagLayer3
~~~

如果本机已经存在 LagLayer3 环境，可以直接执行：

~~~powershell
conda activate LagLayer3
~~~

## 安装依赖

先确认当前 Python 来自 LagLayer3 环境：

~~~powershell
python --version
where.exe python
~~~

安装项目依赖和项目本身：

~~~powershell
python -m pip install --no-build-isolation --no-deps -e .
~~~

当前脚本入口会自动加入项目的 src 目录，因此可以直接从项目根目录运行。项目包名仍然是 laglayer3。需要在其他项目中导入时，再使用当前 LagLayer3 环境安装 editable 包。

检查安装结果：

~~~powershell
python -c "import laglayer3; print(laglayer3.__version__); print(laglayer3.__file__)"
python -c "import jsbsim, torch, gymnasium, yaml, numpy; print('JSBSim, PyTorch, Gymnasium, PyYAML and NumPy are ready')"
~~~

如果从自定义工作目录启动 Python 模块，可以设置源码路径：

~~~powershell
$env:PYTHONPATH = "src"
~~~

## JSBSim 数据

项目已经包含本地 JSBSim 数据目录：

~~~text
third_party/jsbsim_data/
~~~

默认配置使用这个目录。配置文件是 [configs/default.yaml](configs/default.yaml)。其中默认使用本地 JSBSim、F16 模型、60 Hz 仿真、5 Hz 执行层和 1 Hz 首脑层。

如果从其他工作目录启动脚本，请把 simulation.data_root 改成 JSBSim 数据目录的绝对路径，或者从项目根目录运行命令。

## 校验配置

在项目根目录执行：

~~~powershell
python scripts/validate_config.py configs/default.yaml
~~~

预期输出：

~~~text
valid configuration: configs/default.yaml
~~~

## 运行本地 smoke

使用确定性运动学后端进行快速接口检查：

~~~powershell
python scripts/smoke_local.py --steps 120 --backend kinematic
~~~

使用真实 JSBSim F16 后端运行短回合：

~~~powershell
python scripts/smoke_local.py --steps 120 --backend jsbsim
~~~

预期结果包含：

~~~text
timestamp=2.000 aircraft=4 termination=None
~~~

smoke_local.py 运行固定首脑 Provider、固定航迹执行 Provider 和本地仿真。当前 smoke 用于验证接口、时间推进和状态快照，训练入口会在后续阶段逐步接入。

## bsl_aggressive

bsl_aggressive 位于：

~~~text
src/laglayer3/control/adapters/bsl_aggressive/
~~~

公开适配入口是：

~~~python
from laglayer3.control import BslAggressiveAdapter
~~~

控制链使用冻结 Actor、40 维观测、GRU、名义控制、残差组合和 QP 投影，最终输出副翼、升降舵、方向舵和油门控制量。权重和必要的运行文件已经包含在项目中。

## 项目文档

架构和实现顺序见：

- [重构设计决策](docs/restructuring_decisions.md)
- [目录结构设计](docs/directory_structure.md)
- [接口契约](docs/interfaces.md)
- [仿真设计](docs/simulation_design.md)
- [实现交接说明](docs/implementation_handoff.md)

## 固定航迹执行层训练

先生成固定航迹缓存：

~~~powershell
$env:PYTHONPATH = "src"
conda run -n jsbsim python scripts/training/generate_fixed_routes.py
~~~

运行 bsl_aggressive 和 JSBSim 的 5Hz 执行 smoke：

~~~powershell
conda run -n jsbsim python scripts/smoke_execution_bsl.py
~~~

启动 PPO 训练：

~~~powershell
conda run -n jsbsim python scripts/training/train_execution_ppo.py --updates 100 --output runs/execution/first_run
~~~

评估检查点：

~~~powershell
conda run -n jsbsim python scripts/evaluation/evaluate_execution.py runs/execution/first_run/execution.pt
~~~

训练环境固定使用 JSBSim 60 Hz、bsl_aggressive 5 Hz、MultiDiscrete([3,5,3]) 执行动作和配置中的六项奖励权重。训练前必须存在与航迹规格匹配的 datasets/execution/routes/basic_tracking.json。

