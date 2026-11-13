# LAGLayer3 目录结构设计

本文定义 LAGLayer3 的第一版目录边界。目录按模块和 seam 组织，不按旧项目文件名平移。新代码应优先放在 src/laglayer3 下；训练数据、模型和运行结果不进入源码包。

## 1. 根目录

~~~text
LAGLayer3/
├── src/laglayer3/          # 运行时代码和训练可复用模块
├── configs/                # 场景、协议、Provider 和训练配置
├── tools/                  # 数据生成、训练、评估和诊断入口
├── tests/                  # 单元、契约、集成和回归测试
├── docs/                   # 设计、ADR、迁移和实验文档
├── datasets/               # 本地生成数据，默认 gitignored
├── checkpoints/            # 模型检查点，默认 gitignored
├── runs/                   # 训练和评估产物，默认 gitignored
├── pyproject.toml
├── README.md
└── GLOSSARY.md
~~~

根目录只放项目入口和全局配置。模型、数据和运行产物与源码分开，避免训练结果被误当成代码依赖。

## 2. 源码包

~~~text
src/laglayer3/
├── domain/
├── simulation/
├── perception/
├── planning/
├── weapons/
├── safety/
├── flight_control/
├── policies/
├── execution/
├── learning/
├── runtime/
├── evaluation/
└── config/
~~~

每个一级目录是一个有明确 Interface 的模块集合。调用者依赖 Interface，具体 Implementation 通过 Adapter 或 Provider 注入。

### domain

~~~text
domain/
├── entities.py
├── observations.py
├── routes.py
├── tasks.py
├── actions.py
├── weapons.py
├── events.py
├── snapshots.py
├── initialization.py
├── termination.py
├── contracts.py
└── errors.py
~~~

domain 定义不依赖 PyTorch、JSBSim、Socket 或具体策略实现的稳定对象：

- WorldState 和 SimulationSnapshot；
- TeamObservation 和 LocalObservation；
- RoutePrediction 和 WaypointRouteTask；
- ExecutionAction；
- WeaponIntent 和 WeaponCommand；
- SimulationEvent 和 ExecutionStatus；
- ScenarioInitialization 和 TerminationStatus。

这是最稳定、最深的模块。其他模块可以依赖 domain，domain 不反向依赖任何运行实现。

### simulation

~~~text
simulation/
├── interface.py
├── clock.py
├── protocol/
│   ├── state_schema.py
│   ├── command_schema.py
│   ├── event_schema.py
│   ├── snapshot_schema.py
│   ├── initialization_schema.py
│   └── termination_schema.py
├── adapters/
│   ├── local_jsbsim/
│   │   ├── adapter.py
│   │   ├── aircraft.py
│   │   ├── missiles.py
│   │   └── state_mapper.py
│   ├── zk_socket/
│   │   ├── adapter.py
│   │   ├── transport.py
│   │   └── state_mapper.py
│   └── replay/
│       └── adapter.py
├── initialization/
│   ├── official.py
│   ├── combat_start.py
│   ├── custom.py
│   └── validators.py
├── termination/
│   ├── provider.py
│   ├── time_limit.py
│   ├── combat_result.py
│   ├── safety_failure.py
│   └── composition.py
└── snapshots/
    ├── builder.py
    ├── restore.py
    └── versioning.py
~~~

simulation 是仿真 Interface 的深模块。上层只认识 reset、step、snapshot、restore、events 和 termination_status。LocalJSBSimAdapter、ZKSocketAdapter 和 ReplayAdapter 是同一个 seam 的不同 Adapter。

第一版使用本地 JSBSim，但协议、状态、初始化、事件和终止语义按 ZK 契约定义。初始化和终止由独立模块组合，不能写死在环境 reset 中。

### perception

~~~text
perception/
├── interface.py
├── radar/
│   ├── model.py
│   ├── detection.py
│   └── noise.py
├── tracking/
│   ├── track.py
│   ├── manager.py
│   └── lifecycle.py
├── fog_of_war/
│   ├── visibility.py
│   ├── delay.py
│   └── confidence.py
├── communication/
│   ├── team_messages.py
│   └── fusion.py
└── encoders/
    ├── team_observation.py
    └── local_observation.py
~~~

perception 是 WorldState 到 TeamObservation / LocalObservation 的唯一入口。部署模式下首脑和执行 Actor 不能绕过 perception 读取仿真真值。

### planning

~~~text
planning/
├── interface.py
├── frames.py
├── route_prediction.py
├── route_window.py
├── route_tracking.py
├── route_validation.py
├── primitives/
│   ├── base.py
│   ├── formation.py
│   ├── approach.py
│   ├── attack.py
│   ├── support.py
│   ├── bait.py
│   ├── pincer.py
│   ├── missile_evade.py
│   ├── sliceback.py
│   └── return.py
└── compilers/
    ├── analytic_integrator.py
    └── fkr_route_compiler.py
~~~

planning 负责相对航迹表示、固定局部坐标系、航迹窗口、路线验证和可组合机动 primitive。第一版首脑模型直接输出 RoutePrediction；analytic_integrator 用于生成解析教师航迹，fkr_route_compiler 作为规则意图转换工具保留。

planning 不负责仿真推进，也不负责 PPO。它只产生和验证计划对象。

### weapons

~~~text
weapons/
├── interface.py
├── manager.py
├── target_lock.py
├── launch_validator.py
├── weapon_ledger.py
├── missile_lifecycle.py
├── adapters/
│   ├── local_jsbsim.py
│   └── zk_protocol.py
└── policies/
    ├── weapon_intent.py
    └── weapon_rules.py
~~~

WeaponManager 接收首脑层 WeaponIntent，检查锁定、距离、角度、冷却、余弹、任务权限和安全条件，输出 WeaponCommand 或拒绝原因。请求、接受、平台确认和真实导弹实体分开记录。

### safety

~~~text
safety/
├── interface.py
├── supervisor.py
├── constraints/
│   ├── boundary.py
│   ├── altitude.py
│   ├── speed.py
│   ├── overload.py
│   ├── collision.py
│   └── missile_defense.py
├── route_projection.py
└── reports.py
~~~

训练阶段保留物理稳定性和终止保护。推理与正式对战阶段启用完整 SafetySupervisor、航迹验证和覆盖记录。安全模块不替代首脑层规划，也不改变执行层任务语义。

### flight_control

~~~text
flight_control/
├── interface.py
├── adapters/
│   ├── bsl_aggressive/
│   │   ├── adapter.py
│   │   ├── observation_mapper.py
│   │   ├── action_mapper.py
│   │   └── state_store.py
│   ├── bsl/
│   └── pid/
├── contracts.py
└── registry.py
~~~

flight_control 的 seam 接收飞机状态和 ExecutionAction，输出 LowLevelCommand 与 ControllerFeedback。bsl_aggressive 的 40 维观测、GRU、QP 和内部状态都隐藏在 Adapter 内。

### policies

~~~text
policies/
├── interface.py
├── provider.py
├── registry.py
├── fixed.py
├── rule/
│   ├── interface.py
│   └── fkr_v3/
│       ├── adapter.py
│       ├── context.py
│       ├── rules/
│       ├── arbitration.py
│       ├── route_teacher.py
│       └── weapon_teacher.py
└── model/
    ├── common/
    │   ├── embeddings.py
    │   ├── entity_encoder.py
    │   ├── recurrent.py
    │   └── distributions.py
    ├── head/
    │   ├── model.py
    │   ├── route_head.py
    │   ├── weapon_head.py
    │   ├── moe/
    │   │   ├── general.py
    │   │   ├── route.py
    │   │   ├── weapon.py
    │   │   ├── router.py
    │   │   └── balancing.py
    │   └── provider.py
    └── execution/
        ├── model.py
        ├── route_encoder.py
        ├── state_encoder.py
        ├── actor.py
        ├── critic.py
        └── provider.py
~~~

policies 定义可替换 Provider 的 Interface 和 Implementation：

- RulePolicyProvider；
- FixedModelProvider；
- TrainableModelProvider；
- HybridProvider。

首脑模型是团队级 Actor，包含实体编码、常激活通用专家、航迹 MoE、武器 MoE、连续航迹头和离散武器头。执行模型是单机航迹跟踪 Actor，默认输出 MultiDiscrete([3,5,3])。

### execution

~~~text
execution/
├── interface.py
├── coordinator.py
├── route_tracker.py
├── status.py
├── feedback.py
└── adapters/
    ├── route_task.py
    └── controller_action.py
~~~

execution 是运行时执行模块，不是模型目录。它把首脑 RoutePrediction 和当前 LocalObservation 组织成执行模型输入，并管理 running、completed、failed、expired、safety_hold 和 replan_requested 状态。

execution 不改变首脑任务，不处理武器，不重新分配目标。

### learning

~~~text
learning/
├── graph/
│   ├── training_graph.py
│   ├── nodes.py
│   ├── freeze.py
│   ├── gradients.py
│   └── manifests.py
├── common/
│   ├── buffers.py
│   ├── checkpoints.py
│   ├── normalization.py
│   ├── schedules.py
│   └── logging.py
├── teachers/
│   ├── analytic/
│   │   ├── collector.py
│   │   ├── dataset.py
│   │   └── schema.py
│   └── simulated/
│       ├── collector.py
│       ├── dataset.py
│       └── schema.py
├── head/
│   ├── datasets.py
│   ├── buffer.py
│   ├── bc_trainer.py
│   ├── ppo_trainer.py
│   ├── mappo_trainer.py
│   └── losses.py
├── execution/
│   ├── datasets.py
│   ├── buffer.py
│   ├── bc_trainer.py
│   ├── ppo_trainer.py
│   └── losses.py
├── flight_control/
│   └── trainers/
└── credit/
    ├── analyzer.py
    ├── counterfactuals.py
    ├── metrics.py
    ├── confidence.py
    └── report.py
~~~

learning 是训练模块，不进入部署运行核心。每一层有独立 Trainer、optimizer、loss、buffer、checkpoint 和 metrics。TrainingGraph 根据配置决定任意层固定、训练或联合训练，默认层间梯度截断。

第一版 CFC 放在 learning/credit，作为独立 ContributionAnalyzer 和指标模块，不直接重分配首脑训练奖励。

### runtime

~~~text
runtime/
├── loop.py
├── scheduler.py
├── providers.py
├── episode.py
├── events.py
├── trace.py
└── clocks.py
~~~

runtime 负责 60Hz 仿真、5Hz 执行层、1Hz 首脑层和事件触发重规划的调度。它不实现具体策略、飞控或武器规则，只依赖各模块 Interface。

### evaluation

~~~text
evaluation/
├── interface.py
├── metrics/
│   ├── registry.py
│   ├── head.py
│   ├── execution.py
│   ├── weapons.py
│   ├── flight_control.py
│   └── system.py
├── scenarios/
├── replay.py
├── reports.py
└── protocols.py
~~~

evaluation 通过 MetricRegistry 注册和移除指标。原始模型输出、验证输出、执行结果和平台确认分别记录，避免只能看到最终结果而无法定位问题。

### config

~~~text
config/
├── loader.py
├── validation.py
├── versions.py
├── schemas/
│   ├── simulation.py
│   ├── perception.py
│   ├── policies.py
│   ├── routes.py
│   ├── training.py
│   └── evaluation.py
└── defaults/
    ├── simulation.yaml
    ├── head.yaml
    ├── execution.yaml
    ├── flight_control.yaml
    ├── training.yaml
    └── evaluation.yaml
~~~

config 只负责加载、校验和版本，不在配置模块中实现策略逻辑。

## 3. 工具入口

~~~text
tools/
├── simulation/
│   ├── run_local_episode.py
│   ├── replay_snapshot.py
│   └── validate_protocol.py
├── teachers/
│   ├── collect_analytic_head_teacher.py
│   └── collect_simulated_head_teacher.py
├── training/
│   ├── train_head_bc.py
│   ├── train_head_ppo.py
│   ├── train_execution_bc.py
│   ├── train_execution_ppo.py
│   └── train_graph.py
├── evaluation/
│   ├── evaluate_episode.py
│   ├── evaluate_checkpoint.py
│   └── compare_checkpoints.py
└── diagnostics/
    ├── inspect_dataset.py
    ├── inspect_route.py
    ├── inspect_moe.py
    └── inspect_trace.py
~~~

工具只负责命令行参数、配置加载和调用深模块。业务逻辑不应重新写在 tools 中。

## 4. 配置和数据目录

~~~text
configs/
├── scenarios/
│   ├── official/
│   ├── combat_start/
│   └── custom/
├── providers/
│   ├── head/
│   ├── execution/
│   ├── flight_control/
│   └── weapons/
├── training/
│   ├── head_bc/
│   ├── head_ppo/
│   ├── execution_bc/
│   ├── execution_ppo/
│   └── joint/
└── evaluation/

datasets/
├── head_routes/
│   ├── analytic/
│   └── simulated/
├── execution/
└── manifests/

checkpoints/
├── head/
├── execution/
├── flight_control/
└── manifests/
~~~

数据集和 checkpoint 通过 schema、场景、Provider、checkpoint、航迹来源和随机种子生成 dataset_id。训练发现数据缺失或版本不匹配时直接失败。

## 5. 依赖方向

推荐依赖方向：

~~~text
domain
  ↑
simulation / perception / planning / weapons / safety
  ↑
policies / execution / flight_control
  ↑
runtime / evaluation

learning → domain + simulation + policies + execution + evaluation
tools → config + runtime + learning + evaluation
~~~

关键约束：

- domain 不依赖任何外部运行实现；
- policies 不直接依赖 JSBSim 或飞控内部类；
- execution 不依赖 WeaponManager 的内部实现；
- learning 不反向成为 runtime 的必需依赖；
- tools 不复制业务逻辑；
- 通用 utils 目录不作为复杂逻辑的归宿，逻辑应放在拥有它的模块中。

## 6. 第一阶段实际创建顺序

第一阶段不必一次创建所有文件。建议按深模块的 Interface 顺序创建：

1. domain 对象、schema 和错误类型；
2. simulation protocol、LocalJSBSimAdapter、初始化和终止；
3. perception 的最小战争迷雾和观测编码；
4. planning 的局部坐标、RoutePrediction、解析航迹积分器；
5. policies 的 Provider、固定 FKR Adapter 和首脑网络模板；
6. flight_control 的 bsl_aggressive Adapter；
7. execution 的路线输入、执行 Actor 模板和状态机；
8. runtime 的三频率调度和完整 trace；
9. teachers 的解析航迹数据生成；
10. learning 的 TrainingGraph 和执行层 BC/PPO；
11. evaluation 的 MetricRegistry；
12. CFC analyzer 和仿真教师数据。

每一步都应先建立 Interface 和契约测试，再实现 Adapter。不要先迁移旧项目的大文件，再反过来猜新接口。

