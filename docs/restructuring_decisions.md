# LAGLayer3 重构设计决策

本文汇总首脑层、执行层、飞控层、仿真、训练和数据链路的已确认设计。尚未确定的内容继续记录在 open_questions.md。

## 1. 总体边界

LAGLayer3 面向 4v4 空战，使用三层控制链：

~~~text
WorldState
  ↓  Radar / Fog of War / Communication
TeamObservation
  ↓
Head Policy Provider
  ↓
Team Route + Weapon Intent
  ↓
Execution Layer
  ↓
Flight Controller Adapter
  ↓
Local JSBSim / ZK-compatible simulation
~~~

默认频率：

~~~text
仿真和飞控  60 Hz
执行层      5 Hz
首脑层      1 Hz
~~~

首脑层拥有团队任务规划权。执行层是单机黑盒航迹跟踪器，只负责把航迹飞好。飞控层把执行动作转成平台控制量。武器意图由首脑层产生，并由独立 WeaponManager 校验和确认。

## 2. 首脑层输出

第一版首脑模型直接输出连续航迹，不把 Compiler 放在模型和执行层之间。未来可以增加计划模型加 Compiler 的两阶段模式，但不能改变当前航迹接口。

每架飞机输出：

~~~text
route_points[K, 4]
  forward_m
  right_m
  altitude_m
  speed_mps

route_confidence
weapon_request
weapon_type
target_id
fire_permission
fire_window
request_duration
~~~

K 等于 route_horizon_seconds 除以 route_interval_seconds。两个参数均可配置，默认未来 20 秒、每秒一个点。时间索引由配置产生，不由模型输出。

航迹使用决策时刻建立的固定局部坐标系：

~~~text
原点：当前飞机位置
x 轴：当前飞机航向
y 轴：当前飞机右侧
航迹点：相对局部坐标
~~~

不得使用绝对经纬度或世界绝对坐标作为首脑航迹输出。首脑每秒生成滚动航迹；关键事件可以立即触发重规划。新旧航迹切换必须检查连续性并平滑衔接。

武器头与航迹头独立，但共享团队编码。武器请求的有效期通常短于航迹有效期。实际发射必须经过锁定、距离、角度、冷却、余弹、任务权限和安全校验。

## 3. 首脑网络与 MoE

首脑层是一个团队级 Actor，一次输出所有己方飞机。模型采用实体编码和 padding mask，不把 4 架飞机固定展平：

~~~text
aircraft / enemy / missile entities
        ↓
team entity encoder
        ↓
常激活通用专家
        ↓
route branch       weapon branch
        ↓                   ↓
route MoE             weapon MoE
        ↓                   ↓
per-aircraft decoders
~~~

MoE 分为常激活通用专家、航迹分支专家和武器分支专家。航迹和武器分支各自拥有通用专家与特定专家。特定专家按每架飞机分别路由，但可以被多个任务共享。

任务到候选专家集合、top_k、常激活专家数和总专家数均可配置，默认特定专家使用 Top-2 稀疏路由。FKR 任务标签先用于稳定路由，之后允许 Router 自主调整，并使用负载均衡约束防止专家塌缩。

复合机动使用 primitive 序列：

~~~text
单独掉头 = [sliceback]
导弹规避 = [sliceback, tail_offset_45]
~~~

专家输出局部机动片段，复合任务组合这些片段。路线专家链和武器专家链独立记录。

## 4. 执行层

执行层只负责单机航迹跟踪，不负责团队任务规划、目标和角色修改、航迹重规划或武器发射。

输入是固定长度航迹窗口和当前飞机状态：

~~~text
route_points[K, 4]
current_position_in_route_frame
current_velocity_in_route_frame
current_heading_in_route_frame
current_altitude
current_speed
current_flight_state
previous_execution_action
~~~

首脑决策时飞机位于局部坐标原点。执行层把当前状态投影到同一固定坐标系。路线进度、最近点和时间误差只用于奖励和评估。

网络采用 route encoder、state encoder、GRU 和共享 Actor，默认动作空间为 MultiDiscrete([3, 5, 3])，分别表示高度、航向和速度调整。武器动作维度从执行层移除。

执行层奖励只评价位置、航向、高度、速度跟踪误差，能量偏差，控制代价，动作切换惩罚，飞行稳定性，航迹完成、超时和失败。执行层不接收击落、目标选择、发弹或团队角色奖励。

## 5. FKR 和教师数据

FKR v3 是 RulePolicyProvider，通过 Adapter 接入新领域对象，不直接复制比赛单体 runtime，也不依赖飞控内部类。

解析教师不启动 JSBSim，由 FKR 规则参考和确定性运动学积分器生成 planned_route。仿真教师使用固定首脑、固定执行层和固定飞控，在本地 JSBSim 中运行，保存 planned_route、executed_route、route_error、events 和 outcomes。

两类数据由独立脚本预生成并保存到本地。训练只读取已有数据；数据不存在、版本不匹配或配置指纹不匹配时直接终止。每个数据集使用场景、初始化、观测、动作、路线 schema、航迹来源、策略版本、checkpoint 和随机种子生成唯一数据集 ID。

教师记录保存规则候选、最终规则、角色、目标、支援关系、phase、mode、计划航迹、执行航迹、武器请求、管理器结果、安全覆盖、事件和结果。

## 6. Provider 和训练图

首脑层、执行层和飞控层都通过 Provider 配置固定、训练或联合训练。每个 Provider 支持 train、eval、freeze、unfreeze、reset、checkpoint 和 restore。

TrainingGraph 统一调度任意可训练节点组合：只训练首脑层、只训练执行层、首脑层加执行层、执行层加飞控层或三层联合训练。

每层有独立 Trainer、optimizer、loss、buffer、checkpoint 和 metrics。默认层间梯度截断，只有显式配置时才开启联合可微训练。每次训练和评估保存 Provider、checkpoint、冻结状态、策略版本和接口 schema 版本。

## 7. 仿真、初始化和终止

第一版使用本地 JSBSim，但遵循 ZK 的状态、指令、时间、事件、初始化和终止语义。上层只依赖：

~~~text
reset(initialization)
step(command)
snapshot()
events()
restore(snapshot)
termination_status()
~~~

声明式 ScenarioInitialization 支持 official、combat_start 和 custom，可覆盖飞机、导弹、传感器、通信、战争迷雾、武器账本、策略、执行和飞控状态。

TerminationProvider 可组合，支持 time_limit、all_enemy_destroyed、all_own_destroyed、team_no_longer_operational、safe_return、boundary_failure、simulation_stalled、missile_overtime 和 external_platform_done。

SimulationSnapshot 必须可以恢复动力学、武器、感知、战争迷雾、首脑和执行 RNN、飞控隐藏状态、随机数、事件和终止状态。

训练阶段保留物理稳定性和仿真终止保护。完整 SafetySupervisor、航迹修复和安全覆盖主要在推理和正式对战阶段启用，并分别记录原始航迹、验证航迹、接受航迹和覆盖原因。

首脑层只接收执行层的最小反馈：running、completed、failed、expired、safety_hold、event 和 replan_requested。详细误差和安全内部状态只用于校验、日志和下层训练。

## 8. CFC 与指标

第一版 CFC 是独立的 ContributionAnalyzer，参与运行期记录和评估，但不直接改首脑训练奖励。反事实对象迁移为 route、role、target、skill / primitive sequence、support relation、weapon intent 和 team plan。

指标通过 MetricRegistry 插拔。计划执行率、存活、攻击机会、团队收益、联合贡献和排序一致性可以独立添加或移除。真实反事实验证稳定后，计划级 CFC 再接入首脑 PPO 或辅助优势。

## 9. 第一阶段闭环

第一阶段先实现可复现闭环和网络模板：

~~~text
custom initialization
  ↓
ZK-compatible LocalJSBSIM
  ↓
fog-of-war TeamObservation
  ↓
fixed FKR Head Provider
  ↓
analytic route dataset / replay
  ↓
fixed execution provider
  ↓
bsl_aggressive adapter
  ↓
JSBSim
  ↓
snapshot / event / termination / metrics / full trace
~~~

之后依次接入 execution BC / PPO、head BC、head PPO / MAPPO、simulated teacher data、weapon MoE、plan-level CFC 和 joint training。

