# 策略框架与 FKR v3

## FKR v3 的定位

FKR v3 不是新系统的基础架构，而是第一个 `PolicyProvider` Adapter。它将比赛中已经验证的规则策略封装成统一的 `decide()` 接口。

FKR v3 内部可以保留：

- 双翼和牵制支援；
- 进场和编队；
- 攻击窗口；
- 导弹防御；
- 无弹撤离；
- 边界和高度保护；
- 武器请求和发射状态机。

但它对外不应直接暴露 Socket 字典、规则内部变量或 BSL 类。

## 规则决策流水线

```text
TacticalObservation
    ↓
Rule Context
    ↓
Rule Candidates
    ↓
Priority Arbitration
    ↓
DecisionBundle
    ↓
Intent / Weapon Adapter
```

每条规则只产生候选决策。最终决策由仲裁器选择，不能依赖 Python 文件调用顺序隐式决定优先级。

## 普通和紧急输出

普通规则和紧急规则都输出首脑层航迹与规则阶段信息。FKR 的连续参考由解析航迹积分器转换为相对局部航迹，交给执行层跟踪。

复合机动通过 primitive 序列表达，例如 Sliceback 加 45 度摆尾。规则不绕过执行层直接输出飞控舵面。

武器规则只输出 WeaponIntent，最终发射由 WeaponManager 检查和确认。

## 后续策略

同一接口未来可以替换为：

- `FkrV3Policy`；
- `ImitationPolicy`；
- `ExecutionPPOPolicy`；
- `HybridPolicy`；
- `ResidualPolicy`。

这使规则可以作为教师、基线、模仿数据生成器和安全回退策略，而不是一次性比赛脚本。

