# 武器与安全设计

## 武器链路

```text
WeaponIntent
    ↓
TargetLock
    ↓
LaunchValidator
    ↓
WeaponCommand
    ↓
PlatformAdapter
    ↓
LaunchConfirmation
    ↓
MissileLedger
```

中距弹和近距弹分别维护模式、锁定、余弹、挂点、距离、角度、目标占用和生命周期。发射请求不等于平台已经发射，必须通过余弹变化或平台导弹实体确认。

## 安全链路

```text
ProposedIntent
    ↓
SafetySupervisor
    ↓
AcceptedIntent
    ↓
FlightController
```

安全规则包括：

- 地图边界；
- 高度上下限；
- 速度上下限；
- 最大过载；
- 碰撞风险；
- 导弹防御；
- 无弹撤离；
- 飞机失效。

每个安全规则必须有触发条件、持续条件、释放条件、优先级和覆盖原因。

