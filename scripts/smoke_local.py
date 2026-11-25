"""运行 LAGLayer3 的本地最小闭环。

脚本默认使用轻量运动学后端，也可以切换到项目内的 JSBSim 数据，
用于检查首脑 Provider、执行 Provider 和仿真适配器能否协同推进。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from laglayer3.domain.models import AircraftState, ScenarioInitialization, TeamObservation
from laglayer3.policies.fixed_execution import FixedRouteExecutionProvider
from laglayer3.policies.fixed_head import FixedRouteHeadProvider
from laglayer3.runtime.loop import LayeredRuntime
from laglayer3.simulation.local_jsbsim import LocalJSBSimAdapter


def build_initialization(count: int = 4) -> ScenarioInitialization:
    """构造 smoke 回合使用的同方飞机初始状态。"""
    aircraft = tuple(
        AircraftState(
            aircraft_id="red_{}".format(index),
            side="red",
            position_ned_m=(float(index * 1000), 0.0, -8000.0),
            velocity_ned_mps=(220.0, 0.0, 0.0),
            heading_deg=0.0,
            altitude_m=8000.0,
            speed_mps=220.0,
        )
        for index in range(count)
    )
    return ScenarioInitialization(scenario_id="smoke", aircraft=aircraft)


def main() -> None:
    """解析参数，组装固定策略和仿真适配器，并运行短回合。"""
    parser = argparse.ArgumentParser(description="运行 LAGLayer3 本地最小闭环")
    parser.add_argument("--steps", type=int, default=120, help="最多推进的仿真步数")
    parser.add_argument(
        "--backend",
        choices=("kinematic", "jsbsim"),
        default="kinematic",
        help="选择轻量运动学后端或真实 JSBSim 后端",
    )
    args = parser.parse_args()

    # 两个固定 Provider 只用于验证运行时接口，后续可替换为规则或模型 Provider。
    simulation = LocalJSBSimAdapter(
        step_seconds=1.0 / 60.0,
        time_limit_seconds=20.0,
        backend=args.backend,
        data_root="third_party/jsbsim_data" if args.backend == "jsbsim" else None,
    )
    runtime = LayeredRuntime(
        simulation=simulation,
        head=FixedRouteHeadProvider(horizon_seconds=20.0, interval_seconds=1.0),
        execution=FixedRouteExecutionProvider(),
    )
    snapshot = runtime.reset(build_initialization())
    for _ in range(args.steps):
        # 当前快照生成团队观测，运行时按首脑层和执行层的周期调度策略。
        observation = TeamObservation(own_aircraft=snapshot.aircraft, timestamp=snapshot.timestamp)
        snapshot = runtime.step(observation)
        if simulation.termination_status().terminated:
            # 终止条件触发后不再向适配器发送新的动作。
            break
    print(
        "timestamp={:.3f} aircraft={} termination={}".format(
            snapshot.timestamp,
            len(snapshot.aircraft),
            simulation.termination_status().reason,
        )
    )


if __name__ == "__main__":
    main()
