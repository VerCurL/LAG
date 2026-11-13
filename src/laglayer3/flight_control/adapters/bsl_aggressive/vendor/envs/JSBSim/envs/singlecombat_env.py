import numpy as np
import copy
from .env_base import BaseEnv
from ..tasks import SingleCombatTask, SingleCombatDodgeMissileTask, HierarchicalSingleCombatDodgeMissileTask, \
    HierarchicalSingleCombatShootTask, SingleCombatShootMissileTask, HierarchicalSingleCombatTask, \
    AnalyticHierarchicalSingleCombatTask, C23HybridHierarchicalSingleCombatTask, C27GovernedHybridTask
from ..human_task.HumanSingleCombatTask import  HumanSingleCombatTask
from ..utils.utils import NEU2LLA


class SingleCombatEnv(BaseEnv):
    """
    SingleCombatEnv is an one-to-one competitive environment.
    """
    def __init__(self, config_name: str):
        super().__init__(config_name)
        # Env-Specific initialization here!
        assert len(self.agents.keys()) == 2, f"{self.__class__.__name__} only supports 1v1 scenarios!"
        self.init_states = None

    def load_task(self):
        taskname = getattr(self.config, 'task', None)
        if taskname == 'singlecombat':
            self.task = SingleCombatTask(self.config)
        elif taskname == 'hierarchical_singlecombat':
            self.task = HierarchicalSingleCombatTask(self.config)
        elif taskname == 'analytic_hierarchical_singlecombat':
            self.task = AnalyticHierarchicalSingleCombatTask(self.config)
        elif taskname == 'c23_hybrid_hierarchical_singlecombat':
            self.task = C23HybridHierarchicalSingleCombatTask(self.config)
        elif taskname == 'c27_governed_hybrid':
            self.task = C27GovernedHybridTask(self.config)
        elif taskname == 'singlecombat_dodge_missile':
            self.task = SingleCombatDodgeMissileTask(self.config)
        elif taskname == 'singlecombat_shoot':
            self.task = SingleCombatShootMissileTask(self.config)
        elif taskname == 'hierarchical_singlecombat_dodge_missile':
            self.task = HierarchicalSingleCombatDodgeMissileTask(self.config)
        elif taskname == 'hierarchical_singlecombat_shoot':
            self.task = HierarchicalSingleCombatShootTask(self.config)
        elif taskname == 'HumanSingleCombat':
            self.task = HumanSingleCombatTask(self.config)
        else:
            raise NotImplementedError(f"Unknown taskname: {taskname}")

    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.reset_simulators()
        self.task.reset(self)
        obs = self.get_obs()
        return self._pack(obs)

    def reset_simulators(self):
        if bool(
            getattr(
                self.config,
                "CombatInitialStateCurriculum_enable",
                False,
            )
        ):
            self._reset_randomized_combat_geometry()
            return
        # switch side
        if self.init_states is None:
            self.init_states = [sim.init_state.copy() for sim in self.agents.values()]
        # self.init_states[0].update({
        #     'ic_psi_true_deg': (self.np_random.uniform(270, 540))%360,
        #     'ic_h_sl_ft': self.np_random.uniform(17000, 23000),
        # })
        init_states = self.init_states.copy()
        self.np_random.shuffle(init_states)
        for idx, sim in enumerate(self.agents.values()):
            sim.reload(init_states[idx])
        self._tempsims.clear()

    def _reset_randomized_combat_geometry(self):
        """Sample an opt-in, auditable 1v1 geometry around the arena center."""
        if self.init_states is None:
            self.init_states = [
                copy.deepcopy(sim.init_state)
                for sim in self.agents.values()
            ]
        if len(self.init_states) != 2:
            raise ValueError("randomized combat geometry requires two aircraft")

        def choices(name, default):
            values = np.asarray(
                getattr(self.config, name, default),
                dtype=np.float64,
            ).reshape(-1)
            if values.size == 0 or not np.all(np.isfinite(values)):
                raise ValueError("{} must contain finite choices".format(name))
            return values

        range_m = float(
            self.np_random.choice(
                choices(
                    "CombatInitialStateCurriculum_ranges_m",
                    [4000.0, 6000.0, 8000.0],
                )
            )
        )
        relative_bearing_deg = float(
            self.np_random.choice(
                choices(
                    "CombatInitialStateCurriculum_relative_bearings_deg",
                    [-120.0, -60.0, -30.0, 30.0, 60.0, 120.0],
                )
            )
        )
        ego_heading_deg = float(self.np_random.uniform(0.0, 360.0))
        enemy_heading_delta_deg = float(
            self.np_random.choice(
                choices(
                    "CombatInitialStateCurriculum_enemy_heading_deltas_deg",
                    [-180.0, -90.0, -45.0, 0.0, 45.0, 90.0, 180.0],
                )
            )
        )
        ego_altitude_ft = float(
            self.np_random.choice(
                choices(
                    "CombatInitialStateCurriculum_altitudes_ft",
                    [18000.0, 22000.0, 26000.0],
                )
            )
        )
        altitude_delta_ft = float(
            self.np_random.choice(
                choices(
                    "CombatInitialStateCurriculum_altitude_deltas_ft",
                    [-1500.0, 0.0, 1500.0],
                )
            )
        )
        speed_choices_mps = choices(
            "CombatInitialStateCurriculum_speeds_mps",
            [220.0, 250.0, 280.0],
        )
        ego_speed_mps = float(self.np_random.choice(speed_choices_mps))
        enemy_speed_mps = float(self.np_random.choice(speed_choices_mps))

        absolute_bearing_rad = np.deg2rad(
            ego_heading_deg + relative_bearing_deg
        )
        delta_north_m = range_m * float(np.cos(absolute_bearing_rad))
        delta_east_m = range_m * float(np.sin(absolute_bearing_rad))
        ego_lla = NEU2LLA(
            -0.5 * delta_north_m,
            -0.5 * delta_east_m,
            ego_altitude_ft * 0.3048,
            self.center_lon,
            self.center_lat,
            self.center_alt,
        )
        enemy_altitude_ft = ego_altitude_ft + altitude_delta_ft
        enemy_lla = NEU2LLA(
            0.5 * delta_north_m,
            0.5 * delta_east_m,
            enemy_altitude_ft * 0.3048,
            self.center_lon,
            self.center_lat,
            self.center_alt,
        )

        states = [copy.deepcopy(state) for state in self.init_states]
        states[0].update(
            {
                "ic_long_gc_deg": float(ego_lla[0]),
                "ic_lat_geod_deg": float(ego_lla[1]),
                "ic_h_sl_ft": ego_altitude_ft,
                "ic_psi_true_deg": ego_heading_deg % 360.0,
                "ic_u_fps": ego_speed_mps / 0.3048,
            }
        )
        states[1].update(
            {
                "ic_long_gc_deg": float(enemy_lla[0]),
                "ic_lat_geod_deg": float(enemy_lla[1]),
                "ic_h_sl_ft": enemy_altitude_ft,
                "ic_psi_true_deg": (
                    ego_heading_deg + enemy_heading_delta_deg
                )
                % 360.0,
                "ic_u_fps": enemy_speed_mps / 0.3048,
            }
        )
        for sim, state in zip(self.agents.values(), states):
            sim.reload(state)
        self._tempsims.clear()
        self._combat_initial_state_context = {
            "range_m": range_m,
            "relative_bearing_deg": relative_bearing_deg,
            "ego_heading_deg": ego_heading_deg,
            "enemy_heading_delta_deg": enemy_heading_delta_deg,
            "ego_altitude_ft": ego_altitude_ft,
            "enemy_altitude_ft": enemy_altitude_ft,
            "ego_speed_mps": ego_speed_mps,
            "enemy_speed_mps": enemy_speed_mps,
        }
