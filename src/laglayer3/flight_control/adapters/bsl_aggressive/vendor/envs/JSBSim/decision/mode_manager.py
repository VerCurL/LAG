class ModeManager:
    CALM = "calm"
    ATTACK = "attack"
    DEFEND = "defend"
    RECOVER = "recover"

    def __init__(
        self,
        dt: float,
        defend_enter: float = 0.72,
        defend_exit: float = 0.55,
        attack_ao_deg: float = 38.0,
        attack_range_m: float = 6800.0,
        attack_threat_max: float = 0.55,
        recover_delta_es: float = -1500.0,
        recover_safety_margin: float = 0.20,
        dwell_time_s: float = 1.2,
    ):
        self.dt = float(dt)
        self.defend_enter = float(defend_enter)
        self.defend_exit = float(defend_exit)
        self.attack_ao_deg = float(attack_ao_deg)
        self.attack_range_m = float(attack_range_m)
        self.attack_threat_max = float(attack_threat_max)
        self.recover_delta_es = float(recover_delta_es)
        self.recover_safety_margin = float(recover_safety_margin)

        self.dwell_steps = max(1, int(round(dwell_time_s / max(self.dt, 1e-6))))

        self.mode = self.CALM
        self.last_switch_step = 0

    def reset(self):
        self.mode = self.CALM
        self.last_switch_step = 0

    def _can_switch(self, step_idx: int) -> bool:
        return (step_idx - self.last_switch_step) >= self.dwell_steps

    def _permit_by_mode(self, mode: str) -> float:
        if mode == self.DEFEND:
            return 1.0
        if mode == self.ATTACK:
            return 0.75
        if mode == self.RECOVER:
            return 0.35
        return 0.0

    def step(
        self,
        step_idx: int,
        s_threat: float,
        ao_deg: float,
        range_m: float,
        missile_alert: bool,
        delta_es: float,
        safety_margin: float,
    ):
        target_mode = self.mode

        if missile_alert or s_threat >= self.defend_enter:
            target_mode = self.DEFEND
        elif self.mode == self.DEFEND and s_threat > self.defend_exit:
            target_mode = self.DEFEND
        elif (delta_es <= self.recover_delta_es) or (safety_margin <= self.recover_safety_margin):
            target_mode = self.RECOVER
        elif (ao_deg <= self.attack_ao_deg) and (range_m <= self.attack_range_m) and (s_threat <= self.attack_threat_max):
            target_mode = self.ATTACK
        else:
            target_mode = self.CALM

        if target_mode != self.mode and self._can_switch(step_idx):
            self.mode = target_mode
            self.last_switch_step = step_idx

        permit_mode = self._permit_by_mode(self.mode)
        return self.mode, permit_mode
