import numpy as np

class SituationExtractor:

    AIRCRAFT_DIM = 24
    MISSILE_DIM = 14

    def __init__(self, max_missiles: int = 16):
        self.max_missiles = max_missiles

    def extract(self, env):
        agent_ids = list(env.agents.keys())
        all_missiles = []
        for sim in env.agents.values():
            all_missiles.extend(sim.launch_missiles)
        agent_index = {uid: i for i, uid in enumerate(agent_ids)}
        missile_index = {missile.uid: i for i, missile in enumerate(all_missiles)}

        aircraft = np.zeros((len(agent_ids), self.AIRCRAFT_DIM), dtype=np.float32)
        for i, uid in enumerate(agent_ids):
            # ---------- 飞机信息 ----------
            sim = env.agents[uid]
            pos = sim.get_position()
            vel = sim.get_velocity()
            rpy = sim.get_rpy()

            # ---------- 威胁导弹信息 ----------
            threat_missile = sim.check_most_dangerous_missile_warning()
            if threat_missile is not None:
                threat_missile_exist = 1.0
                m_pos = threat_missile.get_position()
                m_vel = threat_missile.get_velocity()
            else:
                threat_missile_exist = 0.0
                m_pos = np.zeros(3, dtype=np.float32)
                m_vel = np.zeros(3, dtype=np.float32)

            # ---------- 发射导弹信息 ----------
            shoot_missiles = sim.launch_missiles
            shoot_index = [-1] * 2
            for index, m in enumerate(shoot_missiles):
                shoot_index[index] = missile_index[m.uid]

            # ---------- 飞机信息汇总 ----------
            aircraft[i] = np.array([
                0.0 if uid.startswith("A") else 1.0,
                float(sim.is_alive),
                float(sim.is_shotdown),
                float(sim.is_crash),
                float(sim.num_left_missiles),
                pos[0], pos[1], pos[2],
                vel[0], vel[1], vel[2],
                rpy[0], rpy[1], rpy[2],
                float(np.linalg.norm(vel)),
                threat_missile_exist,
                m_pos[0], m_pos[1], m_pos[2],
                m_vel[0], m_vel[1], m_vel[2],
                *shoot_index
            ], dtype=np.float32)

        missiles = np.zeros((self.max_missiles, self.MISSILE_DIM), dtype=np.float32)
        for j, missile in enumerate(all_missiles[:self.max_missiles]):
            parent_id = missile.parent_aircraft.uid if missile.parent_aircraft else ""
            target_id = missile.target_aircraft.uid if missile.target_aircraft else ""
            parent_idx = agent_index.get(parent_id, -1)
            target_idx = agent_index.get(target_id, -1)

            pos = missile.get_position()
            vel = missile.get_velocity()

            missiles[j] = np.array([
                float(missile.is_alive),
                float(missile.is_success),
                float(missile.is_done),
                float(missile.m_status),
                float(parent_idx),
                float(target_idx),
                pos[0], pos[1], pos[2],
                vel[0], vel[1], vel[2],
                float(np.linalg.norm(vel)),
                float(missile.target_distance if missile.target_aircraft else 0.0),
            ], dtype=np.float32)

        return {
            "aircraft": aircraft,
            "missiles": missiles
        }

