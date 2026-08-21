from collections import defaultdict

import numpy as np

from scripts.AeroTAF.data.annotation import annotate_episode_detail, fit_detail_thresholds
from scripts.AeroTAF.data.schema import (
    CATEGORY_NAMES,
    CATEGORY_STABLE,
    DetailAnnotationConfig,
    EVENT_NAMES,
)


COUNTERFACTUAL_ACTION_NAMES = (
    "previous",
    "no_op",
    "invert_maneuver",
    "invert_heading",
    "invert_altitude",
    "invert_velocity",
)


def parse_counterfactual_actions(value):
    names = tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    if not names:
        raise ValueError("--CFC-counterfactual-actions must contain at least one action")
    unknown = sorted(set(names) - set(COUNTERFACTUAL_ACTION_NAMES))
    if unknown:
        raise ValueError(f"unknown CFC counterfactual actions: {unknown}")
    return names


def build_counterfactual_action(kind, current_action, previous_action):
    action = np.asarray(current_action).copy()
    shoot = action[..., 3].copy() if action.shape[-1] > 3 else None

    if kind == "previous":
        action[..., :3] = np.asarray(previous_action)[..., :3]
    elif kind == "no_op":
        action[..., :3] = np.asarray([1, 2, 1], dtype=action.dtype)
    elif kind == "invert_maneuver":
        action[..., 0] = 2 - action[..., 0]
        action[..., 1] = 4 - action[..., 1]
        action[..., 2] = 2 - action[..., 2]
    elif kind == "invert_heading":
        action[..., 1] = 4 - action[..., 1]
    elif kind == "invert_altitude":
        action[..., 0] = 2 - action[..., 0]
    elif kind == "invert_velocity":
        action[..., 2] = 2 - action[..., 2]
    else:
        raise ValueError(f"unknown counterfactual action: {kind}")

    if shoot is not None:
        action[..., 3] = shoot
    return action


class OnlineAeroTAFDataset:
    """In-memory point dataset built from one MAPPO rollout."""

    def __init__(
        self,
        obs,
        actions,
        threat_targets,
        attack_targets,
        categories,
        segment_starts,
        history_windows,
        thresholds,
        event_counts,
        train_eligible,
        train_episode_count,
    ):
        self.obs = np.asarray(obs, dtype=np.float32)
        self.actions = np.asarray(actions, dtype=np.float32)
        self.threat_targets = np.asarray(threat_targets, dtype=np.float32)
        self.attack_targets = np.asarray(attack_targets, dtype=np.float32)
        self.categories = np.asarray(categories, dtype=np.int16)
        self.segment_starts = np.asarray(segment_starts, dtype=np.int32)
        self.history_windows = int(history_windows)
        self.thresholds = dict(thresholds)
        self.event_counts = dict(event_counts)
        self.train_eligible = np.asarray(train_eligible, dtype=bool)
        self.train_episode_count = int(train_episode_count)

        self.n_envs, self.time_steps, self.num_agents, self.obs_dim = self.obs.shape
        self.act_dim = self.actions.shape[-1]
        self.all_indices = np.arange(self.n_envs * self.time_steps, dtype=np.int64)

    def __len__(self):
        return int(self.all_indices.size)

    def unravel(self, flat_index):
        return divmod(int(flat_index), self.time_steps)

    def window_length(self, flat_index):
        env_i, t = self.unravel(flat_index)
        start = max(int(self.segment_starts[env_i, t]), t - self.history_windows + 1)
        return t - start + 1

    def window(self, flat_index, counterfactual_kind=None, counterfactual_agent=None):
        env_i, t = self.unravel(flat_index)
        segment_start = int(self.segment_starts[env_i, t])
        start = max(segment_start, t - self.history_windows + 1)
        obs_window = self.obs[env_i, start : t + 1]
        action_window = self.actions[env_i, start : t + 1]

        if counterfactual_kind is not None:
            action_window = action_window.copy()
            agent_i = int(counterfactual_agent)
            previous_t = t - 1 if t > segment_start else t
            action_window[-1, agent_i] = build_counterfactual_action(
                counterfactual_kind,
                self.actions[env_i, t, agent_i],
                self.actions[env_i, previous_t, agent_i],
            )
        return obs_window, action_window, start

    def targets(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        env_i = indices // self.time_steps
        time_i = indices % self.time_steps
        return (
            self.threat_targets[env_i, time_i].reshape(-1, 1),
            self.attack_targets[env_i, time_i].reshape(-1, 1),
            self.categories[env_i, time_i],
        )

    def sampled_training_indices(self, stable_ratio, seed):
        rng = np.random.default_rng(seed)
        flat_categories = self.categories.reshape(-1)
        eligible = self.train_eligible.reshape(-1)
        selected = [self.all_indices[eligible & (flat_categories != CATEGORY_STABLE)]]
        stable = self.all_indices[eligible & (flat_categories == CATEGORY_STABLE)]
        stable_count = int(round(len(stable) * float(stable_ratio)))
        if stable_count > 0:
            selected.append(rng.choice(stable, size=min(stable_count, len(stable)), replace=False))
        indices = np.concatenate(selected) if selected else np.asarray([], dtype=np.int64)
        rng.shuffle(indices)
        return indices

    def grouped_batches(self, indices, batch_size, seed=None):
        rng = np.random.default_rng(seed)
        indices = np.asarray(indices, dtype=np.int64)
        env_i = indices // self.time_steps
        time_i = indices % self.time_steps
        starts = np.maximum(
            self.segment_starts[env_i, time_i],
            time_i - self.history_windows + 1,
        )
        window_lengths = time_i - starts + 1
        unique_lengths = np.unique(window_lengths)
        rng.shuffle(unique_lengths)
        for seq_len in unique_lengths:
            group = indices[window_lengths == seq_len].copy()
            rng.shuffle(group)
            for start in range(0, len(group), int(batch_size)):
                yield group[start : start + int(batch_size)], int(seq_len)

    def stack_windows(self, indices, counterfactual_kind=None, counterfactual_agent=None):
        indices = np.asarray(indices, dtype=np.int64)
        if indices.size == 0:
            raise ValueError("cannot stack an empty AeroTAF window batch")
        env_i = indices // self.time_steps
        time_i = indices % self.time_steps
        segment_starts = self.segment_starts[env_i, time_i]
        starts = np.maximum(segment_starts, time_i - self.history_windows + 1)
        lengths = time_i - starts + 1
        if np.any(lengths != lengths[0]):
            raise ValueError("AeroTAF window batches must contain one sequence length")

        time_grid = starts[:, None] + np.arange(int(lengths[0]), dtype=np.int64)[None]
        obs = self.obs[env_i[:, None], time_grid]
        actions = self.actions[env_i[:, None], time_grid]
        if counterfactual_kind is not None:
            actions = actions.copy()
            agent_i = int(counterfactual_agent)
            previous_time = np.where(time_i > segment_starts, time_i - 1, time_i)
            actions[:, -1, agent_i] = build_counterfactual_action(
                counterfactual_kind,
                self.actions[env_i, time_i, agent_i],
                self.actions[env_i, previous_time, agent_i],
            )
        return obs, actions

    def category_counts(self, indices=None, eligible_only=False):
        if indices is None:
            categories = self.categories.reshape(-1)
            if eligible_only:
                categories = categories[self.train_eligible.reshape(-1)]
        else:
            categories = self.targets(indices)[2]
        return {
            name: int(np.sum(categories == category_id))
            for category_id, name in enumerate(CATEGORY_NAMES)
        }


class AeroTAFRolloutBuffer:
    """Collect snapshots and turn a completed rollout into an online detail dataset."""

    def __init__(self, buffer_size, n_envs):
        self.buffer_size = int(buffer_size)
        self.n_envs = int(n_envs)
        self.snapshots = [[] for _ in range(self.n_envs)]

    def insert(self, infos):
        if len(infos) != self.n_envs:
            raise ValueError(f"expected {self.n_envs} info rows, got {len(infos)}")
        for env_i, info in enumerate(infos):
            self.snapshots[env_i].append(info.get("AeroTAF_snapshot"))

    def clear(self):
        self.snapshots = [[] for _ in range(self.n_envs)]

    def build_dataset(self, shared_buffer, field_calculator, args):
        missing = [
            (env_i, t)
            for env_i, snapshots in enumerate(self.snapshots)
            for t, snapshot in enumerate(snapshots)
            if snapshot is None
        ]
        lengths = [len(items) for items in self.snapshots]
        if any(length != self.buffer_size for length in lengths):
            raise ValueError(f"AeroTAF snapshot lengths do not match rollout: {lengths}")
        if missing:
            raise RuntimeError(f"AeroTAF_snapshot missing at rollout positions: {missing[:10]}")

        threat_flat, attack_flat = field_calculator.build_targets(self.snapshots, shared_buffer)
        threat = threat_flat.reshape(self.n_envs, self.buffer_size)
        attack = attack_flat.reshape(self.n_envs, self.buffer_size)
        masks = shared_buffer.masks

        config = DetailAnnotationConfig(
            ego_team=args.AeroTAF_ego_team,
            high_threat_floor=args.AeroTAF_high_threat_floor,
            high_attack_floor=args.AeroTAF_high_attack_floor,
            high_field_percentile=args.AeroTAF_high_field_percentile,
            delta_floor=args.AeroTAF_delta_floor,
            delta_percentile=args.AeroTAF_delta_percentile,
        )

        segment_starts = np.zeros((self.n_envs, self.buffer_size), dtype=np.int32)
        items = []
        item_locations = []
        for env_i in range(self.n_envs):
            starts = [0]
            starts.extend(
                t
                for t in range(1, self.buffer_size)
                if np.all(masks[t, env_i] <= 0.0)
            )
            ends = starts[1:] + [self.buffer_size]
            for episode_ordinal, (start, end) in enumerate(zip(starts, ends)):
                segment_starts[env_i, start:end] = start
                items.append(
                    {
                        "threat_targets": threat[env_i, start:end, None],
                        "attack_targets": attack[env_i, start:end, None],
                        "snapshots": self.snapshots[env_i][start:end],
                    }
                )
                complete = end < self.buffer_size or np.all(masks[self.buffer_size, env_i] <= 0.0)
                item_locations.append((env_i, start, end, episode_ordinal, complete))

        complete_indices = [
            index
            for index, location in sorted(
                enumerate(item_locations),
                key=lambda pair: (pair[1][3], pair[1][0]),
            )
            if location[4]
        ]
        train_item_indices = complete_indices
        if not train_item_indices:
            raise RuntimeError(
                "the rollout contains no complete episode for AeroTAF training; "
                "increase --buffer-size or collect longer rollouts"
            )
        thresholds = fit_detail_thresholds([items[index] for index in train_item_indices], config)
        categories = np.full((self.n_envs, self.buffer_size), CATEGORY_STABLE, dtype=np.int16)
        train_eligible = np.zeros((self.n_envs, self.buffer_size), dtype=bool)
        event_counts = defaultdict(int)
        train_item_index_set = set(train_item_indices)
        for item_index, (item, location) in enumerate(zip(items, item_locations)):
            env_i, start, end = location[:3]
            annotate_episode_detail(item, thresholds, config)
            categories[env_i, start:end] = item["sample_category"]
            if item_index in train_item_index_set:
                train_eligible[env_i, start:end] = True
                for event_i, event_name in enumerate(EVENT_NAMES):
                    event_counts[str(event_name)] += int(item["event_flags"][:, event_i].sum())

        return OnlineAeroTAFDataset(
            obs=shared_buffer.obs[:-1].transpose(1, 0, 2, 3),
            actions=shared_buffer.actions.transpose(1, 0, 2, 3),
            threat_targets=threat,
            attack_targets=attack,
            categories=categories,
            segment_starts=segment_starts,
            history_windows=args.AeroTAF_history_windows,
            thresholds=thresholds,
            event_counts=event_counts,
            train_eligible=train_eligible,
            train_episode_count=len(train_item_indices),
        )
