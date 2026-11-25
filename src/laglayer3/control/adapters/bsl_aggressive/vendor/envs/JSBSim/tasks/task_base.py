import numpy as np
from gymnasium import spaces
from typing import List, Tuple
from abc import ABC, abstractmethod
from ..core.catalog import Catalog as c


class BaseTask(ABC):
    """
    Base Task class.
    A class to subclass in order to create a task with its own observation variables,
    action variables, termination conditions and reward functions.
    """
    def __init__(self, config):
        self.config = config
        self.reward_functions = []
        self.termination_conditions = []
        self.load_variables()
        self.load_observation_space()
        self.load_action_space()

    @property
    def num_agents(self):
        return 1

    @abstractmethod
    def load_variables(self):
        self.state_var = [
            c.position_long_gc_deg,
            c.position_lat_geod_deg,
            c.position_h_sl_m,
        ]
        self.action_var = [
            c.fcs_aileron_cmd_norm,
            c.fcs_elevator_cmd_norm,
            c.fcs_rudder_cmd_norm,
            c.fcs_throttle_cmd_norm,
        ]

    @abstractmethod
    def load_observation_space(self):
        """
        Load observation space
        """
        self.observation_space = spaces.Discrete(5)

    @abstractmethod
    def load_action_space(self):
        """
        Load action space
        """
        self.action_space = spaces.Discrete(5)

    def reset(self, env):
        """Task-specific reset

        Args:
            env: environment instance
        """
        for reward_function in self.reward_functions:
            reward_function.reset(self, env)

    def step(self, env):
        """ Task-specific step

        Args:
            env: environment instance
        """
        pass

    def begin_transition(self, env):
        """Capture task-specific state before the current action is applied.

        The default implementation is deliberately empty so existing tasks
        retain their exact transition semantics.  Opt-in tasks may use this
        hook to freeze targets or pre-action diagnostics.
        """
        return None

    def get_transition_diagnostics(self, env, agent_id):
        """Return diagnostics produced while evaluating the current transition."""
        return {}

    def validate_post_transition(self, env):
        """Validate post-action state before optional diagnostic samplers run."""
        return None

    def has_transition_numerical_failure(self, agent_id):
        """Whether the current transition must avoid ordinary state samplers."""
        return False

    def get_terminal_observation_fallback(self, agent_id):
        """Return a last finite observation for a numerical true terminal."""
        return None

    def get_terminal_reward(self, env, agent_id, info):
        """Return an optional one-shot terminal contribution."""
        return 0.0

    def end_transition(self, env, info):
        """Finalize task-specific transition state after outcome resolution."""
        return None

    def get_reward(self, env, agent_id, info={}) -> Tuple[float, dict]:
        """
        Aggregate reward functions

        Args:
            env: environment instance
            agent_id: current agent id
            info: additional info

        Returns:
            (tuple):
                reward(float): total reward of the current timestep
                info(dict): additional info
        """
        reward = 0.0
        for reward_function in self.reward_functions:
            reward += reward_function.get_reward(self, env, agent_id)
        return reward, info

    def get_termination(self, env, agent_id, info={}) -> Tuple[bool, dict]:
        """
        Aggregate termination conditions

        Args:
            env: environment instance
            agent_id: current agent id
            info: additional info

        Returns:
            (tuple):
                done(bool): whether the episode has terminated
                info(dict): additional info
        """
        done = False
        success = True
        for condition in self.termination_conditions:
            d, s, info = condition.get_termination(self, env, agent_id, info)
            done = done or d
            success = success and s
            if done:
                break
        return done, info

    def get_obs(self, env, agent_id):
        """Extract useful informations from environment for specific agent_id.
        """
        return np.zeros(2)

    def normalize_action(self, env, agent_id, action):
        """Normalize action to be consistent with action space.
        """
        return np.array(action)

    def sample_substep_action(self, env, agent_id, substep_index):
        """Optionally provide a simulator-rate command for one substep.

        Most tasks hold the decision-rate command written by ``BaseEnv``.
        Controllers that explicitly smooth at simulator rate can override
        this hook and return a four-channel JSBSim command.  ``None`` keeps
        the existing command, preserving all legacy task behavior.
        """
        return None
