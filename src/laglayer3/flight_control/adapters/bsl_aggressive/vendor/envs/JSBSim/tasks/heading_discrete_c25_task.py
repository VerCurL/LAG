"""Native discrete heading task with exact time-limit semantics for C25.

The control problem is intentionally inherited from :class:`HeadingTask`.
Only the episode boundary is disambiguated: route/safety failures are true
terminals, while the fixed episode horizon is an administrative truncation.
"""

from ..termination_conditions import Timeout
from .heading_task import HeadingTask


class HeadingDiscreteC25Task(HeadingTask):
    """Preserve native discrete control while exposing terminated/truncated."""

    def get_termination(self, env, agent_id, info=None):
        info = {} if info is None else info
        terminated = False
        truncated = False

        for condition in self.termination_conditions:
            done, _, info = condition.get_termination(
                self,
                env,
                agent_id,
                info,
            )
            if not done:
                continue
            if isinstance(condition, Timeout):
                truncated = True
            else:
                terminated = True
            break

        done = bool(terminated or truncated)
        if truncated:
            kind = "administrative_limit"
        elif terminated:
            kind = "task_or_safety_terminal"
        else:
            kind = "ongoing"
        info.update(
            {
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "TimeLimit.truncated": bool(truncated),
                "termination_kind": kind,
            }
        )
        return done, info
