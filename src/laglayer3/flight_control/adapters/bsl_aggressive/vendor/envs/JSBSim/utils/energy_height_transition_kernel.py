"""Isolated, opt-in transition from energy balance to explicit height cost.

This module does not change the frozen energy/momentum kernel.  The original
kernel remains responsible for input validation, geometry, physical errors and
normalization.  All states and references must describe the same post-action
instant.  Terminal rewards, safety checks and task admission gates stay outside
this pure running-reward calculation.
"""

from .baiyepower_v2_physics import rational_cost
from .energy_momentum_kernel import compute_energy_momentum_reward
from .flight_path_geometry import _finite_scalar, _positive_scalar


def compute_energy_height_transition_reward(
    altitude_m,
    speed_mps,
    target_altitude_m,
    target_speed_mps,
    direction,
    reference_direction,
    actual_direction_rate,
    reference_direction_rate,
    dt,
    config,
    *,
    transition_eta,
):
    """Return the original diagnostics plus an explicit height-transition cost.

    ``CE = .5*rho(eT/S) + .5*((1-eta)*rho(eB/S) + eta*rho(eh/S))``

    Here ``S`` is exactly the original kernel's ``S_T_m == S_B_m``.  In
    particular, this is *not* the configuration's standalone height scale.
    ``eta`` must be a finite real scalar in [0, 1], including at the endpoints;
    booleans and silent clipping are rejected.  It is required explicitly so
    callers cannot accidentally select a treatment through a default value.

    The original energy/momentum weights and dt integration are retained.  At
    eta=0, every original return field is preserved exactly.  At eta>0, ``C_E``,
    ``weighted_cost`` and ``running_reward`` describe the active candidate.
    ``C_energy_balance`` still records the actual original balance cost, while
    ``C_allocation_transition`` records the active blend.  ``original_*`` fields
    retain the control reward for offline comparisons; they are diagnostics,
    never an extra reward.  No speed dead zone, stage weighting or bonus is
    introduced here.
    """
    eta = _finite_scalar(transition_eta, "transition_eta")
    if not 0.0 <= eta <= 1.0:
        raise ValueError("transition_eta must lie in [0, 1]")
    result = compute_energy_momentum_reward(
        altitude_m,
        speed_mps,
        target_altitude_m,
        target_speed_mps,
        direction,
        reference_direction,
        actual_direction_rate,
        reference_direction_rate,
        dt,
        config,
    )
    original_energy = result["C_E"]
    original_weighted = result["weighted_cost"]
    original_reward = result["running_reward"]
    scale = result["S_T_m"]
    height_cost = rational_cost(result["e_h_m"] / scale)
    balance_cost = result["C_energy_balance"]

    # These branches preserve eta=0 bit-for-bit and avoid roundoff changes in
    # the pure-height slice, where e_k=0 and the two allocation costs coincide.
    allocation_cost = balance_cost
    if eta != 0.0 and height_cost != balance_cost:
        allocation_cost = _finite_scalar(
            (1.0 - eta) * balance_cost + eta * height_cost,
            "transition allocation cost",
        )
        result["C_E"] = _finite_scalar(
            0.5 * result["C_total_energy"] + 0.5 * allocation_cost,
            "transition energy cost",
        )
        result["weighted_cost"] = _finite_scalar(
            config.energy_weight * result["C_E"]
            + config.momentum_weight * result["C_M"],
            "transition weighted cost",
        )
        denominator = _positive_scalar(
            1.0 + result["weighted_cost"], "transition reward denominator",
        )
        result["running_reward"] = _finite_scalar(
            result["dt"] / denominator, "transition running_reward",
        )

    result.update(
        transition_eta=eta,
        original_C_E=original_energy,
        original_weighted_cost=original_weighted,
        original_running_reward=original_reward,
        C_height=height_cost,
        S_height_transition_m=scale,
        C_allocation_transition=allocation_cost,
    )
    return result
