def build_stage1_scenarios():
    return [
        {
            "scenario_id": "stage1_default_random_reset",
            "scenario_bucket": "default_random_reset",
            "fix_position": False,
            "description": "Use the environment's built-in random reset for stage-1 collection.",
        }
    ]
