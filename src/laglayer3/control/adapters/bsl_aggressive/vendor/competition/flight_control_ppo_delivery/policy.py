"""Original40D MLP/GRU architecture with mandatory new observation identity."""
from ...scripts.experiments.c46_compact_policy import CompactPolicy,make_compact_args
from ...competition.flight_control_ppo_delivery import OBSERVATION_SCHEMA,CONTROL_SCHEMA
from ...competition.flight_control_ppo_delivery.controller import FEATURE_NAMES

SCHEMA="platform-residual-ppo-40d-gru64-policy-v1"


class PlatformPolicy(CompactPolicy):
    def metadata(self):
        value=super().metadata()
        value.update(schema=SCHEMA,observation_schema=OBSERVATION_SCHEMA,observation_features=list(FEATURE_NAMES),
            control_schema=CONTROL_SCHEMA,legacy_checkpoint_compatible=False,
            changed_observation_indices=[7,31,34],external_execution="5Hz_QP_ZOH")
        return value

    def copy(self):return PlatformPolicy(self.architecture,self.seed,self.device,self.args)


def create_platform_policy(seed=5,device="cpu",args=None):
    return PlatformPolicy("A",seed,device=device,args=args)
