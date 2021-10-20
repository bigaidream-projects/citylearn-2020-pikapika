
from .ProbModels import GaussianMLP, GaussianHyperNetLinear

from .RLModules import BaseActor, SquashedGaussianHyperNetActor, SquashedGaussianMLPActor
from .RLModules import BaseQNet, BaseDualQNet, QNet, HyperQNet, DualQNet, DualHyperQNet

REGISTRY_ACTOR = dict(
    BaseActor=BaseActor,
    SquashedGaussianMLPActor=SquashedGaussianMLPActor,
    SquashedGaussianHyperNetActor=SquashedGaussianHyperNetActor
)

REGISTRY_CRITIC = dict(
    BaseQNet=BaseQNet,
    BaseDualQNet=BaseDualQNet,
    QNet=QNet,
    HyperQNet=HyperQNet,
    DualQNet=DualQNet,
    DualHyperQNet=DualHyperQNet
)

REGISTRY_PROB_MODEL = dict(
    GaussianMLP=GaussianMLP,
    GaussianHyperNetLinear=GaussianHyperNetLinear,
    SquashedGaussianMLPActor=SquashedGaussianMLPActor,
    SquashedGaussianHyperNetActor=SquashedGaussianHyperNetActor
)
