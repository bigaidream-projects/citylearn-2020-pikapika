import torch
import torch.nn as nn

from .BaseModules import MLP, HyperNetLinear
from model.ProbModels import SquashedGaussianMixin, _GaussianModel

DEFAULT_QNET_NORM = nn.LayerNorm


class BaseQNet(nn.Module):
    __ModelArch__: nn.Module

    def __init__(self, input_size, output_size, **kwargs):
        """

        :param input_size: int Dimension of state
        :param output_size: int Dimension of action
        :param kwargs: For other keyword argument, refer to the specific __ModelArch__
        """
        super(BaseQNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.model = self.__ModelArch__(input_size+output_size, 1, **kwargs)

    def forward(self, obs_act, **kwargs):
        # TODO: Remove Squeeze?
        return self.model(obs_act, **kwargs).squeeze(-1)


class BaseDualQNet(nn.Module):
    __ModelArch__: nn.Module

    def __init__(self, input_size, output_size, **kwargs):
        r"""

        :param input_size: int Dimension of state
        :param output_size: int Dimension of action
        :param kwargs: For other keyword argument, refer to the specific __ModelArch__
        """
        super(BaseDualQNet, self).__init__()
        self.QNet1 = self.__ModelArch__(input_size, output_size, **kwargs)
        self.QNet2 = self.__ModelArch__(input_size, output_size, **kwargs)

    def forward(self, obs_act, **kwargs):
        return self.QNet1(obs_act, **kwargs), self.QNet2(obs_act, **kwargs)


class BaseActor(nn.Module):
    __ModelArch__: nn.Module

    def __init__(self, input_size, output_size, **kwargs):
        r"""

        :param input_size: int Dimension of state
        :param output_size: int Dimension of action
        :param kwargs: For other keyword argument, refer to the specific __ModelArch__
        """
        super(BaseActor, self).__init__()
        self.model = self.__ModelArch__(input_size, output_size, **kwargs)
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, state, **kwargs):
        return self.model(state, **kwargs)


class QNet(BaseQNet):
    __ModelArch__ = MLP

    def __init__(self, input_size, output_size, norm_layer=DEFAULT_QNET_NORM, **kwargs):
        r"""

        :param input_size: int Dimension of state
        :param output_size: int Dimension of action
        :param norm_layer: Type of Normalization Layer
        :param kwargs: For other kwargs, ref to __ModelArch__
        """
        super().__init__(input_size, output_size, norm_layer=norm_layer, **kwargs)


class HyperQNet(BaseQNet):
    __ModelArch__ = HyperNetLinear

    def __init__(self, input_size, output_size, latent_size, **kwargs):
        """

        :param input_size: int Dimension of state
        :param output_size: int Dimension of action
        :param latent_size: int Dimension of latent, used to generate weight dynamically.
        :param kwargs: For other kwargs, ref to __ModelArch__
        """
        super(HyperQNet, self).__init__(input_size, output_size, latent_size=latent_size, **kwargs)
        self.latent_size = latent_size

    def forward(self, obs_act, latent=None, **kwargs):
        """

        :param obs_act: FloatTensor (*, State+Action) or (*, State+Latent+Action) if latent is None
        :param latent: FloatTensor (*, Latent), will infer from obs_act by default
        :param kwargs: other keyword args
        :return: FloatTensor (*, )  # TODO Remove Squeeze
        """
        if latent is None:
            latent, obs_act = self.split_latent(obs_act)
        return super(HyperQNet, self).forward(obs_act, latent=latent, **kwargs)

    def split_latent(self, obs_act):
        obs, latent, act = obs_act.split([self.input_size, self.latent_size, self.output_size], dim=-1)
        obs_act = torch.cat((obs, act), dim=-1)
        return latent, obs_act


class DualQNet(BaseDualQNet):
    __ModelArch__ = QNet

    def __init__(self, input_size, output_size, norm_layer=DEFAULT_QNET_NORM, **kwargs):
        r"""

        :param input_size: int Dimension of state
        :param output_size: int Dimension of action
        :param norm_layer: Type of Normalization Layer
        :param kwargs: For other keyword argument, refer to the specific __ModelArch__
        """
        super(DualQNet, self).__init__(input_size, output_size, norm_layer=norm_layer, **kwargs)


class DualHyperQNet(BaseDualQNet):
    __ModelArch__ = HyperQNet

    def __init__(self, input_size, output_size, latent_size, **kwargs):
        """

        :param input_size: int Dimension of state
        :param output_size: int Dimension of action
        :param latent_size: int Dimension of latent, used to generate weight dynamically.
        :param kwargs: For other kwargs, ref to __ModelArch__
        """
        super(DualHyperQNet, self).__init__(input_size, output_size, latent_size=latent_size, **kwargs)


class SquashedGaussianMLPActor(SquashedGaussianMixin, _GaussianModel):
    __ModelArch__ = MLP

    def sample_action(self, obs=None, **kwargs):
        return self(obs, deterministic=False, require_logp=True, **kwargs)


class SquashedGaussianHyperNetActor(SquashedGaussianMixin, _GaussianModel):
    __ModelArch__ = HyperNetLinear

    def __init__(self, input_size, output_size, latent_size, **kwargs):
        super(SquashedGaussianHyperNetActor, self).__init__(input_size, output_size, latent_size=latent_size, **kwargs)
        self.latent_size = latent_size

    def forward(self, state=None, latent=None, **kwargs):
        """

        :param state: FloatTensor (*, State) or (*, State+Latent) if latent is None
        :param latent: FloatTensor (*, Latent), will infer from obs_act by default
        :param kwargs: other keyword args
        :return: FloatTensor (*, Action)
        """
        if latent is None and state is not None:
            state, latent = state.split([self.input_size, self.latent_size], dim=-1)
        return super(SquashedGaussianHyperNetActor, self).forward(state, latent=latent, **kwargs)

    def sample_action(self, state=None, latent=None, **kwargs):
        return self(state, latent=latent, deterministic=False, require_logp=True, **kwargs)


SGMLPActor = SquashedGaussianMLPActor
SGHNActor = SquashedGaussianHyperNetActor
