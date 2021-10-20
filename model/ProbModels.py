from torch import distributions, nn
from torch.nn import functional as F
from numpy import log

from model.BaseModules import MLP, HyperNetLinear


class GaussianModelMixin:
    LOG_STD_MAX = 2
    LOG_STD_MIN = -20
    Normal = distributions.Normal
    sample_activation = nn.Identity()

    model: nn.Module
    output_size: int
    distribution: distributions.distribution

    def forward(self, x=None, deterministic=False, require_logp=False, param_only=False, **kwargs):
        if x is not None:
            # Update Distribution
            mean, log_std = self.model(x, **kwargs).split(self.output_size, -1)
            std = log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX).exp()
            self.distribution = self.Normal(mean, std)
        else:
            assert self.distribution is not None
            mean, std = self.mean, self.stddev
        if param_only:
            return mean, std
        out = mean if deterministic else self.distribution.rsample()
        return (self.sample_activation(out), self.log_prob(out)) if require_logp \
            else self.sample_activation(out)

    def log_prob(self, value):
        assert self.distribution is not None
        return self.distribution.log_prob(value)

    def set_std_bound(self, lbound=None, ubound=None):
        if lbound is not None:
            self.LOG_STD_MIN = lbound
        if ubound is not None:
            self.LOG_STD_MAX = ubound

    @property
    def mean(self):
        return self.distribution.mean

    @property
    def stddev(self):
        return self.distribution.stddev

    @property
    def std(self):
        # Alias for self.stddev
        return self.stddev


class SquashedGaussianMixin(GaussianModelMixin):
    sample_activation = nn.Tanh()  # Note: Action is squashed to (-1, 1)

    def log_prob(self, value):
        # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
        # NOTE: The correction formula is a little bit magic. To get an understanding
        # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
        # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
        # Try deriving it yourself as a (very difficult) exercise. :)
        logp_pi = super(SquashedGaussianMixin, self).log_prob(value)
        logp_pi -= (2 * (log(2) - value - F.softplus(-2 * value)))
        return logp_pi.sum(-1)


class _GaussianModel(nn.Module):
    Normal = distributions.Normal
    sample_activation = nn.Identity()
    __ModelArch__: nn.Module

    def __init__(self, input_size, output_size, std_lbound=None, std_ubound=None, **kwargs):
        super(_GaussianModel, self).__init__()
        self.model = self.__ModelArch__(input_size, output_size * 2, **kwargs)
        self.input_size = input_size
        self.output_size = output_size
        self.distribution = None
        self.set_std_bound(std_lbound, std_ubound)

    def forward(self, *inputs):
        raise NotImplementedError


class GaussianMLP(GaussianModelMixin, _GaussianModel):
    __ModelArch__ = MLP


class GaussianHyperNetLinear(GaussianModelMixin, _GaussianModel):
    __ModelArch__ = HyperNetLinear
