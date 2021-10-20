import math

import torch
from torch import nn

from model.BaseModules import TransformerEncoderLayer, TransformerDecoderLayer
from model.BaseModules import TemporalConvBlock
from model.BaseModules import MLP
from model.ProbModels import GaussianMLP


class BaseEncoder(nn.Module):
    def __init__(self):
        super(BaseEncoder, self).__init__()

    def forward(self, *inputs):
        raise NotImplementedError

    def register_hooks(self):
        pass


class AutoEncoder(BaseEncoder):
    encoder_arch = MLP
    decoder_arch = MLP
    encoder: MLP
    decoder: MLP

    def __init__(self, input_size, output_size, layer_sizes, unique_kwargs=None, **kwargs):
        super(AutoEncoder, self).__init__()
        if unique_kwargs is None:
            unique_kwargs = {}
        encoder_kwargs = unique_kwargs.get('Encoder_Setting', {})
        decoder_kwargs = unique_kwargs.get('Decoder_Setting', {})
        self.encoder = self.encoder_arch(input_size, output_size, layer_sizes, **encoder_kwargs, **kwargs)
        self.decoder = self.decoder_arch(output_size, input_size, layer_sizes[::-1], **decoder_kwargs, **kwargs)

    def forward(self, x, **kwargs):
        return self.encode(x, **kwargs)

    def encode(self, x, **kwargs):
        out = self.encoder(x, **kwargs)
        return out

    def decode(self, hidden_state, **kwargs):
        return self.decoder(hidden_state, **kwargs)


class VariationalAutoEncoder(AutoEncoder):
    encoder_arch = GaussianMLP
    encoder: GaussianMLP

    def encode(self, x, return_dist=False, **kwargs):
        out = self.encoder(x, **kwargs)
        if return_dist:
            return out, self.encoder.distribution
        return out


AE = AutoEncoder
VAE = VariationalAutoEncoder


class TemporalConvNet(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout,
                 stride=1, time_len=24, norm_layer=None, block_arch=TemporalConvBlock):
        super(TemporalConvNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer
        self.stride = stride
        self.kernel_size = kernel_size
        self.dropout = dropout
        layers = [nn.Linear(input_size, output_size)]
        num_levels = len(num_channels)
        num_channels += [output_size]
        for i in range(num_levels):
            dilation = 2 ** i
            layers.append(block_arch(num_channels[i], num_channels[i + 1], kernel_size, stride=stride,
                                     dilation=dilation, padding=(kernel_size - 1) * dilation,
                                     dropout=dropout, time_len=time_len))
        self.layers = nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        """
        :param x: Float32 Tensor with shape = (*, Time_Channels, State)
        :return: FLoat32 Tensor with shape = (*, Time_Channels, State)
        """
        old_shape = x.shape
        out = x.reshape(-1, *old_shape[-2:])
        out = self.layers(out)
        out = out.reshape(*old_shape[:-2], *out.shape[-2:])
        return out


TCN = TCNEncoder = TemporalConvNet


class TCBAEncoder(BaseEncoder):
    def __init__(self, input_size, output_size, tcn_kwargs, attn_kwargs, **kwargs):
        super(TCBAEncoder, self).__init__()
        self.TempConvModule = TemporalConvNet(input_size, output_size, **tcn_kwargs)
        self.SelfAttentionModule = TransformerEncoderLayer(output_size, **attn_kwargs)
        self.register_hooks()

    def forward(self, x, **kwargs):
        out = self.TempConvModule(x)
        out = self.SelfAttentionModule(out)
        return out

    def register_hooks(self):
        def attn_pre_fwd_hook(module, inputs):
            inputs = list(inputs)
            inputs[0] = inputs[0].permute(2, 1, 0, 3)[-1]
            return inputs

        def attn_fwd_hook(module, inputs, output):
            return output.transpose(0, 1)

        self.SelfAttentionModule.register_forward_pre_hook(attn_pre_fwd_hook)
        self.SelfAttentionModule.register_forward_hook(attn_fwd_hook)
        super(TCBAEncoder, self).register_hooks()


class BATCEncoder(TCBAEncoder):
    def __init__(self, input_size, output_size, tcn_kwargs, attn_kwargs, **kwargs):
        super(BATCEncoder, self).__init__(output_size, output_size, tcn_kwargs, attn_kwargs, **kwargs)
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x, **kwargs):
        # x.shape = Batch, Building, Time, State
        out = self.linear(x.transpose(0, 1))
        seq_len, other_dims, state_dim = out.shape[0], out.shape[1:-1], out.shape[-1]
        out = out.reshape(seq_len, -1, state_dim)
        out = self.SelfAttentionModule(out)
        out = out.reshape(seq_len, *other_dims, state_dim).transpose(0, 1)
        out = self.TempConvModule(out)
        return out


class HRLEncoder(BaseEncoder):
    def __init__(self, input_size, output_size, tcn_kwargs, att_kwargs, **kwargs):
        super(HRLEncoder, self).__init__()
        self.GlobalTCNModule = TemporalConvNet(input_size, output_size, **tcn_kwargs)
        self.Hour2DayModule = TransformerDecoderLayer(output_size, **att_kwargs)
        self.Day2WeekModule = TransformerDecoderLayer(output_size, **att_kwargs)
        self.linear_in = nn.Linear(input_size, output_size)
        self.linear_out = nn.Linear(output_size * 3, output_size)

    def forward(self, x, **kwargs):
        """
        :param x: (batch, building, time=168, state=37)
        :return:
        """
        time_length = 12
        assert x.shape[-2] % time_length == 0
        local_state = self.linear_in(x[:, :, -1])  # (batch, building, state)
        hour_states = self.GlobalTCNModule(x)
        hour_chunks = torch.stack(hour_states.split(time_length, -2))  # (Chunks, Batch, Building, Time_Length, State)

        # <---------------- Hour2Day Attention Module --------------------->
        # Use the last time & last chunk for output state
        # -> (1, Batch, Building, Chunks, State)
        daily_states = self._temporal_abstraction(hour_chunks, self.Hour2DayModule)
        day_state = daily_states[-1, :, :, -1]

        # <----------------- Day2Week Attention Module --------------------->
        # -> (1, Batch, Building, 1, State)
        weekly_states = self._temporal_abstraction(daily_states, self.Day2WeekModule)
        week_state = weekly_states[-1, :, :, -1]

        # <---------------------- output layer ----------------------------->
        out = self.linear_out(torch.cat((local_state, day_state, week_state), -1))
        return out

    @staticmethod
    def _temporal_abstraction(x, attn_module):
        """

        :param x: FloatTensor with shape = (Chunks, Batch, Building, Time_Length, State)
        :param attn_module: attention module
        :return: FloatTensor with shape = (1, Batch, Building, Chunks, State)
        """
        x = x.transpose(0, -2)  # (Time_Length, Batch, Building, Chunks, State)
        old_shape = x.shape[1:-1]
        src = x.reshape(x.size(0), -1, x.size(-1))
        tgt = src[-1:]
        out = attn_module(tgt, src)
        out = out.reshape(-1, *old_shape, out.size(-1))
        return out


class ROMAEncoder(BaseEncoder):
    # Add kwargs if needed
    __RNNArch__ = nn.GRU

    def __init__(self, input_size, output_size, role_size, embed_kwargs=None, infer_kwargs=None, rnn_kwarg=None,
                 **other_kwargs):
        super(ROMAEncoder, self).__init__()
        if embed_kwargs is None:
            embed_kwargs = {}
        if infer_kwargs is None:
            infer_kwargs = {}
        if rnn_kwarg is None:
            rnn_kwarg = {}

        default = dict(
            norm_layer=nn.BatchNorm1d,
            activation=nn.LeakyReLU(inplace=True),
            out_activation=nn.LeakyReLU(inplace=True),
            std_lbound=math.log(2e-3),
            std_ubound=+math.inf
        )

        for k in default:
            embed_kwargs.setdefault(k, default[k])
            infer_kwargs.setdefault(k, default[k])

        self.RoleEncoder = GaussianMLP(input_size, role_size, **embed_kwargs)
        self.PosteriorEstimator = GaussianMLP(input_size + output_size, role_size, **infer_kwargs)
        self.LinearEncoder = MLP(input_size, output_size, out_activation=nn.ReLU(inplace=True))
        self.RNNUnit = self.__RNNArch__(output_size, output_size, **rnn_kwarg)

        self._reset_rnn_params()

    def _reset_rnn_params(self):
        for p in self.RNNUnit.parameters():
            if p.ndim > 1:
                torch.nn.init.orthogonal_(p)

    def forward(self, x, h_in=None, train_mode=False, deterministic=False, **other_kwargs):
        """

        :param x: (*, Time, Input_Size)
        :param h_in: (1, *, Output_Size), default=None, if None, use zero vector instead.
        :param train_mode: See return for details.
        :param deterministic: Whether to sample from distribution
        :param other_kwargs:
        :return: (output, h_out, info)
            output: (*, Time, Output_Size)
            h_out: (1, **, Output_Size)
            info: If train_mode is False, info = role_embed
                               otherwise, info = (role_embed, role_infer, dist_embed, dist_infer)
            role_embed, role_infer: (*, Time, role_size)
        """
        def to_seq_first(tensor):
            tensor = tensor.unsqueeze(0).transpose(0, -2)
            return tensor.reshape(tensor.size(0), -1, tensor.size(-1))

        def undo_seq_first(tensor, lead_dims):
            tensor = tensor.transpose(0, -2)
            return tensor.reshape(*lead_dims, *tensor.shape[-2:])

        lead_dim = x.shape[:-2]
        role_embed = self.RoleEncoder(x, deterministic=deterministic)
        dist_embed = self.RoleEncoder.distribution

        h_s = self.LinearEncoder(to_seq_first(x))
        h_s, h_out = self.RNNUnit(h_s, h_in)
        out = undo_seq_first(h_s, lead_dim)
        if train_mode:
            if h_in is None:
                h_init = torch.zeros_like(h_s[-1], device=h_s.device).reshape(1, -1, h_s.size(-1))
            else:
                h_init = self.get_hidden_init(h_in)
            h_infer = torch.cat((h_init, h_s[:-1]))
            h_infer = undo_seq_first(h_infer, lead_dim)
            role_infer = self.PosteriorEstimator(torch.cat((x, h_infer), -1), deterministic=deterministic)
            dist_infer = self.PosteriorEstimator.distribution
            info = (role_embed, role_infer, dist_embed, dist_infer)
        else:
            info = role_embed

        return out, h_out, info

    def get_hidden_init(self, h_in):
        time_idx = -2 if self.RNNUnit.bidirectional else -1
        if isinstance(self.RNNUnit, nn.LSTM):
            h_in = h_in[0]
        return h_in[time_idx].unsqueeze(0)


class ROMALSTMEncoder(ROMAEncoder):
    __RNNArch__ = nn.LSTM

