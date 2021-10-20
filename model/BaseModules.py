import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules import transformer


class MLP(nn.Module):
    r"""General Multi-Layer Perceptron based on PyTorch, with batch_norm feature.

    Args:
        input_size: dimension of input (layer)
        output_size: dimension of output (layer)
        layer_sizes: number of neurons in each hidden layer, int is perceived as single layer. default=None
        norm_layer: whether to use normalization layer
        norm_kwargs: keyword arguments related to norm_layer.
        activation: specify non-linear activation function, applied to all hidden layers
        out_activation: activation for output
    """

    def __init__(self, input_size, output_size, layer_sizes=None,
                 norm_layer=None, norm_kwargs=None,
                 activation=nn.ReLU(inplace=True),
                 out_activation=None):
        super(MLP, self).__init__()
        if layer_sizes is None:
            layer_sizes = []
        if norm_kwargs is None:
            norm_kwargs = {}
        if not hasattr(layer_sizes, "__iter__"):
            layer_sizes = [layer_sizes]

        layer_sizes = [input_size] + list(layer_sizes) + [output_size]
        if norm_layer:
            def block(layer_size, next_layer_size):
                return [nn.Linear(layer_size, next_layer_size),
                        norm_layer(next_layer_size, **norm_kwargs),
                        activation]
        else:
            def block(layer_size, next_layer_size):
                return [nn.Linear(layer_size, next_layer_size),
                        activation]
        modules = []
        for idx in range(0, len(layer_sizes) - 2):
            modules.extend(block(layer_sizes[idx], layer_sizes[idx + 1]))
        modules.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        if layer_sizes is None and norm_layer is not None:
            modules.append(norm_layer(output_size, **norm_kwargs))
        if out_activation is not None:
            modules.extend([out_activation])
        self.mlp_model = nn.Sequential(*modules)

    def forward(self, x, **kwargs):
        r"""Defines the computation performed at every call.
        Can be overridden by subclasses to add customize functions.
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.
        """
        return self.mlp_model(x.reshape(-1, x.size(-1))).reshape(*x.shape[:-1], -1)


class HyperNetLinear(nn.Module):
    def __init__(self, input_size, output_size, latent_size, bias=True, **kwargs):
        super(HyperNetLinear, self).__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.weight_lin = nn.Linear(latent_size, output_size * input_size)
        self.weight_lin.register_forward_hook(
            lambda m, inputs, output:
            output.reshape(*output.shape[:-1], input_size, output_size)
        )
        if bias:
            self.bias_lin = nn.Linear(latent_size, output_size)
            self.bias_lin.register_forward_hook(
                lambda m, inputs, output: output.unsqueeze(-2)
            )
        else:
            self.bias_lin = lambda latent: torch.tensor(0., dtype=latent.dtype, device=latent.device)

    def forward(self, x, latent, **kwargs):
        out = self._forward(x.reshape(-1, x.size(-1)), latent.reshape(-1, latent.size(-1)))
        return out.reshape(*x.shape[:-1], -1)

    def _forward(self, x, latent):
        weight = self.weight_lin(latent)
        bias = self.bias_lin(latent)
        out = torch.baddbmm(bias, x.unsqueeze(-2), weight).squeeze(-2)
        return out


class TransformerEncoderLayer(transformer.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__(d_model, nhead, dim_feedforward, dropout, activation)
        self._reset_parameters()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.
        Unlike original implementation in PyTorch, we use PreNorm method.
        A.K.A. src = src + Dropout(Self_Attn(self.LayerNorm(src)))

        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.norm1(src)
        src2 = self.self_attn(src2, src2, src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        if hasattr(self, "activation"):
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        else:  # for backward compatibility
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class TransformerDecoderLayer(transformer.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerDecoderLayer, self).__init__(d_model, nhead, dim_feedforward, dropout, activation)
        self._reset_parameters()

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        r"""Pass the inputs (and mask) through the decoder layer.
        Note: This implementation omitted the self-attn layer!

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt2 = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt2, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class TemporalConvBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation,
                 padding, dropout, time_len, downsample=None, norm_layer=None):
        super(TemporalConvBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        if downsample is None and in_channels != out_channels:
            downsample = nn.Conv1d(in_channels, out_channels, 1)

        self.padding = padding
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn1 = norm_layer(time_len)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn2 = norm_layer(time_len)
        self.dropout = nn.Dropout(dropout)
        self.bn3 = norm_layer(time_len)
        self.downsample = downsample
        self.init_params()

    def init_params(self):
        # Append weight norm after initialization.
        for module in [self.conv1, self.conv2]:
            if isinstance(module, nn.Conv1d):
                try:
                    nn.utils.weight_norm(module)
                except RuntimeError:
                    pass

    def forward(self, x):
        identity = x

        out = self.conv1(x.transpose(-2, -1)).transpose(-2, -1)
        out = self._remove_padding(out)
        out = self.bn1(out)
        out = self.dropout(self.relu(out))

        out = self.conv2(out.transpose(-2, -1)).transpose(-2, -1)
        out = self._remove_padding(out)
        out = self.bn2(out)
        out = self.dropout(self.relu(out))

        if self.downsample is not None:
            identity = self.downsample(identity.transpose(-2, -1)).transpose(-2, -1)

        out = self.bn3(identity + out)
        out = self.relu(out)

        return out

    def _remove_padding(self, out):
        out = out[:, :-self.padding]  # Remove info from the "future".
        return out


class SymTemporalConvBlock(TemporalConvBlock):
    def _remove_padding(self, out):
        assert self.padding % 2 == 0
        return out[:, self.padding // 2: - self.padding // 2]  # Remove info symmetrically


class ActionClipping:
    def __init__(self, model, start_bound, stop_bound, decay, step_size, warm_up=0, verbose=False):
        assert start_bound > stop_bound > 0.
        assert 0. < decay <= 1.

        self.model: nn.Module = model
        self.time_step = -warm_up
        self.warm_up = warm_up
        self.step_size = step_size
        self.current_factor = start_bound
        self.start_bound = start_bound
        self.stop_bound = stop_bound
        self.decay = decay
        self.verbose = verbose
        self.hook_handle = None

    def add_clipping(self):
        def clipping_hook(module, inputs, output):
            if isinstance(output, torch.Tensor):
                return output * self.current_factor
            elif isinstance(output[0], torch.Tensor):
                output = list(output)
                output[0] = output[0] * self.current_factor
                return output
            elif isinstance(output, list):  # TODO
                for tensor in output:
                    tensor.mul_(self.current_factor)
                return output
            RuntimeWarning("Unknown output structure! Will NOT perform Action Clipping!")

        self.hook_handle = self.model.register_forward_hook(clipping_hook)

    def remove_clipping(self):
        self.hook_handle.remove()

    def step(self):
        self.time_step += 1
        if self.time_step % self.step_size == 0:
            next_factor = max(self.current_factor * self.decay, self.stop_bound)
            if next_factor != self.current_factor:
                self.current_factor = next_factor
            if self.verbose:
                print("Current Limit Updated to {:.5f} @ Step {}.".format(self.current_factor, self.time_step))

    def reset(self):
        self.time_step = -self.warm_up
        self.current_factor = self.start_bound
