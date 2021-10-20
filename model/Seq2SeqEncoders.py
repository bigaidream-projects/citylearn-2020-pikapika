import torch
from torch import nn

from model.BaseModules import TransformerDecoderLayer
from model.Encoder import BaseEncoder, AutoEncoder, TemporalConvNet


class Seq2SeqLSTM(BaseEncoder):
    def __init__(self, source_size, target_size, hidden_size, **kwargs):
        super(Seq2SeqLSTM, self).__init__()
        self.encoder = nn.LSTM(source_size, hidden_size, **kwargs)
        self.decoder = nn.LSTM(target_size, hidden_size, **kwargs)

    def forward(self, src, tgt, **kwargs):
        enc, enc_hx = self.encoder(src)
        dec, _ = self.decoder(tgt, hx=enc_hx)
        return enc, dec


class Seq2SeqLSTM_new(BaseEncoder):
    def __init__(self, source_size, target_size, pred_len, hidden_size, **kwargs):
        super(Seq2SeqLSTM_new, self).__init__()
        self.pred_len = pred_len
        self.encoder = nn.LSTM(source_size, hidden_size, **kwargs)
        self.decoder = nn.LSTM(target_size, hidden_size, **kwargs)

    def forward(self, src, **kwargs):
        """
        :param src: (Time, batch*building, State)
        :param kwargs:
        :return:
        """
        enc, enc_hx = self.encoder(src)
        h_x, _ = enc_hx
        dims = h_x.ndim - 1
        dec_in = h_x.repeat(self.pred_len, *([1] * dims))
        dec, _ = self.decoder(dec_in, hx=enc_hx)
        return enc, dec


class Seq2SeqAttnEncoder(BaseEncoder):
    def __init__(self, source_size, target_size, hidden_size, target_fn,
                 auto_encoder_kwargs, attn_kwargs, lstm_kwargs=None, **kwargs):
        super(Seq2SeqAttnEncoder, self).__init__()
        if lstm_kwargs is None:
            lstm_kwargs = {}
        self.target_fn = target_fn
        self.auto_encoder = AutoEncoder(source_size, hidden_size, **auto_encoder_kwargs)
        self.seq2seq = Seq2SeqLSTM(hidden_size, target_size, hidden_size, **lstm_kwargs)
        self.HistoryTemporalModule = TransformerDecoderLayer(hidden_size, **attn_kwargs)
        self.ForecastTemporalModule = TransformerDecoderLayer(hidden_size, **attn_kwargs)

    def forward(self, x):
        """

        :param x: the state sequence
        :return: hidden state

        Shape:
            - x: :math:`(Batch, Building, Time, State)`.
            - return: :math:`(Batch, Building, Hidden_State*2)`.
        """

        def to_seq_first(tensor):
            tensor = tensor.unsqueeze(0).transpose(0, -2)
            return tensor.reshape(tensor.size(0), -1, tensor.size(-1))

        def undo_seq_first(tensor, lead_dims):
            tensor.transpose_(0, -2)
            return tensor.reshape(*lead_dims, *tensor.shape[-2:])

        src, tgt = self.target_fn(x)
        assert src.shape[:-2] == tgt.shape[:-2]
        h_s = self.auto_encoder(src.reshape(-1, src.size(-1))).reshape(*src.shape[:-1], -1)

        h_s, tgt = to_seq_first(h_s), to_seq_first(tgt)
        _, h_t = self.seq2seq(src=h_s, tgt=tgt)
        h_cur = h_s[[-1]]

        h_t = self.ForecastTemporalModule(tgt=h_cur, memory=h_t)
        h_s = self.HistoryTemporalModule(tgt=h_cur, memory=h_s)
        out = undo_seq_first(torch.cat((h_s, h_t), dim=-1), src.shape[:-2]).squeeze(-2)

        # out.transpose_(0, 1)  # -> (Building, Batch, State)
        # out = self.BuildingAttnModule(out)
        # out.transpose_(0, 1)  # -> (Batch, Building, State)
        return out


class Seq2SeqTCNEncoder(BaseEncoder):
    def __init__(self, source_size, target_size, hidden_size, target_fn,
                 auto_encoder_kwargs, unique_kwargs_history, unique_kwargs_forecast, lstm_kwargs=None, **kwargs):
        super(Seq2SeqTCNEncoder, self).__init__()
        if lstm_kwargs is None:
            lstm_kwargs = {}
        self.target_fn = target_fn
        self.auto_encoder = AutoEncoder(source_size, hidden_size, **auto_encoder_kwargs)
        self.seq2seq = Seq2SeqLSTM(hidden_size, target_size, hidden_size, **lstm_kwargs)
        self.HistoryTemporalModule = TemporalConvNet(hidden_size, hidden_size, **unique_kwargs_history)
        self.ForecastTemporalModule = TemporalConvNet(hidden_size, hidden_size, **unique_kwargs_forecast)
        # self.BuildingAttnModule = TransformerEncoderLayer(hidden_size * 2, **attn_kwargs)

    def forward(self, x):
        """

        :param x: the state sequence
        :return: hidden state

        Shape:
            - x: :math:`(Batch, Building, Time, State)`.
            - return: :math:`(Batch, Building, Hidden_State*2)`.
        """

        def to_seq_first(tensor):
            tensor = tensor.unsqueeze(0).transpose(0, -2)
            return tensor.reshape(tensor.size(0), -1, tensor.size(-1))

        def undo_seq_first(tensor, lead_dims):
            tensor.transpose_(0, -2)
            return tensor.reshape(*lead_dims, *tensor.shape[-2:])

        def to_TCN_input(tensor):
            # tensor: (seq, batch*building, s_dim)
            tensor = tensor.transpose(1, 0)  # (batch*building, seq, s_dim)
            old_shape = tensor.shape
            return tensor.reshape((-1, 9, *old_shape[-2:]))

        def reverse_t_dim(tensor):
            inv_idx = torch.arange(tensor.size(2) - 1, -1, -1).long().to(tensor.device)
            # or equivalently torch.range(tensor.size(0)-1, 0, -1).long()
            inv_tensor = tensor.index_select(2, inv_idx)
            return inv_tensor

        src, tgt = self.target_fn(x)
        assert src.shape[:-2] == tgt.shape[:-2]
        h_s = self.auto_encoder(src.reshape(-1, src.size(-1))).reshape(*src.shape[:-1], -1)

        h_s, tgt = to_seq_first(h_s), to_seq_first(tgt)
        _, h_t = self.seq2seq(src=h_s, tgt=tgt)
        # h_cur = h_s[[-1]]

        # h_t = self.ForecastTemporalModule(tgt=h_cur, memory=h_t)
        # h_s = self.HistoryTemporalModule(tgt=h_cur, memory=h_s)

        # (seq, batch*building, s_dim)
        # TCN input: (batch, building, seq, s_dim)
        h_t = self.ForecastTemporalModule(reverse_t_dim(to_TCN_input(h_t)))  # (batch, building, 128)
        h_s = self.HistoryTemporalModule(to_TCN_input(h_s))  # reverse the forecast sequence on t-dim

        out = torch.cat((h_s, h_t), dim=-1)  # (batch, building, 256)

        # out.transpose_(0, 1)  # -> (Building, Batch, State)
        # out = self.BuildingAttnModule(out)
        # out.transpose_(0, 1)  # -> (Batch, Building, State)
        return out


class Seq2SeqSymTCNEncoder_old(BaseEncoder):
    def __init__(self, source_size, target_size, hidden_size, target_fn, pred_len,
                 auto_encoder_kwargs, tcn_kwargs, lstm_kwargs=None, **kwargs):
        super(Seq2SeqSymTCNEncoder_old, self).__init__()
        if lstm_kwargs is None:
            lstm_kwargs = {}
        self.pred_len = pred_len
        self.target_fn = target_fn
        self.auto_encoder = AutoEncoder(source_size, hidden_size, **auto_encoder_kwargs)
        self.seq2seq = Seq2SeqLSTM(hidden_size, target_size, hidden_size, **lstm_kwargs)
        self.TemporalModule = TemporalConvNet(hidden_size, hidden_size, **tcn_kwargs)
        # self.BuildingAttnModule = TransformerEncoderLayer(hidden_size * 2, **attn_kwargs)

    def forward(self, x):
        """
        :param x: the state sequence
        :return: hidden state

        Shape:
            - x: :math:`(Batch, Building, Time, State)`.
            - return: :math:`(Batch, Building, Hidden_State*2)`.
        """

        def to_seq_first(tensor):
            tensor = tensor.unsqueeze(0).transpose(0, -2)
            return tensor.reshape(tensor.size(0), -1, tensor.size(-1))

        def undo_seq_first(tensor, lead_dims):
            tensor.transpose_(0, -2)
            return tensor.reshape(*lead_dims, *tensor.shape[-2:])

        def to_TCN_input(tensor):
            # tensor: (seq, batch*building, s_dim)
            tensor = tensor.transpose(1, 0)  # (batch*building, seq, s_dim)
            old_shape = tensor.shape
            return tensor.reshape((-1, 9, *old_shape[-2:]))

        src, tgt = self.target_fn(x)
        assert src.shape[:-2] == tgt.shape[:-2]
        h_s = self.auto_encoder(src.reshape(-1, src.size(-1))).reshape(*src.shape[:-1], -1)

        h_s, tgt = to_seq_first(h_s), to_seq_first(tgt)
        _, h_t = self.seq2seq(src=h_s, tgt=tgt)
        # h_cur = h_s[[-1]]
        h = torch.cat((h_s, h_t), dim=0)

        # (seq, batch*building, s_dim)
        # TCN input: (batch, building, seq, s_dim)
        out = self.TemporalModule((to_TCN_input(h))).unbind(-2)[-1]  # (batch, building, seq, 128)

        # out.transpose_(0, 1)  # -> (Building, Batch, State)
        # out = self.BuildingAttnModule(out)
        # out.transpose_(0, 1)  # -> (Batch, Building, State)
        return out[:, :, -(self.pred_len + 1)]


class Seq2SeqMixedTCNEncoder(BaseEncoder):
    def __init__(self, source_size, target_size, hidden_size, target_fn,
                 auto_encoder_kwargs, unique_kwargs_history, unique_kwargs_forecast, lstm_kwargs=None, **kwargs):
        super(Seq2SeqMixedTCNEncoder, self).__init__()
        if lstm_kwargs is None:
            lstm_kwargs = {}
        self.target_fn = target_fn
        self.auto_encoder = AutoEncoder(21, hidden_size, **auto_encoder_kwargs).eval()
        self.seq2seq = Seq2SeqLSTM(hidden_size, target_size, hidden_size, **lstm_kwargs).eval()
        self.HistoryTemporalModule = TemporalConvNet(source_size, hidden_size, **unique_kwargs_history)
        self.ForecastTemporalModule = TemporalConvNet(hidden_size, hidden_size, **unique_kwargs_forecast)
        # self.BuildingAttnModule = TransformerEncoderLayer(hidden_size * 2, **attn_kwargs)

    def forward(self, x):
        """

        :param x: the state sequence
        :return: hidden state

        Shape:
            - x: :math:`(Batch, Building, Time, State)`.
            - return: :math:`(Batch, Building, Hidden_State*2)`.
        """

        def to_seq_first(tensor):
            tensor = tensor.unsqueeze(0).transpose(0, -2)
            return tensor.reshape(tensor.size(0), -1, tensor.size(-1))

        def undo_seq_first(tensor, lead_dims):
            tensor.transpose_(0, -2)
            return tensor.reshape(*lead_dims, *tensor.shape[-2:])

        def to_TCN_input(tensor):
            # tensor: (seq, batch*building, s_dim)
            tensor = tensor.transpose(1, 0)  # (batch*building, seq, s_dim)
            old_shape = tensor.shape
            return tensor.reshape((-1, 9, *old_shape[-2:]))

        def reverse_t_dim(tensor):
            inv_idx = torch.arange(tensor.size(2) - 1, -1, -1).long().to(tensor.device)
            # or equivalently torch.range(tensor.size(0)-1, 0, -1).long()
            inv_tensor = tensor.index_select(2, inv_idx)
            return inv_tensor

        src_full, src, tgt = self.target_fn(x)
        assert src_full.shape[:-2] == src.shape[:-2] == tgt.shape[:-2]
        h_s = self.auto_encoder(src.reshape(-1, src.size(-1))).reshape(*src.shape[:-1], -1)
        h_s, tgt = to_seq_first(h_s), to_seq_first(tgt)
        _, h_t = self.seq2seq(src=h_s, tgt=tgt)

        # (seq, batch*building, s_dim)
        # TCN input: (batch, building, seq, s_dim)
        # reverse the forecast sequence on t-dim
        h_t = self.ForecastTemporalModule(reverse_t_dim(to_TCN_input(h_t))).unbind(-2)[-1]
        h_s = self.HistoryTemporalModule(src_full).unbind(-2)[-1]  # (batch, building, 128)

        out = torch.cat((h_s, h_t), dim=-1)  # (batch, building, 256)

        # out.transpose_(0, 1)  # -> (Building, Batch, State)
        # out = self.BuildingAttnModule(out)
        # out.transpose_(0, 1)  # -> (Batch, Building, State)
        return out


class Seq2SeqSymTCNEncoder(BaseEncoder):
    def __init__(self, source_size, target_size, hidden_size, target_fn, pred_len,
                 auto_encoder_kwargs, tcn_kwargs, lstm_kwargs=None, **kwargs):
        super(Seq2SeqSymTCNEncoder, self).__init__()
        if lstm_kwargs is None:
            lstm_kwargs = {}
        self.pred_len = pred_len
        self.target_fn = target_fn
        self.history_AE = AutoEncoder(source_size, hidden_size, **auto_encoder_kwargs)  # src_size = 21
        self.auto_encoder = AutoEncoder(source_size - 2, hidden_size, **auto_encoder_kwargs)  # src_size = 19
        self.seq2seq = Seq2SeqLSTM(hidden_size, target_size, hidden_size, **lstm_kwargs)
        self.TemporalModule = TemporalConvNet(hidden_size, hidden_size, **tcn_kwargs)
        # self.BuildingAttnModule = TransformerEncoderLayer(hidden_size * 2, **attn_kwargs)

    def forward(self, x):
        """

        :param x: the state sequence
        :return: hidden state

        Shape:
            - x: :math:`(Batch, Building, Time, State)`.
            - return: :math:`(Batch, Building, Hidden_State*2)`.
        """

        def to_seq_first(tensor):
            tensor = tensor.unsqueeze(0).transpose(0, -2)
            return tensor.reshape(tensor.size(0), -1, tensor.size(-1))

        def undo_seq_first(tensor, lead_dims):
            tensor.transpose_(0, -2)
            return tensor.reshape(*lead_dims, *tensor.shape[-2:])

        def to_TCN_input(tensor):
            # tensor: (seq, batch*building, s_dim)
            tensor = tensor.transpose(1, 0)  # (batch*building, seq, s_dim)
            old_shape = tensor.shape
            return tensor.reshape((-1, 9, *old_shape[-2:]))

        src, tgt = self.target_fn(x)
        src_noSOC = src[:, :, :, :-2]  # discard soc states
        assert src.shape[:-2] == tgt.shape[:-2]
        # generate hidden states of history seq
        h_s_out = self.history_AE(src.reshape(-1, src.size(-1))).reshape(*src.shape[:-1], -1)
        h_s_out = to_seq_first(h_s_out)

        # generate hidden states of forecast seq
        h_s = self.auto_encoder(src_noSOC.reshape(-1, src_noSOC.size(-1))).reshape(*src_noSOC.shape[:-1], -1)
        h_s, tgt = to_seq_first(h_s), to_seq_first(tgt)
        _, h_t_out = self.seq2seq(src=h_s, tgt=tgt)

        h = torch.cat((h_s_out, h_t_out), dim=0)

        # (seq, batch*building, s_dim)
        # TCN input: (batch, building, seq, s_dim)
        out = self.TemporalModule((to_TCN_input(h))).unbind(-2)[-1]  # (batch, building, seq, 128)

        # out.transpose_(0, 1)  # -> (Building, Batch, State)
        # out = self.BuildingAttnModule(out)
        # out.transpose_(0, 1)  # -> (Batch, Building, State)
        return out[:, :, -(self.pred_len + 1)]


class Seq2SeqSymTCNEncoder_new(BaseEncoder):
    def __init__(self, enc_src_size, history_src_size, hidden_size, pred_len,
                 auto_encoder_kwargs, tcn_kwargs, lstm_kwargs=None, **kwargs):
        super(Seq2SeqSymTCNEncoder_new, self).__init__()
        if lstm_kwargs is None:
            lstm_kwargs = {}
        self.pred_len = pred_len
        self.history_AE = AutoEncoder(history_src_size, hidden_size, **auto_encoder_kwargs)  # src_size = 21
        self.auto_encoder = AutoEncoder(enc_src_size, hidden_size, **auto_encoder_kwargs)  # src_size = 31
        self.seq2seq = Seq2SeqLSTM_new(source_size=hidden_size, target_size=hidden_size,
                                       pred_len=pred_len, hidden_size=hidden_size, **lstm_kwargs)
        self.TemporalModule = TemporalConvNet(hidden_size, hidden_size, **tcn_kwargs)
        # self.BuildingAttnModule = TransformerEncoderLayer(hidden_size * 2, **attn_kwargs)

    def forward(self, x):
        """

        :param x: the state sequence
        :return: hidden state

        Shape:
            - x: :math:`(Batch, Building, Time, State)`.
            - return: :math:`(Batch, Building, Hidden_State)`.
        """
        def extract_history(states):
            """
            :param states: dim=33->21
            :return:
            """
            result_list = [
                states[:, :, :, 0:10],
                states[:, :, :, 10:11],
                states[:, :, :, 14:15],
                states[:, :, :, 18:19],
                states[:, :, :, 22:23],
                states[:, :, :, 26:33],
            ]
            return torch.cat(result_list, -1)

        def to_seq_first(tensor):
            tensor = tensor.unsqueeze(0).transpose(0, -2)
            return tensor.reshape(tensor.size(0), -1, tensor.size(-1))

        def undo_seq_first(tensor, lead_dims):
            tensor.transpose_(0, -2)
            return tensor.reshape(*lead_dims, *tensor.shape[-2:])

        def to_TCN_input(tensor):
            # tensor: (seq, batch*building, s_dim)
            tensor = tensor.transpose(1, 0)  # (batch*building, seq, s_dim)
            old_shape = tensor.shape
            return tensor.reshape((-1, 9, *old_shape[-2:]))

        # x dim = 33
        src = extract_history(x)  # src dim=21
        enc_in = x[:, :, :, :-2]  # encoder input: dim = 31

        # generate hidden states of history seq
        h_s_out = self.history_AE(src.reshape(-1, src.size(-1))).reshape(*src.shape[:-1], -1)
        h_s_out = to_seq_first(h_s_out)

        # generate hidden states of forecast seq
        h_s = self.auto_encoder(enc_in.reshape(-1, enc_in.size(-1))).reshape(*enc_in.shape[:-1], -1)
        h_s = to_seq_first(h_s)
        _, h_t_out = self.seq2seq(src=h_s)

        h = torch.cat((h_s_out, h_t_out), dim=0)

        # (seq, batch*building, s_dim)
        # TCN input: (batch, building, seq, s_dim)
        out = self.TemporalModule(to_TCN_input(h))  # (batch, building, seq, 128)

        # out.transpose_(0, 1)  # -> (Building, Batch, State)
        # out = self.BuildingAttnModule(out)
        # out.transpose_(0, 1)  # -> (Batch, Building, State)
        return out[:, :, -(self.pred_len + 1)]