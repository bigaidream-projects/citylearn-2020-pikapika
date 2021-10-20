from torch.optim import Adam
from torch.nn.functional import l1_loss
from pathlib import Path
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from citylearn import CityLearn
from copy import deepcopy
from utils.standardization import normalize_seq2seq_state
from utils.io import get_output_folder
from model.Encoder import AE, Seq2SeqLSTM
# from model.Seq2SeqEncoders import Seq2SeqLSTM
from model.Decoder import LinearDecoder
from utils.util import USE_CUDA
import argparse
import os


def get_one_state_seq(x, y, t):
    """
    :param x: (9, 8760, state_dim) -- memory array
    :param y: (9, 8760, pred_state_dim) -- pred state array
    :param t: start time -- T-24
    :return: selected: (batch, 9, 24+6, state_dim)
    """
    # x = torch.randn(9, 8760, 27)
    # batch_size = 32
    slice_size = 24 + 6
    # batch_idx = np.random.randint(low=0, high=batch_size, size=batch_size)
    # batch_idx = torch.LongTensor(batch_idx).unsqueeze(-1)  # (batch_size, 1)
    # slice_idx = torch.arange(slice_size).unsqueeze(0)  # (1, 30)
    slice_idx = torch.LongTensor([idx for idx in range(t, t+30)])

    # select [T-24:T+6] state sequence, [T:T+6] is used for supervision
    indexes = slice_idx.unsqueeze(0)  # (1, 30)
    selected = x.index_select(-2, indexes.flatten()).split(slice_size, -2)
    selected = torch.stack(selected)

    # select [T-6:T] pred6hr sequence
    slice_size = 6
    indexes = slice_idx[-12:-6].unsqueeze(0)  # (batch_size, 6) -- (T-6, T)'s pred6hr seq
    selected_y = y.index_select(-2, indexes.flatten()).split(slice_size, -2)
    selected_y = torch.stack(selected_y)
    return deepcopy(selected), deepcopy(selected_y)


def print_grad(net):
    for name, parms in net.named_parameters():
        if parms.grad is None:
            continue
        print('-->name:', name, '-->grad_requires:', parms.requires_grad,
              ' -->grad_value:', torch.max(parms.grad), torch.min(parms.grad))


log_per_step = 1000

parser = argparse.ArgumentParser()
# RL Hyper-parameters
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--MAX_EPOCH', type=int, default=500)
parser.add_argument('--seed', '-s', type=int, default=0)
parser.add_argument('--BATCH_SIZE', type=int, default=100)
parser.add_argument('--climate_zone', type=int, default=4)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--load_model', type=bool, default=True)
parser.add_argument('--past_len', type=int, default=24)
parser.add_argument('--future_len', type=int, default=6)
args = parser.parse_args()

assert args.future_len <= args.past_len
filename = "_h_dim{}_past_{}_future_{}_lr_{}_zone{}".format(args.hidden_dim, args.past_len, args.future_len, args.lr, args.climate_zone)

# load data
data_path = Path("../data/Climate_Zone_" + str(args.climate_zone))
building_attributes = data_path / 'building_attributes.json'
weather_file = data_path / 'weather_data.csv'
solar_profile = data_path / 'solar_generation_1kW.csv'
building_state_actions = 'buildings_state_action_space.json'
building_ids = ["Building_1", "Building_2", "Building_3", "Building_4", "Building_5", "Building_6", "Building_7",
                "Building_8", "Building_9"]
objective_function = ['ramping', '1-load_factor', 'average_daily_peak', 'peak_demand',
                      'net_electricity_consumption', 'total']

# Instantiating the env
env = CityLearn(data_path, building_attributes, weather_file, solar_profile, building_ids,
                buildings_states_actions=building_state_actions, cost_function=objective_function)
observations_spaces, actions_spaces = env.get_state_action_spaces()

# Instantiating the Tensorboard writers
PATH_base = 'datas/new/'
PATH_base = get_output_folder(PATH_base, 'scalar_seq2seq' + filename)
PATH_to_log_dir1 = PATH_base + '/loss'
writer = SummaryWriter(PATH_to_log_dir1)


def add_to_memory(memory, arr, t):
    memory[:, t, :] = deepcopy(arr)


def extract_pred_state(state):
    """
    :param state: (seq, batch, 19)
    :return: (seq, batch, 19-8=11)
    """
    return state[:, :, 8:]


# Memory Array slice operation
def get_state_seq(x, y, y2, batch_size, future_len=6):
    """
    :param x, y, y2: (9, 8760, state_dim) -- memory array
    :return selected: (batch, 9, past_len+future_len, 21)
            out: (batch, 9, future_len, 4)
    """
    # x = torch.randn(9, 8760, 27)
    # batch_size = 32
    slice_size = args.past_len + args.future_len
    batch_idx = np.random.randint(low=0, high=batch_size, size=batch_size)
    batch_idx = torch.LongTensor(batch_idx).unsqueeze(-1)  # (batch_size, 1)
    slice_idx = torch.arange(slice_size).unsqueeze(0)  # (1, past+future)

    # select [T-past:T+future] state sequence, [T:T+future] is used for supervision
    indexes = batch_idx + slice_idx  # (batch_size, 30)
    selected = x.index_select(-2, indexes.flatten()).split(slice_size, -2)
    selected = torch.stack(selected)

    indexes = batch_idx + slice_idx[:, -(6 + future_len):-future_len]  # (batch_size, 6) -- (T-6, T)'s pred6hr seq

    selected_y = y.index_select(-2, indexes.flatten()).split(6, -2)

    if future_len == 6:
        # select [T-6:T]'s pred6hr sequence
        out = torch.stack(selected_y)

    elif future_len == 12:
        # select [T-6:T]'s pred6hr + pred12hr sequence
        selected_y2 = y2.index_select(-2, indexes.flatten()).split(6, -2)  # pred12hr
        out = torch.cat((torch.stack(selected_y), torch.stack(selected_y2)), -2)

    return deepcopy(selected), deepcopy(out)


def print_grad(net):
    for name, parms in net.named_parameters():
        if parms.grad is None:
            continue
        print('-->name:', name, '-->grad_requires:', parms.requires_grad,
              ' -->grad_value:', torch.max(parms.grad), torch.min(parms.grad))


state_dim = 19
pred_state_dim = 4
predictable_dim = 11

# Initialize Memory Array
Memory = np.zeros((9, 8760, state_dim))
Pred6hr_Memory = np.zeros((9, 8760, pred_state_dim))
Pred12hr_Memory = np.zeros((9, 8760, pred_state_dim))

# ------------------- Interact with env -----------------------
state = env.reset()
src, tgt_tuple = normalize_seq2seq_state(state, future_len=args.future_len, pretrain=True)

add_to_memory(Memory, src, 0)

if args.future_len == 6:
    add_to_memory(Pred6hr_Memory, tgt_tuple, 0)
elif args.future_len == 12:
    tgt_6hr, tgt_12hr = tgt_tuple
    add_to_memory(Pred6hr_Memory, tgt_6hr, 0)
    add_to_memory(Pred12hr_Memory, tgt_12hr, 0)

done = False
t = 0
while not done:
    t += 1
    action = np.random.uniform(low=-1., high=1., size=(9, 2))  # random action
    next_state, reward, done, _ = env.step(action)
    src, tgt_tuple = normalize_seq2seq_state(next_state, future_len=args.future_len, pretrain=True)
    add_to_memory(Memory, src, t)

    if args.future_len == 6:
        add_to_memory(Pred6hr_Memory, tgt_tuple, t)
    elif args.future_len == 12:
        tgt_6hr, tgt_12hr = tgt_tuple
        add_to_memory(Pred6hr_Memory, tgt_6hr, t)
        add_to_memory(Pred12hr_Memory, tgt_12hr, t)

    state = next_state

# ------------------- Initialize models -------------------

# AE model
auto_encoder = AE(19, 128, [128, 128], {}).eval()
auto_encoder.requires_grad_(False)
auto_encoder.load_state_dict(torch.load('./pretrained_models/AE_noSOC/Models_one_AE_128dim_noSOC_zone{}/AE.pt'.format(str(args.climate_zone))))
print("load AE weights successfully")

# seq2seq model
model = Seq2SeqLSTM(args.hidden_dim, pred_state_dim, args.hidden_dim)  # TODO: clarify the dim in args
linear_decoder = LinearDecoder(predictable_dim, args.hidden_dim)
model.eval()
model.requires_grad_(False)
linear_decoder.eval()
linear_decoder.requires_grad_(False)

if USE_CUDA:
    auto_encoder = auto_encoder.cuda()
    model = model.cuda()
    linear_decoder = linear_decoder.cuda()

model_path = './pretrained_models/seq2seq/Models_seq2seq_withAE_future6_zone' + str(args.climate_zone)
if not os.path.isdir(model_path):
    os.mkdir(model_path)

if args.load_model:
    model.load_state_dict(torch.load('{}/Seq2SeqLSTM.pt'.format(model_path)))
    linear_decoder.load_state_dict(torch.load('{}/linear_decoder.pt'.format(model_path)))
    print("load model successfully")


# Initialize optimizer
params = [
    dict(params=model.parameters()),
    dict(params=linear_decoder.parameters())
]

opt = Adam(params, lr=0.001)
max_epoch = args.MAX_EPOCH
MIN_loss = 9999999
STEP_PER_EPOCH = 10000
BATCH_SIZE = args.BATCH_SIZE
# DROPOUT = 0.2

Memory = torch.FloatTensor(Memory)
Pred6hr_Memory = torch.FloatTensor(Pred6hr_Memory)
Pred12hr_Memory = torch.FloatTensor(Pred12hr_Memory)

for e in range(max_epoch):
    cum_loss = 0.
    for idx in range(STEP_PER_EPOCH):
        full_seq, tgt = get_state_seq(Memory, Pred6hr_Memory, Pred12hr_Memory, BATCH_SIZE, future_len=args.future_len)
        if USE_CUDA:
            full_seq = full_seq.cuda()
            tgt = tgt.cuda()

        src = full_seq[:, :, 0:args.past_len, :].reshape(-1, args.past_len, state_dim)  # (batch*9, seq=24, state_dim=21)
        src = auto_encoder(src).transpose(1, 0)  # (seq=24, batch*9, hidden_dim=128)

        tgt = tgt.reshape(-1, args.future_len, pred_state_dim).transpose(1, 0)

        real_tgt = extract_pred_state(full_seq[:, :, -args.future_len:, :].reshape(-1, args.future_len, state_dim)).transpose(1, 0)

        _, dec = model(src, tgt)
        recon_tgt = linear_decoder(dec)

        ReconstructionLoss_tgt = l1_loss(recon_tgt, real_tgt, reduction='mean')
        loss = ReconstructionLoss_tgt

        # opt.zero_grad()
        # loss.backward()
        # opt.step()

        cum_loss += loss.detach().cpu()

        # if (e * STEP_PER_EPOCH + idx) % log_per_step == 0:
        #     # print(recon_s, pred_s)
        #     print("loss {} at step {}".format(loss, e * STEP_PER_EPOCH + idx))
        #     print_grad(model)
        #     writer.add_scalar('loss_step', loss, e * STEP_PER_EPOCH + idx)

    print("cum loss {} at epoch {}".format(cum_loss, e))
    # if cum_loss < MIN_loss:
    #     MIN_loss = cum_loss
    #     if e > 0:
    #         torch.save(model.state_dict(), '{}/Seq2SeqLSTM.pt'.format(model_path))
    #         torch.save(linear_decoder.state_dict(), '{}/linear_decoder.pt'.format(model_path))
    #         print("save model in epoch {}".format(e))

    writer.add_scalar('loss_epoch', cum_loss, e)



