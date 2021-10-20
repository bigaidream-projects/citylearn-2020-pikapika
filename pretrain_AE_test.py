from torch.optim import Adam
from torch.nn.functional import l1_loss
from torch.distributions import kl_divergence, Normal
from pathlib import Path
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from citylearn import CityLearn

from utils.standardization import normalize_AE_state
from utils.io import get_output_folder
from model.Encoder import AE
from utils.util import USE_CUDA
import os

log_per_step = 1000

# Instantiating the Tensorboard writers
PATH_base = 'datas/new/'
PATH_base = get_output_folder(PATH_base, 'scalar_pretrain_encoder')
PATH_to_log_dir1 = PATH_base + '/pred'
pred_writer = SummaryWriter(PATH_to_log_dir1)
PATH_to_log_dir2 = PATH_base + '/unpred'
unpred_writer = SummaryWriter(PATH_to_log_dir2)
climate_zone = 4

# load data
data_path = Path("../data/Climate_Zone_" + str(climate_zone))
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


# test_sample = torch.zeros((100, 37))
# dataloader = [test_sample]

state = env.reset()

norm_state = normalize_AE_state(state, noSOC=True)
dataloader = [norm_state]

done = False
while not done:
    action = np.zeros((9, 2))
    next_state, reward, done, _ = env.step(action)
    norm_state = normalize_AE_state(next_state, noSOC=True)
    dataloader.append(norm_state)
    state = next_state


model = AE(19, 128, [128, 128], {})

if USE_CUDA:
    model = model.cuda()

opt = Adam(model.parameters(), lr=0.001)
max_epoch = 100
MIN_loss = 9999999
test_climate_zone = 4
model_path = './pretrained_models/AE_noSOC/Models_one_AE_128dim_noSOC_zone{}'.format(str(test_climate_zone))
if not os.path.isdir(model_path):
    os.mkdir(model_path)

# init AE model
model.load_state_dict(torch.load('{}/AE.pt'.format(model_path)))
model.eval()
model.requires_grad_(False)
print("load model successfully")


def print_grad(net):
    for name, parms in net.named_parameters():
        if parms.grad is None:
            continue
        print('-->name:', name, '-->grad_requires:', parms.requires_grad,
              ' -->grad_value:', torch.max(parms.grad), torch.min(parms.grad))


STEP_PER_EPOCH = 10000
BATCH_SIZE = 100
DROPOUT = 0.2

for e in range(max_epoch):
    cum_loss = 0.
    for idx in range(STEP_PER_EPOCH):
        batch_idx = np.random.randint(low=0, high=8760, size=BATCH_SIZE)
        s = torch.FloatTensor(np.array(dataloader)[batch_idx]).reshape(BATCH_SIZE * 9, -1)
        if USE_CUDA:
            s = s.cuda()

        # =========== training VAE1 for predictable variables =========

        hidden_state = model(s)
        # GaussianDist = Normal(torch.zeros_like(dist.mean), torch.ones_like(dist.stddev))  # Gaussian(0, 1)
        # TODO Check gradient flow through kl_divergence
        recon_s = model.decode(hidden_state)

        # <input - output> pair-wise dropout
        mask = torch.ones_like(s)
        mask = torch.nn.Dropout(0.2)(mask)
        mask[mask != 0] = 1.
        recon_s = recon_s * mask
        s = s * mask

        ReconstructionLoss = l1_loss(recon_s, s, reduction='mean')
        loss = ReconstructionLoss

        # opt.zero_grad()
        # loss.backward()
        # opt.step()

        cum_loss += loss.detach().cpu()

        # if (e * STEP_PER_EPOCH + idx) % log_per_step == 0:
        #     # print(recon_s, pred_s)
        #     print("loss {} at step {}".format(loss, e * STEP_PER_EPOCH + idx))
        #     print_grad(model)
        #     pred_writer.add_scalar('pred_loss_step', loss, e * STEP_PER_EPOCH + idx)

    # print("cum loss {} at epoch {}".format(cum_loss, e))
    # if cum_loss < MIN_loss:
    #     MIN_loss = cum_loss
    #     if e > 0:
    #         torch.save(model.state_dict(), '{}/AE.pt'.format(model_path))
    #         print("save pred model in epoch {}".format(e))

    pred_writer.add_scalar('loss_epoch', cum_loss, e)


