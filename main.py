import os
import argparse
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from citylearn import CityLearn

from agent import RL_Agents
from utils.io import get_output_folder
from reward_function import reward_function
from utils.standardization import normalize_seq2seq_state_forRL, normalize_state

MAX_EPISODES = 300
sim_period = 8760

parser = argparse.ArgumentParser()
# RL Hyper-parameters
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--MAX_BUFFER', type=int, default=10000)
parser.add_argument('--seed', '-s', type=int, default=0)
parser.add_argument('--BATCH_SIZE', type=int, default=32)
parser.add_argument('--climate_zone', type=int, default=2)
parser.add_argument('--act_limit', type=float, default=0.5)
parser.add_argument('--decay', type=float, default=1)

# TCN Hyper-parameters
parser.add_argument('--encode_dim', type=int, default=64)
parser.add_argument('--alpha', type=float, default=0.2)

parser.add_argument('--kernel_size', type=int, default=3)
parser.add_argument('--kernel_size_forecast', type=int, default=3)

parser.add_argument('--levels', type=int, default=5)
parser.add_argument('--levels_forecast', type=int, default=3)

parser.add_argument('--seq_len', type=int, default=24)
parser.add_argument('--seq_len_forecast', type=int, default=6)


# reward Hyper-parameters
parser.add_argument('--A_r', type=float, default=0)
parser.add_argument('--B_r', type=float, default=1)
parser.add_argument('--window_len_A', type=int, default=6)
parser.add_argument('--window_len_B', type=int, default=12)
parser.add_argument('--price_factor', type=float, default=0.01)

# logger
parser.add_argument('--print_per_step', type=int, default=1000)
parser.add_argument('--writing_per_step', type=int, default=1000)
parser.add_argument('--start_k', type=int, default=0)
parser.add_argument('--start_c', type=int, default=0)

# load model
parser.add_argument('--continue_flag', type=int, default=0)
parser.add_argument('--load_episode', type=int, default=295)

args = parser.parse_args()

reward_kwargs = OrderedDict(
    alpha=args.A_r,
    beta=args.B_r,
    total_energy_window=args.window_len_A,
    heat_energy_window=args.window_len_B,
    price_factor=args.price_factor,
    ramping_window=6
)

full_dim = 33
src_dim = 21
pred_dim = 4
latent_dim = 3


def split_state_mixedTCN(norm_states):
    # (batch, building, seq, full_dim + src_dim + pred_dim)
    return norm_states[:, :, :, 0:full_dim], \
           norm_states[:, :, :, full_dim:full_dim + src_dim], \
           norm_states[:, :, -args.seq_len_forecast:, -pred_dim:]


def split_state_TCN(norm_states, future_len=6):
    src = norm_states[:, :, :, 0:src_dim]
    if future_len == 6:
        # (batch, building, seq, src_dim + 4)
        return src, norm_states[:, :, -args.seq_len_forecast:, -pred_dim:]
    elif future_len == 12:
        # (batch, building, seq, src_dim + 4(pred12) + 4(pred6))
        pred12hr = norm_states[:, :, -6:, -2*pred_dim:-pred_dim]
        pred6hr = norm_states[:, :, -6:, -pred_dim:]
        pred = torch.cat((pred6hr, pred12hr), -2)
        return src, pred


# Load Model using args dict
def get_agent_kwargs(args, log_path, eval=True):
    # Lazy Import
    from model.BaseModules import ActionClipping
    from model.Encoder import ROMAEncoder
    from model.RLModules import DualHyperQNet, SGHNActor, MLP

    encode_dim = args.encode_dim
    encoder_kwargs = (ROMAEncoder,
                      OrderedDict(  # for ROMAEncoder
                          input_size=33,
                          output_size=64,
                          role_size=latent_dim)
                      )

    model_kwargs = OrderedDict(
        Encoder=encoder_kwargs,
        DualCritic=(DualHyperQNet,
                    OrderedDict(input_size=encode_dim + 2, latent_size=latent_dim)),
        Actor=(SGHNActor,
               OrderedDict(input_size=encode_dim, output_size=2, latent_size=latent_dim)),
        DisparityNet=(MLP,
                      OrderedDict(input_size=3, output_size=1, layer_sizes=[64],
                                  norm_layer=nn.BatchNorm1d, activation=nn.LeakyReLU())),
    )

    default_lr = args.lr
    action_clip_kwargs = OrderedDict(
        start_bound=args.act_limit,
        stop_bound=.1,
        decay=args.decay,
        step_size=20000,
        warm_up=0,
        verbose=True
    )
    # print("act limit:", action_clip_kwargs['start_bound)

    if eval:
        algo_kwargs = None
        print("eval mode, use deterministic policy")
    else:
        unique_optim_kwargs = OrderedDict(
            Encoder=OrderedDict(),
            DualCritic=OrderedDict(),
            Actor=OrderedDict()
        )
        algo_kwargs = OrderedDict(
            batch_size=args.BATCH_SIZE,
            alpha=args.alpha,
            buffer_capacity=args.MAX_BUFFER,
            optim_cls=torch.optim.Adam,
            optim_kwargs=OrderedDict(lr=default_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0),
            unique_optim_kwargs=unique_optim_kwargs,
            verbose=True,
            log_interval=args.print_per_step,
            log_path=log_path
        )

    ac_kwargs = dict(
        model_kwargs=model_kwargs,
        algo_kwargs=algo_kwargs,
        # state_fn=lambda s: normalize_seq2seq_state_forRL(s, future_len=args.seq_len_forecast),
        state_fn=normalize_state,
        # No reward_fn for Hierarchical Agents # TODO Compatibility with original agent
        reward_fn=lambda rewards, states: reward_function(rewards, states, **reward_kwargs),
        action_clipping=lambda model: ActionClipping(model, **action_clip_kwargs),
        memory_size=args.seq_len
    )

    return ac_kwargs



def init_pretrain_models(rl_agents, seq2seq_path='Models_seq2seq_withAE_future'+str(args.seq_len_forecast),
                         autoencoder_path='Models_one_AE_128dim'):
    rl_agents.encoder.history_AE.load_state_dict(torch.load('pretrained_models/AE/{}_zone{}/AE.pt'.format(autoencoder_path, args.climate_zone)))
    rl_agents.encoder.history_AE.eval()
    rl_agents.encoder.history_AE.requires_grad_(False)

    rl_agents.encoder.auto_encoder.load_state_dict(torch.load('pretrained_models/AE_noSOC/{}_noSOC_zone{}/AE.pt'.format(autoencoder_path, args.climate_zone)))
    rl_agents.encoder.auto_encoder.eval()
    rl_agents.encoder.auto_encoder.requires_grad_(False)

    rl_agents.encoder.seq2seq.load_state_dict(torch.load('pretrained_models/seq2seq/{}_zone{}/Seq2SeqLSTM.pt'.format(seq2seq_path, args.climate_zone)))
    rl_agents.encoder.seq2seq.eval()
    rl_agents.encoder.seq2seq.requires_grad_(False)


if __name__ == "__main__":
    filename = "zone_" + str(args.climate_zone) + \
               "_lr" + str(args.lr) + \
               "_predLen_" + str(args.seq_len_forecast) + \
               "_windowlenB_" + str(args.window_len_B) + \
               "_actlimit" + str(args.act_limit) + \
               "_A_" + str(args.A_r) + \
               "_B_" + str(args.B_r) + \
               "_decay_" + str(args.decay)

    # Instantiating the Tensorboard writers
    PATH_base = 'datas/new/'
    PATH_base = get_output_folder(PATH_base, 'scalar_' + filename)
    PATH_to_log_dir1 = PATH_base + '/reward'
    writer = SummaryWriter(PATH_to_log_dir1)
    PATH_to_log_dir2 = PATH_base + '/cost'
    cost_writer = SummaryWriter(PATH_to_log_dir2)

    ac_kwargs = get_agent_kwargs(args, PATH_base, eval=False)

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

    # Provides information on Building type, Climate Zone, Annual DHW demand, Annual Cooling Demand,
    # Annual Electricity Demand, Solar Capacity, and correllations among buildings
    building_info = env.get_building_information()

    # Select many episodes for training. In the final run we will set this value to 1 (the buildings run for one year)
    start_episode = 0

    # RL CONTROLLER
    # Instantiating the control agent(s)
    rl_agent = RL_Agents(building_info, observations_spaces, actions_spaces, ac_kwargs)
    k, c = args.start_k, args.start_c
    cost = {}

    # initialize the model
    best_model_path = "./Models_best_zone" + str(args.climate_zone)

    # pretrained part
    # init_pretrain_models(rl_agent.agent)
    # print("load pretrain models successfully.")

    # non-pretrained part
    if args.continue_flag:
        rl_agent.agent.load_models(best_model_path, args.load_episode)
        print("load model from ", args.load_episode)

    model_path = './Models_' + filename
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    # print(rl_agent.agent.encoder)
    # The number of episodes can be replaces by a stopping criterion (i.e. convergence of the average reward)
    for e in range(0, MAX_EPISODES):
        cum_reward = {}
        for id in range(rl_agent.n_buildings):
            cum_reward[id] = 0
        state = env.reset()
        done = False
        while not done:
            if k % (40000 * 4) == 0:
                print('hour: ' + str(k) + ' of ' + str(sim_period * MAX_EPISODES))

            action = rl_agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            rl_agent.add_to_buffer(state, action, reward, next_state, done)
            state = next_state

            for id in range(rl_agent.n_buildings):
                cum_reward[id] += reward[id]

            if k % args.writing_per_step == 0:
                print("write rewards per " + str(args.writing_per_step) + "step")
                for id in range(rl_agent.n_buildings):
                    writer.add_scalar('building_reward_' + str(id), reward[id], k)
            k += 1

        # write episode-accumulated reward
        for r in range(rl_agent.n_buildings):
            writer.add_scalar('building_cum_reward_' + str(r), cum_reward[r], c)

        # write cost
        cost[e] = env.cost()
        print("cost_writer adding scalar")
        for i in range(len(objective_function)):
            cost_writer.add_scalar(str("cost_") + objective_function[i], cost[e][objective_function[i]], c)
            cost_writer.flush()

        # save models
        if c % 1 == 0:
            rl_agent.agent.save_models(model_path, e)
        c += 1
