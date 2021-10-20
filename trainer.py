# -*- coding:utf-8 -*-
import argparse
from collections import OrderedDict
from pathlib import Path
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from citylearn import CityLearn

from agent import RL_Agents
from utils.io import get_output_folder
from utils.standardization import normalize_state
from reward_function import reward_function

parser = argparse.ArgumentParser()

# RL Hyper-parameters
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--MAX_BUFFER', type=int, default=10000)
parser.add_argument('--seed', '-s', type=int, default=0)
parser.add_argument('--BATCH_SIZE', type=int, default=32)
parser.add_argument('--climate_zone', type=int, default=1)
parser.add_argument('--act_limit', type=float, default=0.5)
parser.add_argument('--decay', type=float, default=1)

# TCN Hyper-parameters
parser.add_argument('--encode_dim', type=int, default=64)
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--alpha', type=float, default=0.2)

parser.add_argument('--kernel_size', type=int, default=3)
parser.add_argument('--kernel_size_forecast', type=int, default=3)

parser.add_argument('--levels', type=int, default=5)
parser.add_argument('--levels_forecast', type=int, default=3)

parser.add_argument('--seq_len', type=int, default=24)
parser.add_argument('--seq_len_forecast', type=int, default=6)

parser.add_argument('--episode_len', type=int, default=168)

# reward Hyper-parameters
parser.add_argument('--A_r', type=float, default=0)
parser.add_argument('--B_r', type=float, default=1)
parser.add_argument('--window_len_A', type=int, default=6)
parser.add_argument('--window_len_B', type=int, default=12)
parser.add_argument('--price_factor', type=float, default=0.01)

# logger
parser.add_argument('--print_per_step', type=int, default=250)
parser.add_argument('--start_k', type=int, default=0)
parser.add_argument('--start_c', type=int, default=0)

# load model
parser.add_argument('--continue_flag', type=int, default=0)
parser.add_argument('--load_episode', type=int, default=295)

# training length
parser.add_argument('--MaxEpisode', type=int, default=100000)
parser.add_argument('--save_per_episode', type=int, default=500)
parser.add_argument('--train', dest='train', default=True, action='store_true')
parser.add_argument('--eval', dest='train', action='store_false')
parser.add_argument('--suffix', type=str, default="")

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
act_dim = 2


# Load Model using args dict
def get_agent_kwargs(args, log_path, train=None):
    # Lazy Import
    from model.BaseModules import ActionClipping
    from model.Encoder import ROMAEncoder, ROMALSTMEncoder
    from model.RLModules import DualHyperQNet, SGHNActor, MLP

    encode_dim = args.encode_dim
    encoder_kwargs = (ROMAEncoder,
                      OrderedDict(  # for ROMAEncoder
                          input_size=33,
                          output_size=encode_dim,
                          role_size=latent_dim,
                          rnn_kwarg=OrderedDict(num_layers=args.num_layers))
                      )

    model_kwargs = OrderedDict(
        Encoder=encoder_kwargs,
        DualCritic=(DualHyperQNet,
                    OrderedDict(input_size=encode_dim, latent_size=latent_dim,
                                action_size=act_dim)),
        Actor=(SGHNActor,
               OrderedDict(input_size=encode_dim, output_size=act_dim, latent_size=latent_dim)),
        DisparityNet=(MLP,
                      OrderedDict(input_size=latent_dim * 2, output_size=1, layer_sizes=[encode_dim],
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

    if train is None:
        train = args.train

    if not train:
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
        reward_fn=None,
        action_clipping=lambda model: ActionClipping(model, **action_clip_kwargs),
        memory_size=args.seq_len
    )

    return ac_kwargs


filename = "zone_" + str(args.climate_zone) + \
           "_lr" + str(args.lr) + \
           "_predLen_" + str(args.seq_len_forecast) + \
           "_actlimit" + str(args.act_limit) + \
           "_num_layers_" + str(args.num_layers) + \
           "_encode_" + str(args.encode_dim) + \
           "_episodeLen_" + str(args.episode_len) + \
           "_seqLen_" + str(args.seq_len) + \
           "_decay_" + str(args.decay) + \
           "_" + str(args.suffix)

# Instantiating the Tensorboard writers
PATH_base = 'datas/new/'
PATH_base = get_output_folder(PATH_base, 'scalar_' + filename)

# for eval stage
reward_writer = SummaryWriter(PATH_base + '/reward')
cost_writer = SummaryWriter(PATH_base + '/cost')
loss_writer = SummaryWriter(PATH_base + '/loss')


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

# Alias
Env = CityLearn
# Instantiating the env
env = Env(data_path, building_attributes, weather_file, solar_profile, building_ids,
          buildings_states_actions=building_state_actions, cost_function=objective_function)
observations_spaces, actions_spaces = env.get_state_action_spaces()

# Provides information on Building type, Climate Zone, Annual DHW demand, Annual Cooling Demand,
# Annual Electricity Demand, Solar Capacity, and correllations among buildings
building_info = env.get_building_information()

# Select many episodes for training. In the final run we will set this value to 1 (the buildings run for one year)
start_episode = 0

ac_kwargs = get_agent_kwargs(args, PATH_base)
rl_agents = RL_Agents(building_info, observations_spaces, actions_spaces, ac_kwargs)

# initialize the model
best_model_path = "./Models_best_zone" + str(args.climate_zone)

model_path = './Models_' + filename
if not os.path.isdir(model_path):
    os.mkdir(model_path)

env_kwargs = dict(
    data_path=data_path,
    building_attributes=building_attributes,
    weather_file=weather_file,
    solar_profile=solar_profile,
    building_ids=building_ids,
    buildings_states_actions=building_state_actions,
    cost_function=objective_function
)
reward_fn = reward_function

MaxEpisode = args.MaxEpisode


def run(time_dim, time_length, episode_maxstep):
    # 若time_length = 1，无需预热也可计算reward
    warmup_step = time_length - 1
    total_step = warmup_step + episode_maxstep
    if args.continue_flag:
        print("training: Load model from episode {}.".format(args.load_episode))
        rl_agents.agent.load_models(model_path, args.load_episode)

    log_iter = 0  # for test reward logging
    for e in range(MaxEpisode):
        if e>0 and e % args.save_per_episode == 0:
            rl_agents.agent.save_models(model_path, e)
            test(model_path, e, log_iter=log_iter)
            log_iter += 1

        # 随机初始化开始日期，8760是一年的总小时，实际下标范围是（0, 8759）
        # 注意无法取到最大值，所以若total_step=1，实际最大只到8758
        start_idx = np.random.randint(0, 8760 - total_step)
        env = Env(simulation_period=(start_idx, start_idx + total_step), **env_kwargs)

        # 注意要重置agent，要清空GRU记忆的状态
        rl_agents.agent.reset()
        # 跑一个Episode，拿warmup + 正常跑的轨迹
        traj, final_state, _ = run_one(env, rl_agents.agent, total_step,
                                       state=None, reset=True)
        # 在时间维度上拼接经验（注意state缺少final_state，后面拼）
        state, action, raw_reward, done = generate_experience(traj, time_dim)

        state_chunks, raw_reward_chunks = magic_manipulation(state, raw_reward,
                                                             time_dim=time_dim,
                                                             time_length=time_length)
        # 获取真正的奖励
        # state_chunks: (Chunks, *, Time_Length, S)
        # raw_reward_chunks: (Chunks, *, Time_Length,)
        # -> rewards:(Chunks, *, 1)
        rewards, reward_logger = reward_fn(raw_reward_chunks, state_chunks, **reward_kwargs)
        rewards = rewards.swapaxes(0, -1)  # -> (*, Chunks) = (*, EpisodeStep)

        # 切割warmup的状态，如果没有warmup，则states长度应该没变化
        warmup_states, states = np.split(state, (warmup_step, ), axis=time_dim)
        if warmup_states.shape[time_dim] <= 0:
            # 如果没有warmup，放None
            warmup_states = None
        _, actions = np.split(action, (warmup_step,), axis=time_dim)
        _, dones = np.split(done, (warmup_step,), axis=time_dim)
        # 现在追加终局状态
        states = np.append(states, np.expand_dims(final_state, time_dim), axis=time_dim)
        # 其他需要的参数都堆在这里
        other = dict(
            init_states=warmup_states,
        )
        # Update
        rl_agents.agent.add_to_buffer(states, actions, rewards, done=dones, **other)
        rl_agents.agent.update_agent()


def print_grad(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param)


def print_weight(model):
    print(list(model.parameters()))


def test(model_path, e, log_iter):
    print("===============test stage start================")
    test_ac_kwargs = get_agent_kwargs(args, PATH_base, train=False)
    test_agents = RL_Agents(building_info, observations_spaces, actions_spaces, test_ac_kwargs)
    print("testing: Load model from episode {}.".format(e))
    test_agents.agent.load_models(model_path, e)
    # print_grad(test_agents.agent.actor)
    # return

    env = CityLearn(**env_kwargs)
    state = env.reset()
    test_agents.agent.reset()

    done = False
    k = 0
    cum_reward = {}
    for id in range(test_agents.n_buildings):
        cum_reward[id] = 0

    cost = {}
    while not done:
        with torch.no_grad():
            action = test_agents.select_action(state)
            next_state, raw_reward, done, _ = env.step(action)
            state = next_state
            if k % 1000 == 0:
                print("testing time step:{}, write rewards".format(k))
                for id in range(test_agents.n_buildings):
                    cum_reward[id] += raw_reward[id]
                    reward_writer.add_scalar('building_reward_' + str(id), raw_reward[id], log_iter * 8760 + k)
            k += 1

    # write episode-accumulated reward
    for r in range(test_agents.n_buildings):
        reward_writer.add_scalar('building_cum_reward_' + str(r), cum_reward[r], e)

    # write cost
    cost[e] = env.cost()
    print("cost_writer adding scalar")
    for i in range(len(objective_function)):
        cost_writer.add_scalar(str("cost_") + objective_function[i], cost[e][objective_function[i]], e)
        cost_writer.flush()


def magic_manipulation(*arrs, time_dim=-2, time_length=24):
    # 其实就是对输入的多个arr进行统一分割
    return [split_chunks(arr, time_dim, time_length) for arr in arrs]


def split_chunks(arr, time_dim, time_length):
    """
    :param arr:
            state: (9, t_len, state)
            reward: (9, t_len)
    :param time_dim:
    :param time_length:
            real_t_len = t_len - (time_length - 1)
    :return: (real_t_len, *, time_length, S)
    """
    t_len = arr.shape[time_dim]  # 看看总共有多少个时间步
    # 生成(总时间步 * 时间片长度）的下标，
    real_t_len = t_len - (time_length - 1)
    idx = np.arange(0, real_t_len)[:, None] + np.arange(0, time_length)[None]
    # 这里做了三件事，首先拍扁idx
    # 然后在time_dim维度上按照下标，选取(总时间步 * 时间片长度）个数据
    # 再分成t_len个片段，得到(t_len, *, time_length, S)

    arr = arr[:, idx].swapaxes(time_dim, 0)
    # arr = torch.FloatTensor(arr)
    # arr = arr.index_select(idx.flatten(), time_dim).numpy()
    return arr


def generate_experience(traj, time_dim):
    # 在time_dim的位置增添1个维度，然后在这个维度上串联数据
    # traj: list of tuple(state, action, reward, done), list len: total length
    # list(zip(*traj)): list len=3, tuple(state, action, reward, done)
    """
    :param traj:
    :param time_dim:
    :return: for case when time_dim=1:
                out = [state_arr, action_arr, reward_arr, done_arr]
                shape:
                    state_arr (9, tot_len, state)
                    action_arr (9, tot_len, 2)
                    reward_arr (9, tot_len)
                    done_arr (9, tot_len)
    """
    out = []
    for arr in list(zip(*traj)):
        if np.array(arr[0]).ndim == 0:  # deal with done
            arr = ([[x] * 9 for x in arr])
        tmp = np.stack(arr, axis=time_dim)
        out.append(tmp)
    # return [np.stack(arr, axis=time_dim) for arr in list(zip(*traj))]
    return out


def run_one(env, rl_agents, step, state=None, reset=False):
    Traj = []
    if state is None:
        assert reset
    if reset:
        state = env.reset()
        rl_agents.reset()

    for _ in range(step):
        action = rl_agents.select_action(state)
        next_state, raw_reward, done, _ = env.step(action)
        Traj.append((state, action, raw_reward, done))
        state = next_state

    return Traj, next_state, done


run(time_dim=1, time_length=args.seq_len, episode_maxstep=args.episode_len)
