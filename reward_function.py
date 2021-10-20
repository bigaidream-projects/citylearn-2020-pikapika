"""
This function is intended to wrap the rewards returned by the CityLearn RL environment, and is meant to 
be modified by the participants of The CityLearn Challenge.
CityLearn returns the energy consumption of each building as a reward. 
This reward_function takes all the electrical demands of all the buildings and turns them into one or
 multiple rewards for the agent(s)

The current code computes a virtual (made-up) electricity price proportional to the total demand for
 electricity in the neighborhood, and multiplies every
reward by that price. Then it returns the new rewards, which should be used by the agent. Participants
 of the CityLearn Challenge are encouraged to completely modify this function
in order to minimize the 5 proposed metrics.
"""
import numpy as np
from utils.standardization import STATIC_LOAD_IDX, SOLAR_POWER_IDX, mean_normalize

# hyper-parameters
NSL_IDX = STATIC_LOAD_IDX
SGEN_IDX = SOLAR_POWER_IDX


# TODO: Renew reward procesing function
def reward_function(rewards, states, alpha=1, beta=1,
                    total_energy_window=12, heat_energy_window=12,
                    price_factor=0.01, ramping_window=6):
    """
    :param rewards: np.array (*, Building, Time),  Negative of building_energy_demand
    :param states: np.array (*, Building, Time, Raw_State),  Raw States of Related Rewards
    :param alpha: float, weight for smoothness of total demand reward
    :param beta: float, weight for heat demand reward
    :param total_energy_window: int, Length of slide window mean for total demand
    :param heat_energy_window: int, Length of slide window mean for heat demand
    :param price_factor: float, weight for global price
    :param ramping_window: int, Length of slide window mean for smoothness of total demand
    :return:
        final_rewards - nd.array (*, Building, 1), processed current reward
        reward_logger - list of min-max value of smoothness reward, heat demand and final reward
    """
    """
    compute demand for DHW & cooling
    rewards[i] = - (cooling_demand[i] + DHW_demand[i] - solar_gen[i] + no_shiftable_load[i])
                = - cooling_demand[i] - DHW_demand[i] + solar_gen[i] - no_shiftable_load[i]
    no_shiftable_load[i] = states[i][23], solar_gen[i] = states[i][24]
    """

    # compute the price
    heat_len = heat_energy_window
    tot_len = total_energy_window
    assert states.shape[:-3] == rewards.shape[:-2]
    lead_dims = states.shape[:-2]
    states = states.reshape(-1, *states.shape[-3:])  # (*, building, Time, State)
    rewards = rewards.reshape(-1, *rewards.shape[-2:])  # (*, building, Time)
    price_total = np.maximum(-rewards[:, :, -1].sum(axis=-1) * price_factor, 0.)

    total_rewards = rewards[:, :, -tot_len:]
    # heat_reward = negative_total_consumption - solar_energy + non-shiftable energy
    heat_rew = rewards[:, :, -heat_len:] - states[:, :, -heat_len:, SGEN_IDX] + states[:, :, -heat_len:, NSL_IDX]
    heat_rew = np.minimum(0., heat_rew)
    current_heat_reward = heat_rew[:, :, -1]

    def get_slider_mean(slider):
        # if slider.shape[-1] == 1 and heat_len == 1:
        if heat_len == 1:
            return 0.
        else:
            return np.mean(slider, axis=-1)

    def ramping_reward(reward, ramp_window):
        if reward.shape[-1] > 1:  # TODO
            return np.abs(reward[:, :, -ramp_window:] - reward[:, :, -ramp_window - 1:-1]).sum(-1)
        else:
            return 0.

    mean_heat_reward = get_slider_mean(heat_rew)
    heat_advantage = beta * mean_normalize(current_heat_reward, mean_heat_reward)

    ramp_len = min(total_rewards.shape[-1] - 1, ramping_window)
    total_ramping = - alpha * ramping_reward(total_rewards, ramp_len)
    final_rewards = np.expand_dims(price_total, -1) * heat_advantage + total_ramping
    final_rewards = final_rewards.reshape(*lead_dims)
    reward_logger = [np.max(heat_advantage), np.min(heat_advantage),
                     np.max(total_ramping), np.min(total_ramping),
                     np.max(final_rewards), np.min(final_rewards),
                     np.max(price_total), np.min(price_total)]

    return final_rewards, reward_logger


# Reward function for the multi-agent (decentralized) agents
def reward_function_ma(electricity_demand):
    return electricity_demand


# Reward function for the single-agent (centralized) agent
def reward_function_sa(electricity_demand):
    total_energy_demand = 0
    for e in electricity_demand:
        total_energy_demand += -e

    price = max(total_energy_demand * 0.01, 0)

    for i in range(len(electricity_demand)):
        electricity_demand[i] = min(price * electricity_demand[i], 0)

    return sum(electricity_demand)


def reward_function_sa_all(electricity_demand):
    total_energy_demand = 0
    for e in electricity_demand:
        total_energy_demand += -e

    price = max(total_energy_demand * 0.01, 0)

    for i in range(len(electricity_demand)):
        electricity_demand[i] = min(price * electricity_demand[i], 0)

    demand_sum = sum(electricity_demand)

    for i in range(len(electricity_demand)):
        electricity_demand[i] = demand_sum

    return electricity_demand


def reward_function_sa_mean(electricity_demand):
    total_energy_demand = 0
    for e in electricity_demand:
        total_energy_demand += -e

    price = max(total_energy_demand * 0.01, 0)

    for i in range(len(electricity_demand)):
        electricity_demand[i] = min(price * electricity_demand[i], 0)

    demand_sum = np.mean(np.array(electricity_demand))

    for i in range(len(electricity_demand)):
        electricity_demand[i] = demand_sum

    return electricity_demand
