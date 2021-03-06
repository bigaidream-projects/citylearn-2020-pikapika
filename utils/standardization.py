from copy import deepcopy
from collections import Iterable

import numpy as np


"""
all input state:

month: 1 (January) through 12 (December)
day: type of day as provided by EnergyPlus (from 1 to 8). 1 (Sunday), 2 (Monday), ..., 7 (Saturday), 8 (Holiday)
hour: hour of day (from 1 to 24).
daylight_savings_status: indicates if the building is under daylight savings period (0 to 1). 0 indicates that the building has not changed its electricity consumption profiles due to daylight savings, while 1 indicates the period in which the building may have been affected.

t_out: outdoor temperature in Celcius degrees.
t_out_pred_6h: outdoor temperature predicted 6h ahead (accuracy: +-0.3C)
t_out_pred_12h: outdoor temperature predicted 12h ahead (accuracy: +-0.65C)
t_out_pred_24h: outdoor temperature predicted 24h ahead (accuracy: +-1.35C)

rh_out: outdoor relative humidity in %.
rh_out_pred_6h: outdoor relative humidity predicted 6h ahead (accuracy: +-2.5%)
rh_out_pred_12h: outdoor relative humidity predicted 12h ahead (accuracy: +-5%)
rh_out_pred_24h: outdoor relative humidity predicted 24h ahead (accuracy: +-10%)

diffuse_solar_rad: diffuse solar radiation in W/m^2.
diffuse_solar_rad_pred_6h: diffuse solar radiation predicted 6h ahead (accuracy: +-2.5%)
diffuse_solar_rad_pred_12h: diffuse solar radiation predicted 12h ahead (accuracy: +-5%)
diffuse_solar_rad_pred_24h: diffuse solar radiation predicted 24h ahead (accuracy: +-10%)

direct_solar_rad: direct solar radiation in W/m^2.
direct_solar_rad_pred_6h: direct solar radiation predicted 6h ahead (accuracy: +-2.5%)
direct_solar_rad_pred_12h: direct solar radiation predicted 12h ahead (accuracy: +-5%)
direct_solar_rad_pred_24h: direct solar radiation predicted 24h ahead (accuracy: +-10%)

t_in: indoor temperature in Celcius degrees.
avg_unmet_setpoint: average difference between the indoor temperatures and the cooling temperature setpoints in the different zones of the building in Celcius degrees. sum((t_in - t_setpoint).clip(min=0) * zone_volumes)/total_volume
rh_in: indoor relative humidity in %.
non_shiftable_load: electricity currently consumed by electrical appliances in kWh.
solar_gen: electricity currently being generated by photovoltaic panels in kWh.
cooling_storage_soc: state of the charge (SOC) of the cooling storage device. From 0 (no energy stored) to 1 (at full capacity).
dhw_storage_soc: state of the charge (SOC) of the domestic hot water (DHW) storage device. From 0 (no energy stored) to 1 (at full capacity).

"""


def periodic_encoding(raw_state, period):
    old_shape = raw_state.shape[:-1]
    phase = raw_state / period
    phase *= 2*np.pi
    out = np.stack((np.sin(phase), np.cos(phase)), axis=-1).reshape(*old_shape, -1)
    return out


def normalize(raw_state, mean, std):
    assert std > 0. if type(std) is float or type(std) is int\
        else (std > 0.).all(), "All std must be greater than 0, got {} instead.".format(std)
    return (raw_state - mean) / std


def mean_normalize(raw_state, mean, min_val=0., max_val=1., eps=1e-12):
    delta = max_val - min_val
    return normalize(raw_state, mean, delta + eps)


def rescale(raw_state, min_val=0., max_val=1., eps=1e-12):
    delta = max_val - min_val
    return normalize(raw_state, min_val, delta + eps)


def to_onehot(raw_state, num_class):
    old_shape = raw_state.shape[:-1]
    onehot_mat = np.eye(num_class)
    onehot_idx = raw_state.flatten().astype(np.int)
    out = onehot_mat[onehot_idx].reshape(*old_shape, -1)
    return out


def daytype_to_onehot_(num):
    out = []
    tmp_num = deepcopy(num)
    cnt = 0
    while cnt < 4:
        out.append(tmp_num % 2)
        tmp_num = tmp_num // 2
        cnt += 1
    return out


def daytype_to_onehot(raw_state):  # (9,1)
    # input: 1~8
    old_shape = raw_state.shape
    raw_state = raw_state.reshape(-1, old_shape[-1])
    batch_size = raw_state.shape[0]
    out = np.zeros((batch_size, 4))
    for i in range(batch_size):
        out[i] = daytype_to_onehot_(raw_state[i][0])
    out = np.array(out.reshape(*old_shape[:-1], -1))
    return out


eps = 0.01
# Start Index
PRED_LEN = 3  # (6hr, 12hr, 24hr)
DATA_LEN = [1] * 4
DATA_LEN += [1+PRED_LEN] * 4
DATA_LEN += [1] * 7
SPLIT_IDX = np.cumsum(DATA_LEN, dtype=np.int)[:-1]  # Remove Last Length
STATIC_LOAD_IDX = 23
SOLAR_POWER_IDX = 24

MIN_T, MAX_T = -5, 40
T_IN_MEAN = 23.


def _log_normalize(state, mean=0., std=1.):
    return normalize(np.log10(state + eps), mean=mean, std=std)


def _rel_humidity_norm(state):
    return rescale(state, max_val=100.)


"""
unpredictable variables:
Month
Day Type
DayLight Saving Status

predictable variables:
Others

Not included:
SOC
"""
# dim: 21 nonpred + 12 pred = 33
func_callbacks = [
    lambda state: periodic_encoding(state, period=12),  # Month - 2
    lambda state: daytype_to_onehot(state-1),  # Day Type - 4
    lambda state: periodic_encoding(state, period=24),  # Hour - 2
    lambda state: to_onehot(state, num_class=2),  # DayLight Saving Status - 2
    lambda state: rescale(state, min_val=MIN_T, max_val=MAX_T),  # Outdoor Temperature
    _rel_humidity_norm,  # Outdoor Relative Humidity
    lambda state: _log_normalize(state, std=3.),  # Diffuse Sunlight Radiation
    lambda state: _log_normalize(state, std=3.),  # Direct Sunlight Radiation
    lambda state: mean_normalize(state, mean=T_IN_MEAN, max_val=T_IN_MEAN),  # Indoor Temperature
    lambda state: _log_normalize(state, std=3.),  # Avg. Unmet SetPoint
    _rel_humidity_norm,  # Indoor Relative Humidity
    lambda state: _log_normalize(state, std=3.),  # Static Load
    lambda state: _log_normalize(state, std=3.),  # Solar_Generation
    lambda state: state,  # Cooling Storage
    lambda state: state  # DHW Storage
]


def normalize_seq2seq_state(states, full_history=False, future_len=6, pretrain=False):
    # TODO: discard SOC variable
    """
    :param states: (9, seq, 27)
    :return: source_state: (9, seq, 19)
             target_state: (9, seq, 4)
    """
    states = deepcopy(states)
    input_shape = states.shape
    if states.ndim == 2:
        states = np.expand_dims(states, 1)

    state_list = np.split(states, SPLIT_IDX, -1)

    pred6hr_state_list = []
    pred12hr_state_list = []
    result_list = []
    for state, func in zip(state_list, func_callbacks):
        result_list.append(func(state))

    # swap Hour and DayLight Saving Status
    result_list[2], result_list[3] = result_list[3], result_list[2]

    # discard prediction states
    for i in range(4, 8):
        if future_len == 6:
            pred6hr_state_list.append(deepcopy(result_list[i][:, :, 1:2]))  # use pred 6hr as target in Seq2Seq
        if future_len == 12:
            pred6hr_state_list.append(deepcopy(result_list[i][:, :, 1:2]))  # use pred 6hr as target in Seq2Seq
            pred12hr_state_list.append(deepcopy(result_list[i][:, :, 2:3]))  # use pred 12hr as target in Seq2Seq

        if not full_history:
            result_list[i] = result_list[i][:, :, 0:1]  # discard predictions

    # discard soc states when called in pre-training stage
    if pretrain:
        result_list = result_list[:-2]

    source_state = np.concatenate(result_list, -1)

    source_state = source_state.reshape(*input_shape[:-1], -1)
    target_state_6hr = np.concatenate(pred6hr_state_list, -1)
    target_state_6hr = target_state_6hr.reshape(*input_shape[:-1], -1)

    if future_len == 6:
        return source_state, target_state_6hr
    elif future_len == 12:
        target_state_12hr = np.concatenate(pred12hr_state_list, -1)
        target_state_12hr = target_state_12hr.reshape(*input_shape[:-1], -1)
        if pretrain:
            return source_state, (target_state_6hr, target_state_12hr)
        else:
            target_arr = np.concatenate((target_state_12hr, target_state_6hr), -1)
            return source_state, target_arr


def normalize_seq2seq_state_forRL(states, future_len):
    # return normalized states for RL training (seq2seqTCN, seq2seqSymTCN)
    """
    :param states: (9, seq, 27)
    :return: out: (9, seq, 21+4 or 21+4+4)
    """
    src, tgt = normalize_seq2seq_state(states, full_history=False, future_len=future_len)
    out = np.concatenate((src, tgt), -1)
    return out


def normalize_seq2seq_mixed_state_forRL(states, future_len):
    # return normalized states for RL training (mixedTCNEncoder)
    """
    :param states: (9, seq, 27)
    :return: out: (9, seq, 33+19+4)
    """
    src, tgt = normalize_seq2seq_state(states, full_history=False, future_len=future_len)
    src_full, _ = normalize_seq2seq_state(states, full_history=True, future_len=future_len)
    out = np.concatenate((src_full, src, tgt), -1)
    return out


def normalize_AE_state(states, noSOC=True):  # return normalized states for AE (no pred, soc)
    """
    :param states: (9, seq, 27)
    :return: (9, seq, 19)
    """
    state_list = np.split(states, SPLIT_IDX, -1)
    result_list = []
    for state, func in zip(state_list, func_callbacks):
        result_list.append(func(state))

    # swap Hour and DayLight Saving Status
    result_list[2], result_list[3] = result_list[3], result_list[2]

    # discard prediction states
    for i in range(4, 8):
        result_list[i] = result_list[i][:, 0:1]

    # discard soc states
    if noSOC:
        return np.concatenate(result_list[:-2], -1)
    else:
        return np.concatenate(result_list, -1)


def normalize_AE_state_with_pred(states, noSOC=True):  # return normalized states for AE (no pred, soc)
    """
    :param states: (9, seq, 27)
    :return: (9, seq, 31)
    """
    state_list = np.split(states, SPLIT_IDX, -1)
    result_list = []
    for state, func in zip(state_list, func_callbacks):
        result_list.append(func(state))

    # swap Hour and DayLight Saving Status
    result_list[2], result_list[3] = result_list[3], result_list[2]
    
    # discard soc states
    if noSOC:
        return np.concatenate(result_list[:-2], -1)
    else:
        return np.concatenate(result_list, -1)


def normalize_state(states):  # return all normalized states
    """
    :param states: (9, seq, 27)
    :return: (9, seq, 33)
    """
    state_list = np.split(states, SPLIT_IDX, -1)
    result_list = []
    for state, func in zip(state_list, func_callbacks):
        result_list.append(func(state))

    return np.concatenate(result_list, -1)


def normalize_disabled_state(states):  # return 21-dim states without pred states
    """
    :param states: (9, seq, 27)
    :return: source_state: (9, seq, 21)
    """
    input_shape = states.shape
    if states.ndim == 2:
        states = np.expand_dims(states, 1)

    state_list = np.split(states, SPLIT_IDX, -1)
    result_list = []
    for state, func in zip(state_list, func_callbacks):
        result_list.append(func(state))

    # swap Hour and DayLight Saving Status
    result_list[2], result_list[3] = result_list[3], result_list[2]

    # discard prediction states
    for i in range(4, 8):
        result_list[i] = result_list[i][:, :, 0:1]
    source_state = np.concatenate(result_list, -1)

    source_state = source_state.reshape(*input_shape[:-1], -1)

    return source_state


if __name__ == "__main__":
    s = np.ones((10, 9, 4, 27), dtype=np.float32)
    normalize_state(s)
