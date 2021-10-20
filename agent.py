from agents.SAC import SACAgent
from agents.ROMASAC import ROMASACAgentCore
from reward_function import reward_function
#
# args = {
#     "A_r": 0,
#     "B_r": 1,
#     "BATCH_SIZE": 32,
#     "MAX_BUFFER": 10000,
#     "act_limit_1": 0.5,
#     "act_limit_2": 0.5,
#     "act_limit_3": 0.5,
#     "act_limit_4": 0.5,
#     "alpha": 0.2,
#     "decay": 1,
#     "encode_dim": 128,
#     "kernel_size": 3,
#     "levels": 5,
#     "load_episode": 0,
#     "lr": 0.0001,
#     "price_factor": 0.01,
#     "print_per_step": 1000,
#     "seed": 0,
#     "seq_len": 24,
#     "window_len_A": 6,
#     "window_len_B": 12,
# }
# # for HRL agents
# args_low_level = {
#     "A_r": 0,
#     "B_r": 1,
#     "BATCH_SIZE": 32,
#     "MAX_BUFFER": 10000,
#     "act_limit_1": 0.5,
#     "act_limit_2": 0.5,
#     "act_limit_3": 0.5,
#     "act_limit_4": 0.5,
#     "alpha": 0.2,
#     "decay": 1,
#     "encode_dim": 128,
#     "kernel_size": 3,
#     "levels": 2,
#     "load_episode": 0,
#     "lr": 0.0001,
#     "price_factor": 0.01,
#     "print_per_step": 500,
#     "seed": 0,
#     "seq_len": 168,
#     "window_len_A": 6,
#     "window_len_B": 12,
#     "goal_dim": 4,
# }
#
# args_high_level = {
#     "A_r": 0,
#     "B_r": 1,
#     "BATCH_SIZE": 32,
#     "MAX_BUFFER": 10000,
#     "act_limit_1": 0.5,
#     "act_limit_2": 0.5,
#     "act_limit_3": 0.5,
#     "act_limit_4": 0.5,
#     "alpha": 0.2,
#     "decay": 1,
#     "encode_dim": 128,
#     "kernel_size": 3,
#     "levels": 2,
#     "load_episode": 0,
#     "lr": 0.0001,
#     "price_factor": 0.01,
#     "print_per_step": 500,
#     "seed": 0,
#     "seq_len": 168,
#     "window_len_A": 6,
#     "window_len_B": 12,
#     "goal_dim": 0,
# }


class RL_Agents:
    def __init__(self, building_info, observation_spaces, action_spaces, ac_kwargs):
        self.n_buildings = len(building_info)
        # self.climate_zone_flag = 1
        # for i in building_info:
        #     self.climate_zone_flag = building_info[i].climate_zone
        # print("climate:", self.climate_zone_flag)
        self.agent = ROMASACAgentCore(observation_spaces=observation_spaces, action_dim=2,
                                      **ac_kwargs)
        # self.best_model_path = 'Models_best_zone' + str(self.climate_zone_flag)
        # self.agent.load_models(self.best_model_path)

    def select_action(self, state):
        return self.agent.select_action(state)

    def add_to_buffer(self, state, actions, raw_rewards, next_state=None, done=None, **kwargs):
        self.agent.add_to_buffer(state, actions, raw_rewards, next_state=next_state, done=done, **kwargs)

