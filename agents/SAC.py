import sys
import time
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from utils.buffer import MemoryBuffer
from agents.BaseAgent import BaseAgent
from model.BaseManager import ModelCallbacks, Memory, ParamGroupCallbacks, MLFeatures
from utils import get_module_device
from utils.io import make_path
from utils.standardization import normalize_state
from utils.util import USE_CUDA, to_numpy, cat_oa, to_tensors

TARGET_SUFFIX = '_target'
stdout = sys.stdout


class SoftActorCriticProperty(MLFeatures):
    __RequiredTarget__ = {'Actor': False, 'DualCritic': True, 'Encoder': False}
    __IgnoreSuffix__ = TARGET_SUFFIX

    @property
    def actor(self):
        return self._model_callbacks['Actor']

    @property
    def critic(self):
        return self._model_callbacks['DualCritic']

    @property
    def critic_target(self):
        return self._model_callbacks['DualCritic' + self.__IgnoreSuffix__]

    @property
    def encoder(self):
        return self._model_callbacks['Encoder']

    @property
    def critic_param_groups(self):
        return [self._param_groups['Encoder'], self._param_groups['DualCritic']]

    @property
    def policy_param_groups(self):
        return [self._param_groups['Actor']]


class SACAlgo(SoftActorCriticProperty, ParamGroupCallbacks):
    __BufferType__ = MemoryBuffer

    def __init__(self, model_callbacks, batch_size, alpha, buffer_capacity,
                 optim_cls, optim_kwargs, unique_optim_kwargs=None,
                 verbose=False, action_clipping=None, log_interval=1000, log_path='',
                 iterations=1, **kwargs):
        super(SACAlgo, self).__init__()
        self._model_callbacks = model_callbacks  # Access Model from agent
        missing_callbacks = set(self.__RequiredTarget__.keys()) - set(self._model_callbacks.keys())
        if len(missing_callbacks) > 0:
            RuntimeError('Missing model_callbacks {}'.format(missing_callbacks))

        self.BATCH_SIZE = batch_size
        self.alpha = alpha
        self.BETA = 0.995
        self.GAMMA = 0.99

        self.time_step = 0
        self.iterations = iterations  # Number of updates of the actor-critic networks every time-step

        self.buffer = self.__BufferType__(buffer_capacity)
        self.rewards_logger = []

        self.verbose = verbose
        self.time = time.time()

        # TODO Remove hard-coding action clipping
        self.action_clipping = None
        if action_clipping is not None:
            self.action_clipping = action_clipping

        # Parameters
        self.grad_norm = 1000
        self.log_interval = log_interval
        self.log_path = log_path

        self.init_param_groups(unique_optim_kwargs=unique_optim_kwargs)
        self._optimizer_initialize(optim_cls, optim_kwargs)

    def _optimizer_initialize(self, optim_cls, optim_kwargs):
        self.critic_optimizer = optim_cls(self.critic_param_groups, **optim_kwargs)
        self.policy_optimizer = optim_cls(self.policy_param_groups, **optim_kwargs)

    def optimize(self):
        """
        Samples action random batch from replay memory and performs optimization
        :return:
        """
        if len(self.buffer) < self.BATCH_SIZE:
            return

        self.time_step += 1
        self._model_callbacks.train()
        model_device = get_module_device(self._model_callbacks)
        for i in range(self.iterations):
            state, action, reward, next_state, done, info = self._sample_buffer(model_device)
            loss_q, q_info = self.update_critic(state, action, reward, next_state, done, **info)
            loss_pi, pi_info = self.update_policy(state, **info)
            if self.verbose and self.time_step % self.log_interval == 0:
                self._logging(loss_pi, loss_q, pi_info, q_info)
            self.update_target()

    def update_critic(self, state, action, reward, next_state, done, **other_kwargs):
        # First run one gradient descent step for Q1 and Q2
        self.critic_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(state, action, reward, next_state, done, **other_kwargs)
        loss_q.backward()
        self.group_grad_norm(self.critic_param_groups)
        self.critic_optimizer.step()
        return loss_q, q_info

    def compute_loss_q(self, state, action, reward, next_state, done, gamma=None, **other_kwargs):
        """
            state/next_state: (batch, s_dim)
            action: (batch, a_dim)
            reward: (batch,)
            done: (batch,)
        """
        assert gamma is not None

        # TODO: Amend dual output
        q1, q2 = self.critic(cat_oa(state, action))
        # Bellman target for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            next_action, logp_a2 = self.actor.sample_action(next_state)
            # Target Q-values
            q_target = torch.min(*self.critic_target(cat_oa(next_state, next_action)))
            target = reward + gamma * (1 - done) * (q_target - self.alpha * logp_a2)

        # MSE loss against Bellman target
        loss_q = F.smooth_l1_loss(q1, target) + F.smooth_l1_loss(q2, target)

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().cpu().numpy(),
                      Q2Vals=q2.detach().cpu().numpy())

        return loss_q, q_info

    def print_weight(self, model):
        print(list(model.parameters()))

    def update_policy(self, state, **other_kwargs):
        # Next run one gradient descent step for pi.
        self.policy_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(state, **other_kwargs)
        loss_pi.backward()
        self.group_grad_norm(self.policy_param_groups)
        self.policy_optimizer.step()
        # TODO Remove hard-coding action clipping
        if self.action_clipping is not None:
            self.action_clipping.step()
        return loss_pi, pi_info

    def compute_loss_pi(self, state, **other_kwargs):
        state = state.detach()  # Detach Gradient from Encoder
        action, logp_pi = self.actor.sample_action(state)
        self.critic.requires_grad_(False)  # Temporarily Disable Gradient Recording
        q_pi = torch.min(*self.critic(cat_oa(state, action)))
        self.critic.requires_grad_(True)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()
        if self.time_step % self.log_interval == 0:
            print("time step:{} loss_pi:{}".format(self.time_step, loss_pi))
        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().cpu().numpy())

        return loss_pi, pi_info

    def update_target(self):
        # soft update
        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.critic.parameters(), self.critic_target.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.BETA)
                p_targ.data.add_((1 - self.BETA) * p.data)

    def _logging(self, loss_pi, loss_q, pi_info, q_info):
        with open('{}/grad_loss.txt'.format(self.log_path), 'a+') as file:
            sys.stdout = file
            print("=======start=========")
            print(str(self.log_interval) + " step time:", time.time() - self.time)
            self.time = time.time()
            print("______actor grads______")
            self.print_grad(self.actor)
            print("______critic grads______")
            self.print_grad(self.critic)
            print("______encoder grads______")
            self.print_grad(self.encoder)
            print("=======end========")
            print("loss:")
            print("actor:", loss_pi)
            print("log p_pi:", np.max(pi_info['LogPi']), np.min(pi_info['LogPi']))
            print("critic:", loss_q)
            print("q1, max:", np.max(q_info['Q1Vals']), "min:", np.min(q_info['Q1Vals']))
            print("q2, max:", np.max(q_info['Q2Vals']), "min:", np.min(q_info['Q2Vals']))
            print("cur rewards:", np.array(self.rewards_logger))
        sys.stdout = stdout

    def _sample_buffer(self, model_device):
        # for SACAgent
        state, action, reward, done, info = self.buffer.sample(self.BATCH_SIZE)
        state, action, reward, done = \
            to_tensors(state, action, reward, done, requires_grad=True, device=model_device)
        info.update(zip(info.keys(), to_tensors(*info.values(), requires_grad=True, device=model_device)))
        return self._process_buffer(state, action, reward, done, **info)

    def _process_buffer(self, state, action, reward, done, gamma=None, next_state=None, **info):
        state = self._encode_state(state)
        next_state = self._encode_state(next_state)
        gamma = torch.stack(gamma.expand(*done.shape[1:], -1).unbind(-1))
        info.update(gamma=gamma)
        return state, action, reward, next_state, done, info

    def _encode_state(self, state, **kwargs):
        return self.encoder(state, **kwargs)

    def group_grad_norm(self, param_group):
        grad_norms = []
        for pg in param_group:
            grad_norms.append(torch.nn.utils.clip_grad_norm_(pg['params'], self.grad_norm))
        return grad_norms

    def update_buffer(self, state, action, rewards, done, gamma=None, reward_logger=None, **kwargs):
        # Information contained in the building_info variable can be used to
        # choose the number of buffers and what information goes to each buffer
        # add item to buffer
        if gamma is None:
            gamma = self.GAMMA
        self.buffer.append(state, action, rewards, done, gamma=gamma, **kwargs)
        self.rewards_logger = reward_logger

    @staticmethod
    def print_grad(net):
        for name, parms in net.named_parameters():
            if parms.grad is None:
                continue
            print('-->name:', name, '-->grad_requires:', parms.requires_grad,
                  ' -->grad_value:', torch.max(parms.grad), torch.min(parms.grad))


class SACAgentCore(SoftActorCriticProperty, ModelCallbacks, BaseAgent):
    __AlgorithmArch___ = SACAlgo

    def __init__(self, observation_spaces, action_dim,
                 model_kwargs=None, algo_kwargs=None, state_fn=normalize_state, reward_fn=None, action_clipping=None,
                 memory_size=1, **kwargs):
        super(SACAgentCore, self).__init__()
        if memory_size <= 0:
            raise AttributeError("memory_size should be larger than 0.")

        # Hyper-parameters
        self.observation_spaces = np.stack([obs.sample() for obs in observation_spaces])
        self.action_dim = action_dim

        # State sequence Storage
        self.state_fn = state_fn
        if reward_fn is not None:
            # Redefine reward_fn
            rew_fn = reward_fn

            def reward_fn(reward, state, valid=None):
                # For the shape of reward & state, ref Memory in BaseManager.py
                if valid is not None:
                    # TODO: only correct when episode trajectory is not used
                    # TODO: elements in valid mask are all True when use seq2seq Model
                    lead_dims = state.shape[:-1]
                    reward = reward[valid].reshape(*lead_dims)
                    state = state[valid].reshape(*lead_dims, -1)
                return rew_fn(reward, state)

        else:
            # noinspection PyUnusedLocal
            def reward_fn(reward, *inputs, **other):
                # For the shape of reward & state, ref Memory in BaseManager.py
                reward = reward[:, -1]
                return reward, [max(reward), min(reward)]
        self.reward_func = reward_fn

        self.init_model_callbacks(model_kwargs)
        self._initialize()

        # Action Clipping
        self.action_clipping = None
        if action_clipping is not None:
            self.action_clipping = action_clipping(self.actor)
            self.action_clipping.add_clipping()

        # Algo for model update
        if algo_kwargs is not None:
            # Train Mode
            algo_kwargs.update(action_clipping=self.action_clipping)
            self.algo = self.__AlgorithmArch___(self._model_callbacks, **algo_kwargs)
            self.deterministic = False  # Use Stochastic Policy for Training
        else:
            # Eval Mode Only
            self.algo = None
            self.eval()

    def _initialize(self):
        # Create actor-critic module and target networks
        input_size, output_size = self.make_env_to_model_kwargs(self.observation_spaces), self.action_dim
        for kwarg in self._model_kwargs.values():
            kwarg.setdefault('input_size', input_size)
            kwarg.setdefault('output_size', self.action_dim)
        self._init_models(update_keys=self._model_type_callbacks.keys())
        if USE_CUDA:
            self._model_callbacks.cuda()

    def make_env_to_model_kwargs(self, observation_spaces):
        obs_shape = self.state_fn(observation_spaces).shape[-1]
        return obs_shape

    def select_action(self, state, deterministic=None, **kwargs):
        """
        gets the action from actor added with exploration noise
        :param state: state (Numpy array)
        :param deterministic: Bool, use agent flag by default

        :return: sampled action (Numpy array)
        """
        self._model_callbacks.eval()
        if deterministic is None:
            deterministic = self.deterministic
        kwargs.update(deterministic=deterministic)

        with torch.no_grad():
            hidden_state = self._encode_state(state, **kwargs)
            action = self.actor(hidden_state, **kwargs)
            action = to_numpy(action)

        return action

    def _encode_state(self, state, **kwargs):
        state = to_tensors(self.state_fn(state), device=get_module_device(self._model_callbacks))
        hidden_state = self.encoder(state.unsqueeze(0), **kwargs).squeeze(0)
        return hidden_state

    def add_to_buffer(self, state, action, rewards, done, **kwargs):
        # Information contained in the building_info variable can be used to
        # choose the number of buffers and what information goes to each buffer
        self._add_to_buffer(state, action, rewards, done, **kwargs)
        if np.all(done):
            self.reset()

    def _add_to_buffer(self, state, action, reward, done,
                       init_states=None, next_state=None, valid_mask=None, **kwargs):
        if self.algo is not None:
            # Pre-processing
            rewards, rewards_logger = self.reward_func(reward, state, valid_mask)
            if init_states is not None:
                kwargs.update(init_states=self.state_fn(init_states))
            if next_state is not None:
                kwargs.update(next_state=self.state_fn(next_state))
            if valid_mask is not None:
                kwargs.update(valid_mask=valid_mask)
            self.algo.update_buffer(self.state_fn(state), action, rewards, done,
                                    reward_logger=rewards_logger, **kwargs)

    def update_agent(self):
        if self.algo is not None:
            self.algo.optimize()

    def reset(self):
        pass
        # Clean Memory & Reset Time Step
        # self.time_step = 0
        # self.state_memory.fill(0.)
        # self.reward_memory.fill(0.)
        # self.valid_mask.fill(False)

    def train(self, mode=True):
        self.deterministic = ~mode
        super(SACAgentCore, self).train(mode)

    def load_models(self, path, episode=None):
        if episode is not None:
            path = '{}/entry_{}'.format(path, episode)
        super(SACAgentCore, self).load_models(path)

    def save_models(self, path, episode=None):
        # TODO: Remove this line
        if self.action_clipping is not None:
            self.action_clipping.remove_clipping()
        if episode is not None:
            path = '{}/entry_{}'.format(path, episode)
        super(SACAgentCore, self).save_models(path)
        if self.action_clipping is not None:
            self.action_clipping.add_clipping()


class SACAgent(SACAgentCore, Memory):
    def __init__(self, building_info, observation_spaces, action_dim, memory_size=1, **kwargs):
        super(SACAgent, self).__init__(observation_spaces, action_dim, **kwargs)
        if memory_size <= 0:
            raise AttributeError("memory_size should be larger than 0.")
        # Hyper-parameters
        self.memory_size = memory_size

        # Can be used to create different RL agents based on basic building attributes or climate zones
        self.building_info = building_info
        self.n_buildings = len(building_info)

        # State sequence Storage
        self.time_step = 0
        self.init_memory(self.observation_spaces, self.memory_size)

    def rule_based_action(self):
        hour_day = self.time_step
        if 9 <= hour_day <= 21:
            return np.array([[-0.08, -0.08]] * self.n_buildings)
        else:
            return np.array([[0.091, 0.091]] * self.n_buildings)

    def select_action(self, state, **kwargs):
        self.update_state_memory(state)
        if self.time_step < 24:
            return self.rule_based_action()
        else:
            return super(SACAgent, self).select_action(self.state_memory, **kwargs)

    def add_to_buffer(self, state, action, raw_rewards, done, next_state=None, **kwargs):
        # Information contained in the building_info variable can be used to
        # choose the number of buffers and what information goes to each buffer
        self.time_step += 1
        self.update_memory(raw_rewards, self.reward_memory, axis=1, copy=False)
        if self.time_step > self.memory_size:
            state = self.state_memory
            reward = self.reward_memory
            if self.algo is not None:
                if isinstance(done, bool):
                    done = np.array([done] * self.n_buildings)
                if next_state is not None:
                    if next_state.ndim < state.ndim:
                        next_state = np.expand_dims(next_state, 1)
                    next_state = np.concatenate((state[:, next_state.shape[1]:], next_state), axis=1)
                super(SACAgent, self).add_to_buffer(state, action, reward, done,
                                                    next_state=next_state, valid_mask=self.valid_mask,
                                                    **kwargs)
        self.update_agent()


class HSACAlgo(SACAlgo):
    __RequiredTarget__ = {'Actor': False, 'DualCritic': True}

    @property
    def critic_param_groups(self):
        group_list = [self._param_groups['DualCritic']]
        if self._param_groups.get('Encoder') is not None:
            group_list.append(self._param_groups['Encoder'])
        return group_list

    def _process_buffer(self, state, action, reward, done, **info):
        state, action, reward, done, next_state, info = \
            super(HSACAlgo, self)._process_buffer(state, action, reward, done, **info)
        goal = info['goal']
        return (state, goal), action, reward, (next_state, goal), done, info

    def _encode_state(self, state_goal_tuple, **kwargs):
        state, goal = state_goal_tuple
        return torch.cat((self.encoder(state, **kwargs), goal), -1)


class HSACActionMixin:
    model_callbacks: nn.ModuleDict
    state_fn: staticmethod
    encoder: nn.Module

    def _encode_state(self, state, goal=None, **kwargs):
        state = to_tensors(self.state_fn(state), device=get_module_device(self.model_callbacks))
        hidden_state = self.encoder(state.unsqueeze(0)).squeeze(0)
        if goal is not None:
            goal = to_tensors(goal, device=get_module_device(self.model_callbacks))
            hidden_state = torch.cat((hidden_state, goal), -1)
        return hidden_state


class HSACLeader(HSACActionMixin, SACAgentCore):
    pass


class HSACAgent(HSACActionMixin, SACAgentCore):
    __AlgorithmArch___ = HSACAlgo
    __RequiredTarget__ = {'Actor': False, 'DualCritic': True}

    def set_encoder(self, encoder):
        self._model_callbacks['Encoder'] = encoder


class HSACEnsemble(Memory):
    def __init__(self, building_info, observation_spaces, threshold, H, memory_size, reward_fn,
                 agent_kwargs_list, path_base):
        self.HSAC = []  # order: from bottom to top
        # print((agent_kwargs_list[0]))
        self.HSAC.append(HSACAgent(observation_spaces, 2, **(agent_kwargs_list[0])))  # low level
        self.HSAC.append(HSACLeader(observation_spaces, 4, **(agent_kwargs_list[1])))  # top level
        self.HSAC[0].set_encoder(self.HSAC[1].encoder)
        self.goals = [None] * 2
        self.time_step = 0
        self.eval_timestep = 0
        self.reward = 0
        self.threshold = threshold
        self.eval_goal = None
        self.H = H
        self.gamma = .99
        self.n_buildings = len(building_info)

        if memory_size <= 0:
            raise AttributeError("memory_size should be larger than 0.")

        # Hyper-parameters
        self.memory_size = memory_size
        self.observation_spaces = np.stack([obs.sample() for obs in observation_spaces])
        self.init_memory(self.observation_spaces, self.memory_size)

        self.reward_func = reward_fn
        self.writer = SummaryWriter(path_base + '/reward')
        self.cum_rewards = [None for _ in range(self.n_buildings)]
        self.reset()

    def reset(self):
        # Clean Memory & Reset Time Step
        self.time_step = 0
        self.cum_rewards = [0 for _ in range(self.n_buildings)]
        self.reward_memory.fill(0.)
        self.valid_mask.fill(False)

    def check_goal_reward(self, state, goal, action, threshold):
        """
        :param state: (9, s_dim)
        :param goal: (9, 4)
                    goal[0]: [soc1_goal, soc2_goal, act1_limit, act2_limit]
        :param action: (9, 2)
        :param threshold: (9,) deviation from the mean value of the heat demand slider (negative one)
        :return:
        """
        # goal_weight = [1, 1, 1, 1]
        state_ = deepcopy(state)
        goal_ = deepcopy(goal)
        action_ = deepcopy(action)
        act_td_error = abs(goal_[:, 0:2] - action_)  # (9,2)
        soc_td_error = state_[:, -2:] - goal_[:, -2:]  # (9,2)

        # if the td error between real act and goal act are larger than threshold, flag = -1
        flag_act = np.zeros((self.n_buildings, 2))
        flag_act[act_td_error > threshold] = -1

        # if real soc < goal soc, flag = -1; else flag = 0
        flag_soc = np.zeros((self.n_buildings, 2))
        flag_soc[soc_td_error < 0] = -1

        check_flag = np.concatenate((flag_act, flag_soc), -1)  # (9, 4)
        reward = np.sum(- act_td_error + np.minimum(0., soc_td_error), axis=1, keepdims=True)
        return check_flag, reward

    def check_goal(self, state, goal, action, threshold):
        zero_arr = np.zeros((self.n_buildings, 4))
        check_flag, reward = self.check_goal_reward(state, goal, action, threshold)
        if self.time_step % 100 == 0:
            print(check_flag.sum())
            print(reward)
        achieve_goal = (check_flag == zero_arr).all()
        return achieve_goal, reward

    def train(self, env, level, state, goal, log_per_step=1000):  # recurse from top to bottom
        done = None
        # logging updates
        self.goals[level] = goal

        if level > 0:
            #   <================ top level policy ================>
            for time_idx in range(int(8760. / self.H)):  # 365 days
                # Copy current state for later
                old_state = deepcopy(self.state_memory)
                # old_reward = deepcopy(self.reward_memory)
                # State Memory is manipulated in Bottom Level
                sub_goal = self.HSAC[level].select_action(self.state_memory)  # for top level, goal=None
                # Pass sub_goal to lower level
                done, reward_list, state_list = self.train(env, level - 1, state, sub_goal, log_per_step=log_per_step)

                # TODO Add 1~n-step exp.
                next_state = self.update_memory(state_list, deepcopy(old_state), axis=1)
                gamma = self.gamma ** np.array(range(0, reward_list.shape[1]))
                next_gamma = self.gamma ** (reward_list.shape[1])
                cum_reward = np.sum(reward_list * gamma, axis=-1)
                cum_reward = np.expand_dims(cum_reward, 1)
                self.HSAC[level].add_to_buffer(old_state, sub_goal, cum_reward, [done] * self.n_buildings,
                                               gamma=[next_gamma] * self.n_buildings, next_state=next_state)

                # gamma_idx = 0
                # gamma = 1

                # # add N-step transition to buffer, N:[1, H]
                # for j in range(len(reward_list)):  # reward_list: (H, 9,)
                #     cum_reward += gamma * reward_list[j]  # (9,)
                #     gamma_idx += 1
                #     gamma = self.gamma ** gamma_idx
                #     next_state = np.concatenate((old_state[:, j:], state_list[:, :j]))
                #     self.HSAC[level].add_to_buffer(old_state, sub_goal, self.reward_memory, next_state, done,
                #                                    gamma=gamma, valid_mask=self.valid_mask,
                #                                    # No Goal for Top Level)

                # # for hindsight action transition
                # action = next_state

        else:
            #   <================ low level policy ================>
            reward_list = []
            state_list = []
            for time_idx in range(self.H):
                self.update_state_memory(state)  # Get New State
                action = self.HSAC[level].select_action(self.state_memory, goal=goal)
                # take primitive action
                next_state, raw_rewards, done, _ = env.step(action)
                self.update_memory(raw_rewards, self.reward_memory, axis=1, copy=False)

                # TODO Should update after add new replay?
                if self.time_step > self.memory_size:  # low level
                    self.update(0)
                if self.time_step > self.memory_size * self.H:  # top level
                    self.update(1)

                global_reward, _ = self.reward_func(self.reward_memory[:, self.valid_mask],
                                                    self.state_memory[:, self.valid_mask])
                reward_list.append(global_reward)  # (Time, Building)
                state_list.append(next_state)  # (Time, Building, State)

                self.cum_rewards += global_reward  # add up vector

                # logging processed reward
                if self.time_step % log_per_step == 0:
                    print("write reward per " + str(log_per_step) + "step")
                    for idx in range(self.n_buildings):
                        self.writer.add_scalar('building_reward_' + str(idx), global_reward[idx], self.time_step)

                # add transition to buffer
                goal_achieved, goal_reward = self.check_goal(next_state, goal, action, self.threshold)
                next_state_sequence = self.update_memory(next_state, deepcopy(self.state_memory), axis=1)
                if goal_achieved and time_idx == self.H - 1:
                    # Mission Successful and Timeout, State Terminated with gamma = 0
                    gamma = 0
                else:
                    gamma = self.gamma
                self.HSAC[level].add_to_buffer(self.state_memory, action, goal_reward, [done] * self.n_buildings,
                                               next_state=next_state_sequence,
                                               gamma=[gamma] * self.n_buildings, goal=goal)

                # tmp_storage = (self.reward_memory, self.state_memory, self.valid_mask)

                # if time_idx == self.H - 1:  # termination time step
                #     # self, state, goal, action, heat_adv, threshold
                #     goal_achieved, check_goal_rew = self.check_goal(next_state, goal, action, self.threshold)
                #     next_state_sequence = self.update_memory(next_state, deepcopy(self.state_memory), time_dim=1)
                #     if goal_achieved:  # achieve the goal, termination -> gamma = 0
                #         # TODO compatibility
                #         self.HSAC[level].add_to_buffer(self.state_memory, action, global_reward,
                #                                        next_state_sequence, done,
                #                                        gamma=0, goal=goal,
                #                                        time_step=self.time_step, memory_size=self.memory_size)
                #     else:
                #         self.HSAC[level].add_to_buffer(self.state_memory, action, global_reward,
                #                                        next_state_sequence, done,
                #                                        gamma=self.gamma, goal=goal,
                #                                        time_step=self.time_step, memory_size=self.memory_size)
                # else:
                #     _, check_goal_rew = self.check_goal(next_state, goal, action, self.threshold)
                #
                #     self.HSAC[level].add_to_buffer_HSAC(state, (sub_goal, goal), check_goal_rew, next_state,
                #                                         (self.gamma, done), tmp_storage, self.memory_size,
                #                                         self.time_step)

                #   <================ finish one step/transition ================>

                state = next_state
                self.time_step += 1
                if done:
                    print("done!")
                    break

            #   <================ finish H attempts ================>

            # # hindsight goal transition
            # # last transition reward and discount is 0
            # goal_transitions[-1][2] = 0.0
            # goal_transitions[-1][5] = 0.0
            # for transition in goal_transitions:
            #     # last state is goal for all transitions
            #     transition[4] = next_state
            #     self.replay_buffer[level].add(tuple(transition))

            reward_list = np.array(reward_list, dtype=np.float32)
            state_list = np.array(state_list, dtype=np.float32)
            reward_list = reward_list.swapaxes(1, 0)  # (Building, Time)
            state_list = state_list.swapaxes(1, 0)  # (Building, Time, State)

            return done, reward_list, state_list

    def update(self, key):
        self.HSAC[key].update_agent()

    def select_action(self, state, **kwargs):  # used for eval mode
        self.eval_timestep += 1
        low_agent = self.HSAC[0]
        high_agent = self.HSAC[1]
        low_agent.eval()
        high_agent.eval()
        if self.eval_goal is None or self.eval_timestep % self.H:  # change the high-level policy every H time step
            self.eval_goal = high_agent.select_action(state, goal=None)
        return low_agent.select_action(state, goal=self.eval_goal)

    def add_to_buffer(self, *args, **kwargs):
        # No-Op for HSACEnsemble
        pass

    def load_models(self, path, episode):
        for i_level in range(2):
            for key, model in self.HSAC[i_level].model_callbacks.items():
                model.load_state_dict(torch.load('{}/entry_{}/{}_{}.pt'.format(path, episode, key, i_level)))
        print('Models loaded successfully')

    def save_models(self, path, episode):
        save_path = '{}/entry_{}'.format(path, episode)
        make_path(save_path)
        for i_level in range(2):
            if self.HSAC[i_level].action_clipping is not None:
                self.HSAC[i_level].action_clipping.remove_clipping()
            for key, model in self.HSAC[i_level].model_callbacks.items():
                torch.save(model.state_dict(), '{}/{}_{}.pt'.format(save_path, key, i_level))
            if self.HSAC[i_level].action_clipping is not None:
                self.HSAC[i_level].action_clipping.add_clipping()
        print('Models saved successfully')
