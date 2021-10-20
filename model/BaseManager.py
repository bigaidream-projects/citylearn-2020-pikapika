from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch
from torch import nn

from utils.io import make_path


class MLFeatures:
    __RequiredTarget__: OrderedDict
    __IgnoreSuffix__: str = '__'
    _model_callbacks: nn.ModuleDict
    _model_kwargs: OrderedDict
    _model_type_callbacks: OrderedDict
    _optim_callbacks: OrderedDict
    _param_groups: OrderedDict

    @property
    def model_callbacks(self):
        return self._model_callbacks


class ModelCallbacks(MLFeatures):
    def init_model_callbacks(self, model_kwargs, **kwargs):
        # Model Initialization
        model_names = model_kwargs.keys()
        model_types, init_kwargs = zip(*model_kwargs.values())
        missing_callbacks = set(self.__RequiredTarget__.keys()) - set(model_names)
        if len(missing_callbacks) > 0:
            RuntimeError('Model initialization failed, missing model_callbacks {}'.format(missing_callbacks))
        self._model_callbacks = nn.ModuleDict()
        self._model_kwargs = OrderedDict(zip(model_names, init_kwargs))
        self._model_type_callbacks = OrderedDict(zip(model_names, model_types))

    def _init_models(self, update_keys=None):
        if update_keys is None:
            update_keys = self._model_callbacks.keys()
        update_list = []
        for name in update_keys:
            model_arch = self._model_type_callbacks[name]
            init_kwargs = self._model_kwargs.get(name)
            if init_kwargs is None:
                init_kwargs = {}
            model = model_arch(**init_kwargs)
            update_list.append((name, model))
            if self.__RequiredTarget__[name]:
                target_model = deepcopy(model)
                for p in target_model.parameters():
                    p.requires_grad = False
                update_list.append((name + self.__IgnoreSuffix__, target_model))
        self._model_callbacks.update(update_list)
        del update_list

    def train(self, mode=True):
        self._model_callbacks.train(mode)

    def eval(self):
        self.train(mode=False)

    def load_models(self, path, **kwargs):
        for key, model in self._model_callbacks.items():
            model.load_state_dict(torch.load('{}/{}.pt'.format(path, key)))
        print('Models loaded successfully')

    def save_models(self, path, **kwargs):
        make_path(path)
        for key, model in self._model_callbacks.items():
            torch.save(model.state_dict(), '{}/{}.pt'.format(path, key))
        print('Models saved successfully')


class ParamGroupCallbacks(MLFeatures):
    def init_param_groups(self, unique_optim_kwargs=None):
        if unique_optim_kwargs:
            self._param_groups = OrderedDict(unique_optim_kwargs)
        else:
            self._param_groups = OrderedDict()
        for key, model in self._model_callbacks.items():
            if key.endswith(self.__IgnoreSuffix__):
                continue
            self._param_groups.setdefault(key, {})  # Ensure every required key has a dict
            self._param_groups[key].update(params=model.parameters())


class Memory:
    # TODO Make it into Mixin
    @classmethod
    def init_memory(cls, observation_spaces, memory_size, **kwargs):
        # Scene Memory for Agent - (Building, Time, Std_Observation)
        # Reward Memory for Agent - (Building, Time)
        state_example = observation_spaces[:, None].repeat(memory_size, 1)
        lead_dim = state_example.shape[:-1]
        cls.state_memory = np.zeros_like(state_example, dtype=np.float32)  # Save as double-precision for preprocess
        cls.reward_memory = np.zeros(lead_dim, dtype=np.float32)  # TODO Reward Dim?
        cls.valid_mask = np.zeros(lead_dim, dtype=np.bool)

    def update_state_memory(self, state):
        # separate adding state memory from _encoder_state (which is called by select_action)
        axis = 1
        state, time_length = self._check_time_dim(state, self.state_memory, axis=axis)
        self.update_memory(state, self.state_memory, axis=axis, copy=False)
        self.update_memory(np.ones(time_length, dtype=np.bool), self.valid_mask, axis=axis, copy=False)

    @staticmethod
    def update_memory(data: np.ndarray, memory: np.ndarray, axis=0, copy=True):
        # TODO
        if copy:
            memory = np.copy(memory)
        data, time_length = Memory._check_time_dim(data, memory, axis=axis)
        old_memory, new_memory = np.split(memory, [-time_length], axis=axis)
        old_memory[:] = np.split(memory, [time_length], axis=axis)[1]
        new_memory[:] = data

        return memory

    @staticmethod
    def _check_time_dim(data, memory, axis=0):
        data = np.asarray(data)
        if data.ndim == memory.ndim-1:
            data = np.copy(data)
            data = np.expand_dims(data, axis)
        assert data.ndim == memory.ndim
        return data, data.shape[axis]
