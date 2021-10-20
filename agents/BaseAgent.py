from abc import abstractmethod

import numpy as np

from model.BaseManager import Memory


class BaseAgent:
    @abstractmethod
    def select_action(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def add_to_buffer(self, *other, **kwargs):
        raise NotImplementedError

