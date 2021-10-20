import numpy as np
import random
from collections import deque
from copy import deepcopy


class MemoryBuffer:

    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        self.maxSize = size
        self.transaction_structure = None
        self.keywords = None

    def __len__(self):
        return len(self.buffer)

    def sample(self, count):
        """
        samples a random batch from the replay memory buffer
        :param count: batch size
        :return: batch: list of (numpy array)
        """
        batch = deepcopy(random.sample(self.buffer, count))
        batch = list(zip(*batch))
        args, kwargs = batch[:-1], batch[-1]
        args = [np.stack(arr) for arr in args]
        out = dict(zip(self.keywords, list(zip(*[d.values() for d in kwargs]))))
        for k, v in out.items():
            out[k] = np.stack(v)
        return (*args, out)

    def append(self, *args, **kwargs):
        args = [deepcopy(arg) for arg in args]
        ordered_kwargs = {}
        for k, v in sorted(kwargs.items()):
            ordered_kwargs[k] = v

        if self.transaction_structure is None:
            self.transaction_structure = [np.shape(arr) for arr in args]
        if self.keywords is None:
            self.keywords = ordered_kwargs.keys()

        assert np.all([np.shape(cur) == shape for cur, shape in zip(args, self.transaction_structure)]), \
            "Oops, transaction is not consistent with the structure,\n" \
            "expect shape {},\n" \
            "but got {} instead".format([np.shape(cur) for cur in args], self.transaction_structure)
        assert np.all([self.keywords == ordered_kwargs.keys()]), \
            "Oops, keywords is not consistent with the keywords previously seen,\n" \
            "expect keywords {},\n" \
            "but got {} instead".format(self.keywords, ordered_kwargs.keys())

        self.buffer.append((*args, ordered_kwargs))


class _LegacyMemoryBuffer:

    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        self.maxSize = size
        self.transaction_structure = None

    def sample(self, count):
        """
        samples a random batch from the replay memory buffer
        :param count: batch size
        :return: batch (numpy array)
        """
        batch = deepcopy(random.sample(self.buffer, count))
        batch = [np.array(arr) for arr in zip(*batch)]

        return batch

    def __len__(self):
        return len(self.buffer)

    def append(self, *args, **kwargs):
        """
        adds a particular transaction in the memory buffer,
        in most cases the structure will be (state, action, reward, next_state, terminal)
        allowing customize transaction,
        :param args: transaction tuple that has same dimension along each element
        :return:
        """
        args = [deepcopy(arg) for arg in args]
        if len(kwargs) > 0:
            _, sorted_values = zip(*sorted(kwargs.items()))
            sorted_values = [deepcopy(arg) for arg in sorted_values]
            args = list(args) + sorted_values
        if self.transaction_structure is None:
            self.transaction_structure = [np.shape(arr) for arr in args]
        else:
            assert np.all([np.shape(cur) == shape for cur, shape in zip(args, self.transaction_structure)]), \
                "Oops, transaction is not consistent with the structure,\n" \
                "expect shape {},\n" \
                "but got {} instead".format([np.shape(cur) for cur in args], self.transaction_structure)
        self.buffer.append(args)


