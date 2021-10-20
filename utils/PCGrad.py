import torch
from torch import nn


class PCGrad:
    def __init__(self):
        pass

    def optimize(self, grads):
        """

        :param grads: FloatTensor (Chunk, *,
        :return:
        """
        # Iterate along chunk
        # Compare with other chunk #Sequentially#, random order along chunk
        # Check Direction
        # If conflict, Update with Projected Grad
        # Repeat until all processed
        # Return mean/sum grad
        pass
