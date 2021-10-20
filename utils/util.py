import random

import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

from torch.backends import cudnn

USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
# torch.set_default_tensor_type(FLOAT)


def clone_modules(module, num):
    return nn.ModuleList([deepcopy(module) for _ in range(num)])


def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()


def as_tensor(data, requires_grad=False, tensor_type=FLOAT):
    return torch.as_tensor(data).requires_grad_(requires_grad).type(tensor_type)


def to_tensor(data, requires_grad=False, tensor_type=FLOAT):
    if isinstance(data, torch.Tensor):
        return data.clone().detach().requires_grad_(requires_grad).type(tensor_type)
    else:
        return torch.tensor(data, requires_grad=requires_grad).type(tensor_type)


def elementwise_clamp(data, lower_bound, upper_bound):
    return torch.max(torch.min(data, upper_bound), lower_bound)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def init_seed(seed=None):
    if seed is None:
        seed = random.randint(1, 10000)
    print("Random Seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = True
    return seed


def get_module_device(module):
    for p in module.parameters():
        return p.device
    return None


def set_module_device(model, cuda_idx=None):
    if torch.cuda.is_available() and cuda_idx is not None:
        device = torch.device('cuda', cuda_idx)
        model.to(device)
        # if torch.cuda.device_count() > 1:
        #     device_ids = list(range(torch.cuda.device_count()))
        #     device_ids.pop(device_ids.index(CUDA_IDX))
        #     device_ids = [device.index] + device_ids
        #     model = DataParallel(model, device_ids=device_ids, output_device=CUDA_IDX, dim=0)
    else:
        device = torch.device('cpu')
    return model, device


def cat_oa(obs, act):
    return torch.cat((obs, act), -1)


def to_tensors(*data_list, requires_grad=False, device='cpu', dtype=torch.float32):
    result = [torch.tensor(data, requires_grad=requires_grad, device=device, dtype=dtype)
              for data in data_list]
    if len(result) == 1:
        result = result[0]
    return result