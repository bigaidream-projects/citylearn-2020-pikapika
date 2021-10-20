import os
from collections import namedtuple

from torch import load


def load_file(file_path, strict=True):
    try:
        result = load(file_path)
    except (not strict and FileNotFoundError):
        RuntimeWarning('File {} is not found, will return None instead'.format(file_path))
        return None
    return result


def load_param_dict(self, param_dict, strict=True):
    r"""Copies parameters from :attr:`param_dict` into this module
    Note that only non-empty params with key existed in self.__dict__.keys() will be overwritten.

    [reference] torch.nn.Module.load_state_dict function

    Args:
        self (object): object needs to load param dict.
        param_dict (dict): a dict containing parameters of self
        strict (bool, optional): whether to strictly enforce that all the keys
            in :attr:`param_dict` existed the keys returned by this module's
            :meth:`__dict__.keys()` function. Default: ``True``

    Returns:
        ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
            * **missing_keys** is a list of str containing the missing keys
            * **unexpected_keys** is a list of str containing the unexpected keys
    """
    if param_dict is None:
        if strict:
            RuntimeError("Trying to set global params with empty param_dict")
        else:
            param_dict = {}

    error_msgs = []
    local_dict = {name: param for name, param in param_dict.items() if param is not None}
    unexpected_keys = list(set(local_dict.keys()) - set(self.__dict__.keys()))
    missing_keys = list(set(self.__dict__.keys()) - set(local_dict.keys()))
    self.__dict__.update(
        {name: param for name, param in local_dict.items() if name in self.__dict__.keys()}
    )

    if strict:
        if len(unexpected_keys) > 0:
            error_msgs.insert(
                0, 'Unexpected key(s) in state_dict: {}. '.format(
                    ', '.join('"{}"'.format(k) for k in unexpected_keys)))
        if len(missing_keys) > 0:
            error_msgs.insert(
                0, 'Missing key(s) in state_dict: {}. '.format(
                    ', '.join('"{}"'.format(k) for k in missing_keys)))

    if len(error_msgs) > 0:
        raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                           self.__class__.__name__, "\n\t".join(error_msgs)))

    return _IncompatibleKeys(missing_keys, unexpected_keys)


def get_output_folder(parent_dir, env_name):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except OSError:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    os.makedirs(parent_dir, exist_ok=True)
    return parent_dir


def make_path(path):
    try:
        os.makedirs(path)
    except OSError:
        pass


_IncompatibleKeys = namedtuple('IncompatibleKeys', ['missing_keys', 'unexpected_keys'])