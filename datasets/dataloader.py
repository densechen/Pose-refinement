'''
Descripttion: densechen@foxmail.com
version: 0.0
Author: Dense Chen
Date: 1970-01-01 08:00:00
LastEditors: Dense Chen
LastEditTime: 2020-08-12 20:44:23
'''
import torch
import torch.utils.data._utils.collate as collate
from prefetch_generator import BackgroundGenerator
from pytorch3d.structures.meshes import Meshes, join_meshes_as_batch
from torch.utils.data import DataLoader


def meshes_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if collate.np_str_obj_array_pattern.search(
                    elem.dtype.str) is not None:
                raise TypeError(
                    collate.default_collate_err_msg_format.format(elem.dtype))

            return meshes_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, collate.int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, collate.string_classes):
        return batch
    elif isinstance(elem, collate.container_abcs.Mapping):
        return {key: meshes_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(meshes_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collate.container_abcs.Sequence):
        transposed = zip(*batch)
        return [meshes_collate(samples) for samples in transposed]
    elif isinstance(elem, Meshes):
        return join_meshes_as_batch(batch)

    raise TypeError(collate.default_collate_err_msg_format.format(elem_type))


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
