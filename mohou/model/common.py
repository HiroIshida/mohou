from abc import ABC, abstractmethod
import copy
import pickle
import hashlib
from functools import reduce
import operator
import torch
import torch.nn as nn

from typing import Generic, TypeVar, Any, Dict, Optional, List

import logging
logger = logging.getLogger(__name__)

class LossDict(Dict[str, torch.Tensor]):
    """
    A dictionary containing loss info. 
    For example in VAE, loss is sum of reconstruction loss and
    regularization loss. Instead of returning the sum value directrly,
    returning LossDict in this case is beneficiall for debugging and
    visualization purposes.
    """

    def total(self) -> torch.Tensor: 
        return reduce(operator.add, self.values())

    def detach_clone(self) -> None:
        for key in self.keys():
            val = self[key].detach().clone().cpu()
            self[key] = val

def average_loss_dict(dicts: List[LossDict]):
    dict_new = copy.deepcopy(dicts[0])
    for key in dict_new.keys():
        dict_new[key] = torch.mean(torch.stack([d[key] for d in dicts]))
    return dict_new

class ModelConfigBase:

    @property
    def hash_value(self) -> str:
        data_pickle = pickle.dumps(self)
        data_md5 = hashlib.md5(data_pickle).hexdigest()
        return data_md5[:7]

ModelConfigT = TypeVar('ModelConfigT', bound=ModelConfigBase)

class ModelBase(nn.Module, Generic[ModelConfigT]):
    device: torch.device
    config: ModelConfigT

    def __init__(self, device: torch.device, config: ModelConfigT):
        super().__init__()
        self._create_layers(config)
        self.device = device
        self.config = config
        logger.info('model name: {}'.format(self.__class__.__name__))
        logger.info('model config: {}'.format(config))
        logger.info('model is initialized')

    def put_on_device(self): self.to(self.device)

    @property
    def hash_value(self) -> str: 
        return self.config.hash_value

    @abstractmethod
    def _create_layers(self, config: ModelConfigT) -> None: ...
    # create and set layers

    @abstractmethod
    def loss(self, sample : Any) -> LossDict: ...

ModelT = TypeVar('ModelT', bound=ModelBase)
