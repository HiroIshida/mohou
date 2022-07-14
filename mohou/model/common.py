import copy
import logging
import operator
from abc import ABC, abstractmethod
from functools import reduce
from typing import Any, Dict, Generic, List, Optional, TypeVar

import numpy as np
import torch
import torch.nn as nn

from mohou.types import HashableMixin
from mohou.utils import detect_device

logger = logging.getLogger(__name__)


class FloatLossDict(Dict[str, float]):
    def total(self) -> float:
        return reduce(operator.add, self.values())

    def __str__(self) -> str:
        def to_exp_notation(val: float) -> str:
            return np.format_float_scientific(val, precision=3, exp_digits=2)

        string = "total: {}".format(to_exp_notation(self.total()))
        for k, v in self.items():
            string += ", {}: {}".format(k, to_exp_notation(v))
        return string


def average_float_loss_dict(dicts: List[FloatLossDict]) -> FloatLossDict:
    dict_new = copy.deepcopy(dicts[0])
    for key in dict_new.keys():
        dict_new[key] = np.mean([d[key] for d in dicts]).item()
    return dict_new


class LossDict(Dict[str, torch.Tensor]):
    """A dictionary containing loss info.

    For example in VAE, loss is sum of reconstruction loss and regularization loss.
    Instead of returning the sum value directrly, returning LossDict in this case
    is beneficiall for debugging and
    visualization purposes.
    """

    def total(self) -> torch.Tensor:
        return reduce(operator.add, self.values())

    def to_float_lossdict(self) -> FloatLossDict:
        fld = FloatLossDict()
        for key in self.keys():
            fld[key] = self[key].detach().clone().cpu().item()
        return fld


class ModelConfigBase(HashableMixin):
    pass


ModelConfigT = TypeVar("ModelConfigT", bound=ModelConfigBase)


class ModelBase(nn.Module, Generic[ModelConfigT], ABC):
    config: ModelConfigT
    device: torch.device

    def __init__(self, config: ModelConfigT, device: Optional[torch.device] = None):
        super().__init__()
        self._setup_from_config(config)

        if device is None:
            device = detect_device()

        self.device = device
        self.config = config
        logger.info("model name: {}".format(self.__class__.__name__))
        logger.info("model config: {}".format(config))
        logger.info("hash value of config: {}".format(self.hash_value))
        logger.info("model is initialized")

    def put_on_device(self):
        self.to(self.device)

    @property
    def hash_value(self) -> str:
        return self.config.hash_value

    @abstractmethod
    def _setup_from_config(self, config: ModelConfigT) -> None:
        pass

    @abstractmethod
    def loss(self, sample: Any) -> LossDict:
        pass

    @property
    def n_parameter(model):
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            params = parameter.numel()
            total_params += params
        return total_params


ModelT = TypeVar("ModelT", bound=ModelBase)
