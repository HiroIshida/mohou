import copy
import logging
import uuid
from dataclasses import dataclass
from typing import Generic, List, Optional, Tuple, Type, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

import mohou
from mohou.file import dump_object, load_objects
from mohou.model import (
    FloatLossDict,
    LossDict,
    ModelBase,
    ModelConfigBase,
    ModelT,
    average_float_loss_dict,
)
from mohou.utils import log_package_version_info, log_text_with_box, split_with_ratio

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    batch_size: int = 200
    valid_data_ratio: float = 0.1
    learning_rate: float = 0.001
    n_epoch: int = 1000


TrainCacheT = TypeVar("TrainCacheT", bound="TrainCache")


class TrainCache(Generic[ModelT]):
    project_name: str
    epoch: int
    train_loss_dict_seq: List[FloatLossDict]
    validate_loss_dict_seq: List[FloatLossDict]
    min_validate_loss: float
    best_model: Optional[ModelT]
    latest_model: ModelT
    file_uuid: str

    def __init__(self, project_name: str):
        self.project_name = project_name
        self.train_loss_dict_seq = []
        self.validate_loss_dict_seq = []
        self.file_uuid = str(uuid.uuid4())[-6:]
        self.best_model = None

    def on_startof_epoch(self, epoch: int, dataset: Dataset):
        logger.info("new epoch: {}".format(epoch))
        self.epoch = epoch

    def on_train_loss(self, loss_dict: FloatLossDict, epoch: int):
        self.train_loss_dict_seq.append(loss_dict)
        logger.info("train loss => {}".format(loss_dict))

    def on_validate_loss(self, loss_dict: FloatLossDict, epoch: int):
        self.validate_loss_dict_seq.append(loss_dict)
        logger.info("validate loss => {}".format(loss_dict))

    @staticmethod
    def get_file_postfix(model: ModelT, file_uuid: Optional[str]) -> str:
        class_name = model.__class__.__name__
        config_hash = model.hash_value

        postfix = "{}-{}".format(class_name, config_hash)
        if file_uuid is not None:
            postfix += "-{}".format(file_uuid)
        return postfix

    def on_endof_epoch(self, model: ModelT, epoch: int):
        model = copy.deepcopy(model)
        model.to(torch.device("cpu"))
        self.latest_model = model

        totals = [dic.total() for dic in self.validate_loss_dict_seq]
        min_loss_sofar = min(totals)

        update_model = totals[-1] == min_loss_sofar
        if update_model:
            self.min_validate_loss = min_loss_sofar
            self.best_model = model
            logger.info("model is updated")
            self.dump()

    def dump(self):
        assert self.best_model is not None
        postfix = self.get_file_postfix(self.best_model, self.file_uuid)
        dump_object(self, self.project_name, postfix)

    def visualize(self, fax: Optional[Tuple] = None):
        fax = plt.subplots() if fax is None else fax
        fig, ax = fax
        train_loss_seq = [dic.total() for dic in self.train_loss_dict_seq]
        valid_loss_seq = [dic.total() for dic in self.validate_loss_dict_seq]
        ax.plot(train_loss_seq)
        ax.plot(valid_loss_seq)
        ax.set_yscale("log")
        ax.legend(["train", "valid"])

    @staticmethod
    def _choose_lowest_validation_loss(
        tcache_list: List["TrainCache[ModelT]"],
    ) -> "TrainCache[ModelT]":
        min_validate_loss_list = []
        for tcache in tcache_list:
            if hasattr(tcache, "min_validate_loss"):
                min_validate_loss_list.append(tcache.min_validate_loss)
            else:
                from warnings import warn

                warn("for backward compatibility. will be removed", DeprecationWarning)
                totals = [dic.total().item() for dic in tcache.validate_loss_dict_seq]
                min_validate_loss_list.append(min(totals))

        idx_min_validate = np.argmin(min_validate_loss_list)
        return tcache_list[idx_min_validate]

    @classmethod
    def load(
        cls,
        project_name: Optional[str],
        model_type: Type[ModelT],
        model_config: Optional[ModelConfigBase] = None,
    ) -> "TrainCache[ModelT]":

        # TODO(HiroIshida): get_file_postfix function ????
        postfix = model_type.__name__
        if model_config is not None:
            postfix += "-{}".format(model_config.hash_value)

        tcache_list = load_objects(TrainCache, project_name, postfix)
        return cls._choose_lowest_validation_loss(tcache_list)


def train(
    tcache: TrainCache,
    dataset: Dataset,
    model: Optional[ModelBase] = None,
    config: TrainConfig = TrainConfig(),
) -> None:

    log_package_version_info(logger, mohou)
    log_text_with_box(logger, "train log")
    logger.info("train start with config: {}".format(config))

    if model is None:
        assert tcache.best_model is not None
        model = tcache.best_model

    def move_to_device(sample):
        if isinstance(sample, torch.Tensor):
            return sample.to(model.device)
        elif isinstance(sample, list):  # NOTE datalodaer return list type not tuple
            return tuple([e.to(model.device) for e in sample])
        else:
            raise RuntimeError

    dataset_train, dataset_validate = split_with_ratio(dataset, config.valid_data_ratio)

    train_loader = DataLoader(dataset=dataset_train, batch_size=config.batch_size, shuffle=True)
    validate_loader = DataLoader(
        dataset=dataset_validate, batch_size=config.batch_size, shuffle=True
    )
    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    model.put_on_device()
    for epoch in tqdm.tqdm(range(config.n_epoch)):
        tcache.on_startof_epoch(epoch, dataset)

        model.train()
        train_ld_list: List[LossDict] = []
        for samples in train_loader:
            optimizer.zero_grad()
            samples = move_to_device(samples)
            loss_dict = model.loss(samples)
            loss_dict.total().backward()

            fld = loss_dict.to_float_lossdict()
            train_ld_list.append(fld)
            optimizer.step()

        train_ld_mean = average_float_loss_dict(train_ld_list)
        tcache.on_train_loss(train_ld_mean, epoch)

        model.eval()
        validate_ld_list: List[LossDict] = []
        for samples in validate_loader:
            samples = move_to_device(samples)
            loss_dict = model.loss(samples)
            fld = loss_dict.to_float_lossdict()
            validate_ld_list.append(fld)

        validate_ld_mean = average_float_loss_dict(validate_ld_list)
        tcache.on_validate_loss(validate_ld_mean, epoch)

        tcache.on_endof_epoch(model, epoch)
