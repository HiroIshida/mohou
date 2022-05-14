import copy
import logging
import uuid
from dataclasses import dataclass
from typing import Generic, List, Optional, Tuple, Type, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset
from torch.optim import Adam
from torch.utils.data import DataLoader

from mohou.file import dump_object, load_objects
from mohou.model import LossDict, ModelBase, ModelT, average_loss_dict
from mohou.utils import split_with_ratio

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    batch_size: int = 200
    valid_data_ratio: float = 0.1
    learning_rate: float = 0.001
    n_epoch: int = 1000


TrainCacheT = TypeVar('TrainCacheT', bound='TrainCache')


class TrainCache(Generic[ModelT]):
    project_name: str
    epoch: int
    train_loss_dict_seq: List[LossDict]
    validate_loss_dict_seq: List[LossDict]
    min_validate_loss: float
    best_model: Optional[ModelT]
    latest_model: ModelT
    file_uuid: str
    dump_always: bool

    def __init__(self, project_name: str, dump_always: bool = False):
        self.project_name = project_name
        self.train_loss_dict_seq = []
        self.validate_loss_dict_seq = []
        self.file_uuid = str(uuid.uuid4())[-6:]
        self.dump_always = dump_always
        self.best_model = None

    def on_startof_epoch(self, epoch: int, dataset: Dataset):
        logger.info('new epoch: {}'.format(epoch))
        self.epoch = epoch

    def on_train_loss(self, loss_dict: LossDict, epoch: int):
        self.train_loss_dict_seq.append(loss_dict)
        logger.info('train loss => {}'.format(loss_dict))

    def on_validate_loss(self, loss_dict: LossDict, epoch: int):
        self.validate_loss_dict_seq.append(loss_dict)
        logger.info('validate loss => {}'.format(loss_dict))

    def on_endof_epoch(self, model: ModelT, epoch: int):
        model = copy.deepcopy(model)
        model.to(torch.device('cpu'))
        self.latest_model = model

        totals = [dic.total().item() for dic in self.validate_loss_dict_seq]
        min_loss_sofar = min(totals)

        update_model = (totals[-1] == min_loss_sofar)
        if update_model:
            self.min_validate_loss = min_loss_sofar
            self.best_model = model
            logger.info('model is updated')

        postfix = self.best_model.__class__.__name__ + '-' + self.file_uuid

        if update_model or self.dump_always:
            dump_object(self, self.project_name, postfix)

    def visualize(self, fax: Optional[Tuple] = None):
        fax = plt.subplots() if fax is None else fax
        fig, ax = fax
        train_loss_seq = [dic.total() for dic in self.train_loss_dict_seq]
        valid_loss_seq = [dic.total() for dic in self.validate_loss_dict_seq]
        ax.plot(train_loss_seq)
        ax.plot(valid_loss_seq)
        ax.set_yscale('log')
        ax.legend(['train', 'valid'])

    @classmethod
    def load(cls, project_name: str, model_type: Type[ModelT]) -> 'TrainCache[ModelT]':
        postfix = model_type.__name__
        tcache_list = load_objects(TrainCache, project_name, postfix)
        assert len(tcache_list) > 0, 'No train cache found for {}'.format(model_type.__name__)
        min_validate_loss_list = []

        for tcache in tcache_list:
            if hasattr(tcache, 'min_validate_loss'):
                min_validate_loss_list.append(tcache.min_validate_loss)
            else:
                from warnings import warn
                warn('for backward compatibility. will be removed', DeprecationWarning)
                totals = [dic.total().item() for dic in tcache.validate_loss_dict_seq]
                min_validate_loss_list.append(min(totals))

        idx_min_validate = np.argmin(min_validate_loss_list)
        return tcache_list[idx_min_validate]

    """
    @classmethod
    def load(cls: Type[TrainCacheT], project_name: str, model_type: type,
            cache_postfix: Optional[str]=None) -> TrainCacheT:
        # requiring "model_type" seems redundant but there is no way to
        # use info of ModelT from @classmethod

        # If multiple caches are found, choose best one respect to valid loss
        tcache_list = load_pickled_data(project_name, cls, model_type.__name__, cache_postfix)
        loss_list = [tcache.validate_loss_dict_seq[-1]['total'] for tcache in tcache_list]
        idx = np.argmin(loss_list)
        return tcache_list[idx]

    # TODO: probably has better design ...
    @classmethod
    def load_multiple(cls: Type[TrainCacheT], project_name: str, model_type: type,
            cache_postfix: Optional[str]=None) -> List[TrainCacheT]:
        # requiring "model_type" seems redundant but there is no way to
        # use info of ModelT from @classmethod
        data_list = load_pickled_data(project_name, cls, model_type.__name__, cache_postfix)
        assert len(data_list) > 1, "data_list has {} elements.".format(len(data_list))
        return data_list
    """


def train(
        tcache: TrainCache,
        dataset: Dataset,
        model: Optional[ModelBase] = None,
        config: TrainConfig = TrainConfig()) -> None:

    logger.info('train start with config: {}'.format(config))

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

    train_loader = DataLoader(
        dataset=dataset_train, batch_size=config.batch_size, shuffle=True)
    validate_loader = DataLoader(
        dataset=dataset_validate, batch_size=config.batch_size, shuffle=True)
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

            loss_dict.detach_clone()
            train_ld_list.append(loss_dict)
            optimizer.step()

        train_ld_mean = average_loss_dict(train_ld_list)
        tcache.on_train_loss(train_ld_mean, epoch)

        model.eval()
        validate_ld_list: List[LossDict] = []
        for samples in validate_loader:
            samples = move_to_device(samples)
            loss_dict = model.loss(samples)
            loss_dict.detach_clone()
            validate_ld_list.append(loss_dict)

        validate_ld_mean = average_loss_dict(validate_ld_list)
        tcache.on_validate_loss(validate_ld_mean, epoch)

        tcache.on_endof_epoch(model, epoch)
