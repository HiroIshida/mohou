import copy
import logging
import uuid
from dataclasses import dataclass
from typing import Generic, List, Optional, Tuple, TypeVar

import matplotlib.pyplot as plt
import torch
import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from mohou.model import LossDict, ModelBase, ModelT, average_loss_dict

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    batch_size: int = 200
    learning_rate: float = 0.001
    n_epoch: int = 1000


TrainCacheT = TypeVar('TrainCacheT', bound='TrainCache')


class TrainCache(Generic[ModelT]):
    project_name: str
    epoch: int
    train_loss_dict_seq: List[LossDict]
    validate_loss_dict_seq: List[LossDict]
    best_model: ModelT
    latest_model: ModelT
    cache_postfix: str

    def __init__(self, project_name: str, cache_postfix: Optional[str] = None):
        if cache_postfix is None:
            cache_postfix = ""
        self.project_name = project_name
        self.train_loss_dict_seq = []
        self.validate_loss_dict_seq = []
        self.cache_postfix = cache_postfix
        self.uuid_str = str(uuid.uuid4())[-6:]

    def on_startof_epoch(self, epoch: int):
        logger.info('new epoch: {}'.format(epoch))
        self.epoch = epoch

    def on_train_loss(self, loss_dict: LossDict, epoch: int):
        self.train_loss_dict_seq.append(loss_dict)
        logger.info('train_total_loss: {}'.format(loss_dict.total().item()))

    def on_validate_loss(self, loss_dict: LossDict, epoch: int):
        self.validate_loss_dict_seq.append(loss_dict)
        logger.info('validate_total_loss: {}'.format(loss_dict.total().item()))

    def on_endof_epoch(self, model: ModelT, epoch: int):
        model = copy.deepcopy(model)
        model.to(torch.device('cpu'))
        self.latest_model = model

        totals = [dic.total().item() for dic in self.validate_loss_dict_seq]
        min_loss = min(totals)
        if(totals[-1] == min_loss):
            self.best_model = model
            logger.info('model is updated')
        """
        postfix = '-'.join([self.cache_postfix, model.hash_value, self.uuid_str])
        dump_pickled_data(self, self.project_name,
                self.best_model.__class__.__name__, postfix)
        """

    def visualize(self, fax: Optional[Tuple] = None):
        fax = plt.subplots() if fax is None else fax
        fig, ax = fax
        train_loss_seq = [dic['total'] for dic in self.train_loss_dict_seq]
        valid_loss_seq = [dic['total'] for dic in self.validate_loss_dict_seq]
        ax.plot(train_loss_seq)
        ax.plot(valid_loss_seq)
        ax.set_yscale('log')
        ax.legend(['train', 'valid'])

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
        model: ModelBase,
        dataset_train: Dataset,
        dataset_validate: Dataset,
        tcache: TrainCache,
        config: TrainConfig = TrainConfig()) -> None:

    logger.info('train start with config: {}'.format(config))

    def move_to_device(sample):
        if isinstance(sample, torch.Tensor):
            return sample.to(model.device)
        elif isinstance(sample, list):  # NOTE datalodaer return list type not tuple
            return tuple([e.to(model.device) for e in sample])
        else:
            raise RuntimeError

    train_loader = DataLoader(
        dataset=dataset_train, batch_size=config.batch_size, shuffle=True)
    validate_loader = DataLoader(
        dataset=dataset_validate, batch_size=config.batch_size, shuffle=True)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    model.put_on_device()
    for epoch in tqdm.tqdm(range(config.n_epoch)):
        tcache.on_startof_epoch(epoch)

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
