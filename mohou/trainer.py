import copy
import logging
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generic, List, Optional, Tuple, Type, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

import mohou
from mohou.model import FloatLossDict, ModelConfigBase, ModelT, average_float_loss_dict
from mohou.utils import log_package_version_info, log_text_with_box, split_with_ratio

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    batch_size: int = 200
    valid_data_ratio: float = 0.1
    learning_rate: float = 0.001
    n_epoch: int = 1000


TrainCacheT = TypeVar("TrainCacheT", bound="TrainCache")


@dataclass
class TrainCache(Generic[ModelT]):
    best_model: ModelT
    train_loss_dict_seq: List[FloatLossDict]
    validate_loss_dict_seq: List[FloatLossDict]
    file_uuid: str

    @classmethod
    def from_model(cls, model: ModelT) -> "TrainCache[ModelT]":
        file_uuid = str(uuid.uuid4())[-6:]
        return cls(model, [], [], file_uuid)

    @property
    def min_validate_loss(self) -> float:
        totals = [dic.total() for dic in self.validate_loss_dict_seq]
        min_loss_sofar = min(totals)
        return min_loss_sofar

    @staticmethod
    def train_result_base_path(project_path: Path) -> Path:
        return project_path / "train_result"

    def train_result_path(self, project_path: Path):
        base_path = self.train_result_base_path(project_path)
        class_name = self.best_model.__class__.__name__
        config_hash = self.best_model.config.hash_value
        result_path = base_path / "{}-{}-{}".format(class_name, config_hash, self.file_uuid)
        return result_path

    @classmethod
    def filter_result_paths(
        cls,
        project_path: Path,
        model_type: Optional[Type[ModelT]],
        model_config: Optional[ModelConfigBase],
    ) -> List[Path]:
        """filter train cache path. see filter_predicate for the logic."""

        def filter_predicate(path: Path):
            name = path.name

            if model_type is None:
                # if not specified, always pass
                return True
            else:
                assert model_type is not None  # this is for mypy
                if not name.startswith(model_type.__name__):
                    # if specified, name must be start with model_type.__name__
                    return False
                else:
                    if model_config is None:
                        # if not specified, always pass
                        return True
                    else:
                        return model_config.hash_value in name

        ps = filter(filter_predicate, cls.train_result_base_path(project_path).iterdir())
        return list(ps)

    @staticmethod
    def dump_flds_as_npz_dict(flds: List[FloatLossDict]) -> Dict:
        kwargs = {}
        for key in flds[0].keys():
            values = np.array([fld[key] for fld in flds])
            kwargs[key] = values
        return kwargs

    @staticmethod
    def load_flds_from_npz_dict(npz_dict: Dict) -> List[FloatLossDict]:
        keys = list(npz_dict.keys())
        flds: List[FloatLossDict] = []
        n_seqlen = len(npz_dict[keys[0]])
        for i in range(n_seqlen):
            fld = FloatLossDict({k: npz_dict[k][i] for k in keys})
            flds.append(fld)
        return flds

    def update_and_save(
        self,
        model: ModelT,
        train_loss_dict: FloatLossDict,
        validate_loss_dict: FloatLossDict,
        project_path: Path,
    ) -> None:

        self.train_loss_dict_seq.append(train_loss_dict)
        self.validate_loss_dict_seq.append(validate_loss_dict)

        totals = [dic.total() for dic in self.validate_loss_dict_seq]
        min_loss_sofar = min(totals)
        require_update_model = totals[-1] == min_loss_sofar
        if require_update_model:
            model = copy.deepcopy(model)
            model = model.to(torch.device("cpu"))
            self.best_model = model

            # save everything
            # TODO: error handling
            base_path = self.train_result_path(project_path)
            base_path.mkdir(exist_ok=True, parents=True)
            model_path = base_path / "model.pth"
            valid_loss_path = base_path / "validation_loss.npz"
            train_loss_path = base_path / "train_loss.npz"

            def save():
                torch.save(self.best_model, model_path)
                np.savez(train_loss_path, **self.dump_flds_as_npz_dict(self.train_loss_dict_seq))
                np.savez(valid_loss_path, **self.dump_flds_as_npz_dict(self.validate_loss_dict_seq))

            # error handling for keyboard interrupt
            try:
                save()
            except KeyboardInterrupt:
                logger.info("got keyboard interuppt. but let me dump the object...")
                save()
            except Exception as e:
                logger.info("cannot saved model and losses correctly")
                raise e
            logger.info("model is updated and saved")

    @classmethod
    def load_from_base_path(cls, base_path: Path) -> "TrainCache":
        model_path = base_path / "model.pth"
        valid_loss_path = base_path / "validation_loss.npz"
        train_loss_path = base_path / "train_loss.npz"
        m = re.match(r"(\w+)-(\w+)-(\w+)", base_path.name)
        assert m is not None
        file_uuid = m[3]

        best_model = torch.load(model_path)
        train_loss = cls.load_flds_from_npz_dict(np.load(train_loss_path))
        valid_loss = cls.load_flds_from_npz_dict(np.load(valid_loss_path))
        return cls(best_model, train_loss, valid_loss, file_uuid)

    @classmethod
    def load_all(
        cls,
        project_path: Path,
        model_type: Optional[Type[ModelT]] = None,
        model_config: Optional[ModelConfigBase] = None,
    ) -> "List[TrainCache[ModelT]]":

        ps = cls.filter_result_paths(project_path, model_type, model_config)
        tcache_list = [cls.load_from_base_path(p) for p in ps]
        if len(tcache_list) == 0:
            raise FileNotFoundError
        return tcache_list

    @classmethod
    def load(
        cls,
        project_path: Path,
        model_type: Type[ModelT],
        model_config: Optional[ModelConfigBase] = None,
    ) -> "TrainCache[ModelT]":
        tcache_list = cls.load_all(project_path, model_type, model_config)
        tcaceh_list_sorted = sorted(tcache_list, key=lambda tcache: tcache.min_validate_loss)
        return tcaceh_list_sorted[0]

    def visualize(self, fax: Optional[Tuple] = None):
        fax = plt.subplots() if fax is None else fax
        fig, ax = fax
        train_loss_seq = [dic.total() for dic in self.train_loss_dict_seq]
        valid_loss_seq = [dic.total() for dic in self.validate_loss_dict_seq]
        ax.plot(train_loss_seq)
        ax.plot(valid_loss_seq)
        ax.set_yscale("log")
        ax.legend(["train", "valid"])


def train(
    project_path: Path,
    tcache: TrainCache,
    dataset: Dataset,
    config: TrainConfig = TrainConfig(),
) -> None:

    log_package_version_info(logger, mohou)
    log_text_with_box(logger, "train log")
    logger.info("train start with config: {}".format(config))

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
        logger.info("new epoch: {}".format(epoch))

        model.train()
        train_ld_list: List[FloatLossDict] = []
        for samples in train_loader:
            optimizer.zero_grad()
            samples = move_to_device(samples)
            loss_dict = model.loss(samples)
            loss_dict.total().backward()

            fld = loss_dict.to_float_lossdict()
            train_ld_list.append(fld)
            optimizer.step()

        train_ld_mean = average_float_loss_dict(train_ld_list)

        model.eval()
        validate_ld_list: List[FloatLossDict] = []
        for samples in validate_loader:
            samples = move_to_device(samples)
            loss_dict = model.loss(samples)
            fld = loss_dict.to_float_lossdict()
            validate_ld_list.append(fld)

        validate_ld_mean = average_float_loss_dict(validate_ld_list)

        # update
        logger.info("epoch: {}".format(epoch))
        logger.info("train loss => {}".format(train_ld_mean))
        logger.info("validate loss => {}".format(validate_ld_mean))
        tcache.update_and_save(model, train_ld_mean, validate_ld_mean, project_path)
