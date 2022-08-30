import json
import logging
import re
import uuid
import warnings
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
from mohou.model import FloatLossDict, ModelT, average_float_loss_dict
from mohou.utils import (
    change_color_to_yellow,
    log_package_version_info,
    log_text_with_box,
    split_with_ratio,
)

logger = logging.getLogger(__name__)
warnings.simplefilter("always")


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
    train_lossseq_table: Dict[str, List[float]]
    validate_lossseq_table: Dict[str, List[float]]
    file_uuid: str
    cache_path: Optional[Path] = None  # model specific cache path

    @classmethod
    def from_model(cls, model: ModelT) -> "TrainCache[ModelT]":
        file_uuid = str(uuid.uuid4())[-6:]
        return cls(model, {}, {}, file_uuid)

    @property
    def keys(self) -> List[str]:
        return list(self.train_lossseq_table.keys())

    @staticmethod
    def reduce_to_lossseq(name_lossseq_table: Dict[str, List[float]]) -> List[float]:
        partial_lossseq_list = []
        for partial_lossseq in name_lossseq_table.values():
            partial_lossseq_list.append(partial_lossseq)
        return np.array(partial_lossseq_list).sum(axis=0).tolist()

    @property
    def min_validate_loss(self) -> float:
        lossseq = self.reduce_to_lossseq(self.validate_lossseq_table)
        min_loss_sofar = min(lossseq)
        return min_loss_sofar

    @staticmethod
    def train_result_base_path(project_path: Path) -> Path:
        base_path = project_path / "models"
        return base_path

    def train_result_path(self, project_path: Path):
        base_path = self.train_result_base_path(project_path)
        class_name = self.best_model.__class__.__name__
        result_path = base_path / "{}-{}".format(class_name, self.file_uuid)
        return result_path

    @classmethod
    def filter_result_paths(
        cls,
        project_path: Path,
        model_type: Optional[Type[ModelT]],
        **kwargs,
    ) -> List[Path]:
        """filter train cache path. see filter_predicate for the logic."""

        def is_config_consistent(path: Path) -> bool:

            config_query_specified = len(kwargs) > 0
            if not config_query_specified:
                return True

            config_path = path / "config.json"
            with config_path.open(mode="r") as f:
                config = json.load(f)
            for key, val in kwargs.items():
                assert isinstance(
                    val, (int, float, str)
                ), "currently only naive cast from native type is supported"  # TODO
                t = type(val)
                val_decoded = t(config[key])
                if val != val_decoded:
                    return False
            return True

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
                    return is_config_consistent(path)

        base_path = cls.train_result_base_path(project_path)
        ps = filter(filter_predicate, base_path.iterdir())
        return list(ps)

    @staticmethod
    def dump_lossseq_table_as_npz_dict(name_lossseq_table: Dict[str, List[float]]) -> Dict:
        kwargs = {}
        for key in name_lossseq_table.keys():
            kwargs[key] = np.array(name_lossseq_table[key])
        return kwargs

    @staticmethod
    def load_lossseq_table_from_npz_dict(npz_dict: Dict) -> Dict[str, List[float]]:
        keys = list(npz_dict.keys())
        table = {}
        for key in keys:
            table[key] = npz_dict[key].tolist()
        return table

    def update_and_save(
        self,
        model: ModelT,
        train_loss_dict: FloatLossDict,
        validate_loss_dict: FloatLossDict,
        project_path: Path,
    ) -> None:

        assert train_loss_dict.keys() == validate_loss_dict.keys()

        is_dict_initialized = len(self.train_lossseq_table) > 0
        if not is_dict_initialized:
            for key in train_loss_dict.keys():
                self.train_lossseq_table[key] = []
                self.validate_lossseq_table[key] = []

        # update tables
        for key in train_loss_dict.keys():
            self.train_lossseq_table[key].append(train_loss_dict[key])
            self.validate_lossseq_table[key].append(validate_loss_dict[key])

        validate_loss_list = self.reduce_to_lossseq(self.validate_lossseq_table)
        require_update_model = validate_loss_list[-1] == self.min_validate_loss
        if require_update_model:
            self.best_model = model

            def save():
                base_path = self.train_result_path(project_path)
                base_path.mkdir(exist_ok=True, parents=True)
                model_path = base_path / "model.pth"
                model_config_path = base_path / "config.json"
                valid_loss_path = base_path / "validation_loss.npz"
                train_loss_path = base_path / "train_loss.npz"

                with model_config_path.open(mode="w") as f:
                    d = self.best_model.config.to_dict()
                    json.dump(d, f, indent=2)

                torch.save(self.best_model, model_path)
                np.savez(
                    train_loss_path, **self.dump_lossseq_table_as_npz_dict(self.train_lossseq_table)
                )
                np.savez(
                    valid_loss_path,
                    **self.dump_lossseq_table_as_npz_dict(self.validate_lossseq_table),
                )

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
    def load_from_cache_path(cls, cache_path: Path) -> "TrainCache":
        model_path = cache_path / "model.pth"
        valid_loss_path = cache_path / "validation_loss.npz"
        train_loss_path = cache_path / "train_loss.npz"

        # [mohou < v0.4]
        # (model_type)-(config_hash)-(uuid)
        # example LSTM-924e4427-bd5653
        m = re.match(r"(\w+)-(\w+)-(\w+)", cache_path.name)
        is_legacy_model_path_exist = m is not None
        if is_legacy_model_path_exist:
            assert m is not None  # nothing but for mypy
            file_uuid = m[3]
            message = "NOTE: legacy model's file name (probably created by mohou<0.4) detected."
            logger.warn(change_color_to_yellow(message))
        else:
            # [mohou > v0.4.0]
            # (model_type)-(uuid)
            # example LSTM-bd5653
            m = re.match(r"(\w+)-(\w+)-(\w+)", cache_path.name)
            m = re.match(r"(\w+)-(\w+)", cache_path.name)
            assert m is not None
            file_uuid = m[2]

        best_model: ModelT = torch.load(model_path, map_location=torch.device("cpu"))
        best_model.put_on_device(torch.device("cpu"))
        train_loss = cls.load_lossseq_table_from_npz_dict(np.load(train_loss_path))
        valid_loss = cls.load_lossseq_table_from_npz_dict(np.load(valid_loss_path))
        return cls(best_model, train_loss, valid_loss, file_uuid, cache_path)

    @classmethod
    def load_all(
        cls,
        project_path: Path,
        model_type: Optional[Type[ModelT]] = None,
        **kwargs,
    ) -> "List[TrainCache[ModelT]]":

        # warining for legacy users
        legacy_train_result_path = project_path / "train_result"
        if legacy_train_result_path.exists():
            message = "NOTE: Legacy train_result directory found.\n"
            message += "Please rename {} to {}.".format(
                legacy_train_result_path, cls.train_result_base_path(project_path)
            )
            logger.warn(change_color_to_yellow(message))

        ps = cls.filter_result_paths(project_path, model_type, **kwargs)
        tcache_list = [cls.load_from_cache_path(p) for p in ps]
        if len(tcache_list) == 0:
            if model_type is None:
                model_name = "<not-specified>"
            else:
                model_name = model_type.__name__
            message = "[query] model_name => {}, kwargs => {}, project_path => {}".format(
                model_name, kwargs, project_path
            )
            raise FileNotFoundError(message)

        return tcache_list

    @classmethod
    def load(
        cls,
        project_path: Path,
        model_type: Type[ModelT],
        **kwargs,
    ) -> "TrainCache[ModelT]":
        tcache_list = cls.load_all(project_path, model_type, **kwargs)
        tcaceh_list_sorted = sorted(tcache_list, key=lambda tcache: tcache.min_validate_loss)
        return tcaceh_list_sorted[0]

    def visualize(self, fax: Optional[Tuple] = None):
        fax = plt.subplots() if fax is None else fax
        fig, ax = fax
        train_lossseq = self.reduce_to_lossseq(self.train_lossseq_table)
        valid_lossseq = self.reduce_to_lossseq(self.validate_lossseq_table)
        ax.plot(train_lossseq)
        ax.plot(valid_lossseq)
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
