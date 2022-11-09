import json
import logging
import pickle
import re
import uuid
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, Generic, Iterable, List, Optional, Tuple, Type, TypeVar

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
    detect_device,
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
    utc_time_created: Optional[datetime] = None
    utc_time_saved: Optional[datetime] = None

    class AttrFileName(Enum):
        model = "model.pth"
        model_config = "config.json"
        valid_loss = "validation_loss.npz"
        train_loss = "train_loss.npz"
        utc_time_created = "utc_time_created.pkl"
        utc_time_saved = "utc_time_saved.pkl"

    def __post_init__(self):
        self.utc_time_created = datetime.now(timezone.utc)

    def cache_path(self, project_path: Path) -> Path:
        class_name = self.best_model.__class__.__name__
        cache_name = "{}-{}".format(class_name, self.file_uuid)
        return self.train_result_base_path(project_path) / cache_name

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
    def min_valid_loss(self) -> Tuple[int, float]:
        """get index and valid loss value if loss corresponding best_model"""
        lossseq = self.reduce_to_lossseq(self.validate_lossseq_table)
        index = int(np.argmin(lossseq))
        return index, lossseq[index]

    @staticmethod
    def train_result_base_path(project_path: Path) -> Path:
        base_path = project_path / "models"
        return base_path

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

    def save(self, project_path: Path):
        self.utc_time_saved = datetime.now(timezone.utc)

        def save():
            base_path = self.cache_path(project_path)
            assert base_path is not None  # for mypy
            base_path.mkdir(exist_ok=True, parents=True)
            model_path = base_path / self.AttrFileName.model.value
            model_config_path = base_path / self.AttrFileName.model_config.value
            valid_loss_path = base_path / self.AttrFileName.valid_loss.value
            train_loss_path = base_path / self.AttrFileName.train_loss.value
            utc_time_created_path = base_path / self.AttrFileName.utc_time_created.value
            utc_time_saved_path = base_path / self.AttrFileName.utc_time_saved.value

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
            with utc_time_created_path.open(mode="wb") as f:
                pickle.dump(self.utc_time_created, f)
            with utc_time_saved_path.open(mode="wb") as f:
                pickle.dump(self.utc_time_saved, f)

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
        require_update = len(validate_loss_list) - 1 == self.min_valid_loss[0]
        if require_update:
            assert validate_loss_list[-1] == self.min_valid_loss[1]
            self.best_model = model
            self.save(project_path)

    @classmethod
    def load_from_cache_path(cls, cache_path: Path) -> "TrainCache":
        model_path = cache_path / cls.AttrFileName.model.value
        valid_loss_path = cache_path / cls.AttrFileName.valid_loss.value
        train_loss_path = cache_path / cls.AttrFileName.train_loss.value
        utc_time_created_path = cache_path / cls.AttrFileName.utc_time_created.value
        utc_time_saved_path = cache_path / cls.AttrFileName.utc_time_saved.value

        # [mohou < v0.4]
        # (model_type)-(config_hash)-(uuid)
        # example LSTM-924e4427-bd5653
        m = re.match(r"(\w+)-(\w+)-(\w+)", cache_path.name)
        is_legacy_model_path_exist = m is not None
        if is_legacy_model_path_exist:
            assert m is not None  # nothing but for mypy
            file_uuid = m[3]
            message = "NOTE: legacy cache directory path {} (probably created by version lower than 0.4.0) detected.".format(
                cache_path
            )
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

        # must consider that regacy TrainCache may not have utc_time
        if utc_time_created_path.exists():
            with utc_time_created_path.open(mode="rb") as f:
                utc_time_created = pickle.load(f)
        else:
            utc_time_created = None

        if utc_time_saved_path.exists():
            with utc_time_saved_path.open(mode="rb") as f:
                utc_time_saved = pickle.load(f)
        else:
            utc_time_saved = None

        return cls(best_model, train_loss, valid_loss, file_uuid, utc_time_created, utc_time_saved)

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
            message += "NOTE: Please rename {} to {}.".format(
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
        tcaceh_list_sorted = sorted(tcache_list, key=lambda tcache: tcache.min_valid_loss[1])
        return tcaceh_list_sorted[0]

    @classmethod
    def load_latest(
        cls,
        project_path: Path,
        model_type: Type[ModelT],
        **kwargs,
    ) -> "TrainCache[ModelT]":
        def get_cache_time_stamp(tcache: TrainCache) -> float:
            msg = "legacy tcache {} does not have timestamp".format(tcache.cache_path(project_path))
            assert tcache.utc_time_saved is not None, msg
            return tcache.utc_time_saved.timestamp()

        tcache_list = cls.load_all(project_path, model_type, **kwargs)
        tcache_list_sorted = sorted(tcache_list, key=get_cache_time_stamp)
        return tcache_list_sorted[-1]

    def visualize(self) -> Tuple:
        fig, axes = plt.subplots(1, 3)
        ax = axes[0]
        train_lossseq = self.reduce_to_lossseq(self.train_lossseq_table)
        valid_lossseq = self.reduce_to_lossseq(self.validate_lossseq_table)
        ax.plot(train_lossseq)
        ax.plot(valid_lossseq)
        ax.set_yscale("log")
        ax.legend(["train", "valid"])
        ax.title.set_text("valid and train total loss")

        keys = list(self.validate_lossseq_table.keys())

        for key in keys:
            axes[1].plot(self.train_lossseq_table[key])
            axes[2].plot(self.validate_lossseq_table[key])
            axes[1].title.set_text("each train loss")
            axes[2].title.set_text("each valid loss")
        for ax in [axes[1], axes[2]]:
            ax.legend(keys)
            ax.set_yscale("log")

        return (fig, axes)


def train_lower(
    project_path: Path,
    tcache: TrainCache,
    train_loader: Iterable,
    validate_loader: Iterable,
    config: TrainConfig = TrainConfig(),
    device: Optional[torch.device] = None,
) -> None:
    r"""
    low-level train function that accepts train loader
    """

    log_package_version_info(logger, mohou)
    log_text_with_box(logger, "train log")
    logger.info("train start with config: {}".format(config))
    logger.info("model cache path: {}".format(tcache.cache_path(project_path)))

    model = tcache.best_model
    if device is None:
        device = detect_device()
    model.put_on_device(device)
    logger.info("put model on {}".format(device))

    def move_to_device(sample):
        if isinstance(sample, torch.Tensor):
            return sample.to(model.device)
        elif isinstance(sample, list):  # NOTE datalodaer return list type not tuple
            return tuple([e.to(model.device) for e in sample])
        else:
            raise RuntimeError

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


def train(
    project_path: Path,
    tcache: TrainCache,
    dataset: Dataset,
    config: TrainConfig = TrainConfig(),
    device: Optional[torch.device] = None,
) -> None:
    r"""
    higher-level train function that auto create dataloader from the dataset
    """

    dataset_train, dataset_validate = split_with_ratio(dataset, config.valid_data_ratio)

    train_loader = DataLoader(dataset=dataset_train, batch_size=config.batch_size, shuffle=True)
    validate_loader = DataLoader(
        dataset=dataset_validate, batch_size=config.batch_size, shuffle=True
    )
    train_lower(project_path, tcache, train_loader, validate_loader, config=config, device=device)
