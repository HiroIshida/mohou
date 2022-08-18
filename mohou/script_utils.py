import logging
import random
import time
from logging import Logger
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import matplotlib.pyplot as plt
import numpy as np
import tqdm

from mohou.model.autoencoder import VariationalAutoEncoder

try:
    from moviepy.editor import ImageSequenceClip
except Exception:
    ImageSequenceClip = None

from mohou.dataset import (
    AutoEncoderDataset,
    AutoEncoderDatasetConfig,
    AutoRegressiveDataset,
    AutoRegressiveDatasetConfig,
)
from mohou.default import load_default_image_encoder
from mohou.encoding_rule import EncodingRule
from mohou.model import (
    LSTM,
    PBLSTM,
    AutoEncoder,
    AutoEncoderBase,
    AutoEncoderConfig,
    LSTMConfig,
    PBLSTMConfig,
)
from mohou.model.chimera import Chimera, ChimeraConfig, ChimeraDataset
from mohou.model.lstm import LSTMBase, LSTMConfigBase
from mohou.propagator import PropagatorBase
from mohou.trainer import TrainCache, TrainConfig, train
from mohou.types import (
    ElementDict,
    EpisodeBundle,
    EpisodeData,
    ImageBase,
    TerminateFlag,
    VectorBase,
)
from mohou.utils import canvas_to_ndarray

logger = logging.getLogger(__name__)


def create_default_logger(project_path: Path, prefix: str) -> Logger:
    timestr = "_" + time.strftime("%Y%m%d%H%M%S")
    log_dir_path = project_path / "log"
    log_dir_path.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir_path / (prefix + timestr + ".log")

    FORMAT = "[%(levelname)s] %(asctime)s %(name)s: %(message)s"
    logging.basicConfig(filename=str(log_file_path), format=FORMAT)
    logger = logging.getLogger("mohou")
    logger.setLevel(level=logging.INFO)

    log_sym_path = log_dir_path / ("latest_" + prefix + ".log")

    logger.info("create log symlink :{0} => {1}".format(log_file_path, log_sym_path))
    if log_sym_path.is_symlink():
        log_sym_path.unlink()
    log_sym_path.symlink_to(log_file_path)

    return logger


def train_autoencoder(
    project_path: Path,
    image_type: Type[ImageBase],
    model_config: AutoEncoderConfig,
    dataset_config: AutoEncoderDatasetConfig,
    train_config: TrainConfig,
    ae_type: Type[AutoEncoderBase] = AutoEncoder,
    bundle: Optional[EpisodeBundle] = None,
    warm_start: bool = False,
):

    if bundle is None:
        bundle = EpisodeBundle.load(project_path)

    dataset = AutoEncoderDataset.from_bundle(bundle, image_type, dataset_config)
    if warm_start:
        logger.info("warm start")
        tcache = TrainCache.load(project_path, ae_type)
        train(project_path, tcache, dataset, config=train_config)
    else:
        model = ae_type(model_config)  # type: ignore
        tcache = TrainCache.from_model(model)  # type: ignore[var-annotated]
        train(project_path, tcache, dataset, config=train_config)


def train_lstm(
    project_path: Path,
    encoding_rule: EncodingRule,
    model_config: LSTMConfigBase,
    dataset_config: AutoRegressiveDatasetConfig,
    train_config: TrainConfig,
    model_type: Type[LSTMBase] = LSTM,
    bundle: Optional[EpisodeBundle] = None,
    warm_start: bool = False,
    context_list: Optional[List[np.ndarray]] = None,
):
    # a dirty assertion TODO: do this by generic typing
    compat_table: Dict[Type[LSTMBase], Type[LSTMConfigBase]] = {
        LSTM: LSTMConfig,
        PBLSTM: PBLSTMConfig,
    }
    assert model_type in compat_table
    assert compat_table[model_type] == type(model_config)

    if bundle is None:
        bundle = EpisodeBundle.load(project_path)

    if context_list is None:
        assert model_config.n_static_context == 0
    else:
        for context in context_list:
            assert len(context) == model_config.n_static_context

    dataset = AutoRegressiveDataset.from_bundle(
        bundle,
        encoding_rule,
        augconfig=dataset_config,
        static_context_list=context_list,
    )

    if warm_start:
        logger.info("warm start")
        tcache = TrainCache.load(project_path, LSTM)
        train(project_path, tcache, dataset, config=train_config)
    else:
        model = model_type(model_config)
        tcache = TrainCache.from_model(model)  # type: ignore
        train(project_path, tcache, dataset, config=train_config)


def train_chimera(
    project_path: Path,
    encoding_rule: EncodingRule,
    lstm_config: LSTMConfig,
    train_config: TrainConfig,
    bundle: Optional[EpisodeBundle] = None,
):  # TODO(HiroIshida): minimal args

    if bundle is None:
        bundle = EpisodeBundle.load(project_path)

    dataset = ChimeraDataset.from_bundle(bundle, encoding_rule)
    ae = TrainCache.load(project_path, AutoEncoder).best_model
    conf = ChimeraConfig(lstm_config, ae_config=ae)
    model = Chimera(conf)  # type: ignore[var-annotated]
    tcache = TrainCache.from_model(model)  # type: ignore[var-annotated]
    train(project_path, tcache, dataset, train_config)


def visualize_train_histories(project_path: Path):
    plot_dir_path = project_path / "train_history"
    plot_dir_path.mkdir(exist_ok=True)

    all_result_paths = TrainCache.filter_result_paths(project_path, None)
    for result_path in all_result_paths:
        tcache = TrainCache.load_from_base_path(result_path)
        image_path = plot_dir_path / (result_path.name + ".png")
        fig, ax = plt.subplots()
        tcache.visualize((fig, ax))
        fig.savefig(str(image_path))
        print("saved to {}".format(image_path))


def visualize_image_reconstruction(
    project_path: Path,
    bundle: EpisodeBundle,
    autoencoder: AutoEncoderBase,
    n_vis: int = 5,
    prefix: Optional[str] = None,
):
    save_dir_path = project_path / "autoencoder_result"
    save_dir_path.mkdir(exist_ok=True)

    bundle_untouch = bundle.get_untouch_bundle()
    bundle_touch = bundle.get_touch_bundle()

    image_type = autoencoder.image_type  # type: ignore[union-attr]
    no_aug = AutoEncoderDatasetConfig(0)  # to feed not randomized image
    dataset_untouch = AutoEncoderDataset.from_bundle(bundle_untouch, image_type, no_aug)
    dataset_touch = AutoEncoderDataset.from_bundle(bundle_touch, image_type, no_aug)

    for dataset, postfix in zip([dataset_untouch, dataset_touch], ["untouch", "touch"]):
        idxes = list(range(len(dataset)))
        random.shuffle(idxes)
        idxes_test = idxes[: min(n_vis, len(dataset))]

        for i, idx in enumerate(idxes_test):

            image_torch = dataset[idx].unsqueeze(dim=0)
            image_torch_reconstructed = autoencoder(image_torch)

            img = dataset.image_type.from_tensor(image_torch.squeeze(dim=0))
            img_reconstructed = dataset.image_type.from_tensor(
                image_torch_reconstructed.squeeze(dim=0)
            )

            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle("left: original, right: reconstructed")
            ax1.imshow(img.to_rgb()._data)
            ax2.imshow(img_reconstructed.to_rgb()._data)
            filename = "result-{}-{}.png".format(postfix, i)
            if prefix is not None:
                filename = prefix + "-" + str(filename)
            file_path = save_dir_path / filename
            plt.savefig(str(file_path))


def visualize_variational_autoencoder(project_path: Path) -> None:

    tcache = TrainCache[VariationalAutoEncoder[ImageBase]].load(
        project_path, VariationalAutoEncoder
    )
    vae = tcache.best_model

    save_dir_path = project_path / "autoencoder_result"
    save_dir_path.mkdir(exist_ok=True)

    for axis in range(vae.config.n_bottleneck):
        images = vae.get_latent_axis_images(axis)
        assert ImageSequenceClip is not None, "check if your moviepy is properly installed"
        clip = ImageSequenceClip([im.to_rgb().numpy() for im in images], fps=20)
        file_path = save_dir_path / "vae-axis{}.gif".format(axis)
        clip.write_gif(str(file_path), fps=20)


def add_text_to_image(image: ImageBase, text: str, color: str):
    fig = plt.figure(tight_layout={"pad": 0})
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    ax.imshow(image.to_rgb()._data)
    bbox = dict(boxstyle="round", facecolor="white", alpha=0.4)
    ax.text(7, 1, text, fontsize=15, color=color, verticalalignment="top", bbox=bbox)
    fig.canvas.draw()
    fig.canvas.flush_events()
    return canvas_to_ndarray(fig)


class VectorSequencePlotter:
    fig: Any
    axs: List
    axis_index_table: Dict[Tuple[Type[VectorBase], int], int]
    inverse_axis_index_table: List[Tuple[Type[VectorBase], int]]
    min_val_per_axis: List[float]
    max_val_per_axis: List[float]
    is_finalized: bool = False

    def __init__(self, project_path: Path, plot_elem_types: List[Type[VectorBase]]) -> None:
        bundle = EpisodeBundle.load(project_path)

        axis_index_table: Dict[Tuple[Type[VectorBase], int], int] = {}
        inverse_axis_index_table: List[Tuple[Type[VectorBase], int]] = []
        axis_count = 0
        for elem_type in plot_elem_types:
            shape = bundle.type_shape_table[elem_type]
            assert len(shape) == 1  # because it's a vector
            for i in range(shape[0]):
                axis_index_table[(elem_type, i)] = axis_count
                inverse_axis_index_table.append((elem_type, i))
                axis_count += 1

        fig, axs = plt.subplots(axis_count, 1)
        if axis_count == 1:
            axs = [axs]

        for ax in axs:
            ax.set_xticklabels([])
            ax.yaxis.set_tick_params(labelsize=5)

        fig.tight_layout(pad=3.0)

        self.fig = fig
        self.axs = axs
        self.axis_index_table = axis_index_table
        self.inverse_axis_index_table = inverse_axis_index_table
        self.min_val_per_axis = [np.inf for _ in range(axis_count)]
        self.max_val_per_axis = [-np.inf for _ in range(axis_count)]

    def add_plot(
        self,
        x_seq: Optional[np.ndarray],
        y_seq: np.ndarray,
        elem_type: Type[VectorBase],
        idx: int,
        **kwargs
    ):
        assert y_seq.ndim == 1
        assert not self.is_finalized
        axis_index = self.axis_index_table[(elem_type, idx)]
        if x_seq is not None:
            assert x_seq.ndim == 1
            self.axs[axis_index].plot(x_seq, y_seq, **kwargs)
        else:
            self.axs[axis_index].plot(y_seq, **kwargs)
        self.min_val_per_axis[axis_index] = min(self.min_val_per_axis[axis_index], np.min(y_seq))
        self.max_val_per_axis[axis_index] = max(self.max_val_per_axis[axis_index], np.max(y_seq))

    def finalize(self):
        ax_last = self.axs[-1]
        ax_last.legend(fontsize=5)

        for axis_index, ax in enumerate(self.axs):
            val_min = self.min_val_per_axis[axis_index]
            val_max = self.max_val_per_axis[axis_index]
            margin = max(0.1, 0.1 * (val_max - val_min))
            ax.set_ylim(val_min - margin, val_max + margin)
            ax.grid()

            elem_type, vector_index = self.inverse_axis_index_table[axis_index]
            ax.text(
                0.5,
                0.5,
                "{} dimension {}".format(elem_type.__name__, vector_index),
                fontsize=5,
                transform=ax.transAxes,
                horizontalalignment="center",
                verticalalignment="center",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            )
        self.is_finalized = True

    def save_fig(self, image_path: Path):
        assert self.is_finalized
        self.fig.savefig(str(image_path), format="png", dpi=300)


def visualize_lstm_propagation(project_path: Path, propagator: PropagatorBase, n_prop: int):
    bundle = EpisodeBundle.load(project_path).get_untouch_bundle()
    save_dir_path = project_path / "lstm_result"
    save_dir_path.mkdir(exist_ok=True)
    prop_name = propagator.__class__.__name__

    if propagator.require_static_context:
        raise NotImplementedError("not implemented yet")

    image_encoder = load_default_image_encoder(project_path)
    image_type = image_encoder.elem_type

    for idx, episode_data in enumerate(bundle):

        print("start lstm propagation")
        n_feed = 20
        feed_partial_episode = episode_data[:n_feed]
        for i in range(n_feed):
            feed_edict = feed_partial_episode[i]
            propagator.feed(feed_edict)
        pred_edict_list = propagator.predict(n_prop)
        pred_partial_episode = EpisodeData.from_edict_list(
            pred_edict_list, check_terminate_flag=False
        )
        print("finish lstm propagation")

        # Plot history of Vector types
        elem_types = propagator.encoding_rule.keys()
        vector_elem_types = [et for et in elem_types if issubclass(et, VectorBase)]

        vecseq_plotter = VectorSequencePlotter(project_path, vector_elem_types)

        for elem_type in vector_elem_types:

            # create ground truth data
            seq_gt = episode_data.get_sequence_by_type(elem_type)
            np_seq_gt = np.array([e.numpy() for e in seq_gt])

            # create feed + pred data
            seq_feed = feed_partial_episode.get_sequence_by_type(elem_type)
            seq_pred = pred_partial_episode.get_sequence_by_type(elem_type)
            np_seq_feedpred = np.array(
                [e.numpy() for e in seq_feed] + [e.numpy() for e in seq_pred]
            )

            n_dim = seq_feed.elem_shape[0]
            for i in range(n_dim):
                np_seq_gt_slice = np_seq_gt[:, i]
                np_seq_feedpred_slice = np_seq_feedpred[:, i]
                vecseq_plotter.add_plot(
                    None, np_seq_gt_slice, elem_type, i, color="blue", lw=1, label="ground truth"
                )
                vecseq_plotter.add_plot(
                    None,
                    np_seq_feedpred_slice,
                    elem_type,
                    i,
                    color="red",
                    lw=1,
                    label="feed + pred",
                )

        image_path = save_dir_path / "seq-vectors-{}{}.png".format(prop_name, idx)
        vecseq_plotter.finalize()
        vecseq_plotter.save_fig(image_path)
        print("saved to {}".format(image_path))

        # save gif image
        print("adding text to images...")
        feed_images = feed_partial_episode.get_sequence_by_type(image_type)
        fed_images_with_text = [
            add_text_to_image(image, "fed (original) image)", "blue") for image in feed_images
        ]
        clamp = lambda x: max(min(x, 1.0), 0.0)  # noqa
        pred_images = pred_partial_episode.get_sequence_by_type(image_type)
        pred_flags = pred_partial_episode.get_sequence_by_type(TerminateFlag)
        pred_images_with_text = [
            add_text_to_image(
                image,
                "predicted image (prob-terminated={:.2f})".format(clamp(flag.numpy().item())),
                "green",
            )
            for image, flag in zip(pred_images, pred_flags)
        ]

        images_with_text = fed_images_with_text + pred_images_with_text

        image_path = save_dir_path / "seq-{}-rgb{}.gif".format(prop_name, idx)
        assert ImageSequenceClip is not None, "check if your moviepy is properly installed"
        clip = ImageSequenceClip(images_with_text, fps=20)
        clip.write_gif(str(image_path), fps=20)


def plot_execution_result(
    project_path: Path, propagator: PropagatorBase, edict_seq: List[ElementDict], n_prop: int = 10
):
    timestr = "_" + time.strftime("%Y%m%d%H%M%S")
    propagator.clear_fed_data()

    elem_types = propagator.encoding_rule.keys()
    vector_elem_types = [et for et in elem_types if issubclass(et, VectorBase)]
    vecseq_plotter = VectorSequencePlotter(project_path, vector_elem_types)

    # plot actual data
    episode = EpisodeData.from_edict_list(edict_seq, check_terminate_flag=False)
    for elem_type in vector_elem_types:
        seq = episode.get_sequence_by_type(elem_type)
        np_seq = np.array([e.numpy() for e in seq])
        n_dim = np_seq.shape[1]
        for i in range(n_dim):
            y_seq = np_seq[:, i]
            vecseq_plotter.add_plot(None, y_seq, elem_type, i, color="blue", lw=1.0)

    # plot prediction
    for time_index, edict in enumerate(tqdm.tqdm(edict_seq)):
        x_seq = np.array([time_index + 1 + j for j in range(n_prop)])

        propagator.feed(edict)
        edict_list_predicted = propagator.predict(n_prop)
        episode_predicted = EpisodeData.from_edict_list(
            edict_list_predicted, check_terminate_flag=False
        )

        for elem_type in vector_elem_types:
            seq = episode_predicted.get_sequence_by_type(elem_type)
            np_seq = np.array([e.numpy() for e in seq])
            n_dim = np_seq.shape[1]

            for i in range(n_dim):
                y_seq = np_seq[:, i]
                vecseq_plotter.add_plot(
                    x_seq, y_seq, elem_type, i, color="red", lw=0.1, marker=".", markersize="0.2"
                )

    save_dir_path = project_path / "execution_result"
    save_dir_path.mkdir(exist_ok=True)
    prop_name = propagator.__class__.__name__
    image_path = save_dir_path / "result-{}-{}.png".format(prop_name, timestr)
    vecseq_plotter.finalize()
    vecseq_plotter.save_fig(image_path)
