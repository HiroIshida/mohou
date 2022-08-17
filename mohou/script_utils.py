import logging
import random
import time
from logging import Logger
from pathlib import Path
from typing import Dict, List, Optional, Type

import matplotlib.pyplot as plt
import numpy as np

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
from mohou.types import EpisodeBundle, EpisodeData, ImageBase, TerminateFlag, VectorBase
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
    bbox = dict(boxstyle="round", facecolor="white", alpha=0.7)
    ax.text(7, 1, text, fontsize=15, color=color, verticalalignment="top", bbox=bbox)
    fig.canvas.draw()
    fig.canvas.flush_events()
    return canvas_to_ndarray(fig)


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

        vector_dims: List[int] = []
        for et in vector_elem_types:
            shape = bundle.type_shape_table[et]
            assert len(shape) == 1  # because it's a vector
            vector_dims.append(shape[0])

        total_vector_dim = sum(vector_dims)
        fig, axs = plt.subplots(total_vector_dim, 1)
        fig.tight_layout(pad=3.0)
        idx_axis = -1

        for dim, elem_type in zip(vector_dims, vector_elem_types):

            # create ground truth data
            seq_gt = episode_data.get_sequence_by_type(elem_type)
            np_seq_gt = np.array([e.numpy() for e in seq_gt])

            # create feed + pred data
            seq_feed = feed_partial_episode.get_sequence_by_type(elem_type)
            seq_pred = pred_partial_episode.get_sequence_by_type(elem_type)
            np_seq_feedpred = np.array(
                [e.numpy() for e in seq_feed] + [e.numpy() for e in seq_pred]
            )

            for i in range(dim):
                idx_axis += 1
                ax = axs[idx_axis]
                ax.set_xticklabels([])
                ax.yaxis.set_tick_params(labelsize=5)

                ax.text(
                    0.1,
                    0.85,
                    "{} dimension {}".format(elem_type.__name__, i),
                    fontsize=5,
                    transform=ax.transAxes,
                    horizontalalignment="center",
                    verticalalignment="center",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
                )

                np_seq_gt_slice = np_seq_gt[:, i]
                np_seq_feedpred_slice = np_seq_feedpred[:, i]

                ax.plot(np_seq_gt_slice, color="blue", lw=1, label="ground truth")
                ax.plot(np_seq_feedpred_slice, color="red", lw=1, label="feed + pred")

                val_min = min(np.min(np_seq_gt_slice), np.min(np_seq_feedpred_slice))
                val_max = max(np.max(np_seq_gt_slice), np.max(np_seq_feedpred_slice))
                margin = max(0.1, 0.1 * (val_max - val_min))
                ax.set_ylim(val_min - 0.1 * margin, val_max + 0.1 * margin)

        ax_last = axs[-1]
        ax_last.legend(fontsize=5)

        for ax in axs:
            ax.grid()
        plt.subplot_tool()

        image_path = save_dir_path / "seq-vectors-{}{}.png".format(prop_name, idx)
        fig.savefig(str(image_path), format="png", dpi=300)
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
