import logging
import random
import time
from logging import Logger
from pathlib import Path
from typing import List, Optional, Type, Union

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
    WeightPolicy,
)
from mohou.default import load_default_image_encoder
from mohou.encoding_rule import EncodingRule
from mohou.model import (
    LSTM,
    AutoEncoder,
    AutoEncoderBase,
    AutoEncoderConfig,
    LSTMConfig,
)
from mohou.model.chimera import Chimera, ChimeraConfig, ChimeraDataset
from mohou.propagator import Propagator
from mohou.trainer import TrainCache, TrainConfig, train
from mohou.types import (
    AngleVector,
    ElementDict,
    EpisodeBundle,
    GripperState,
    ImageBase,
    TerminateFlag,
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
    model_config: LSTMConfig,
    dataset_config: AutoRegressiveDatasetConfig,
    train_config: TrainConfig,
    weighting: Optional[Union[WeightPolicy, List[np.ndarray]]] = None,
    bundle: Optional[EpisodeBundle] = None,
    warm_start: bool = False,
    context_list: Optional[List[np.ndarray]] = None,
):

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
        weighting=weighting,
        static_context_list=context_list,
    )

    if warm_start:
        logger.info("warm start")
        tcache = TrainCache.load(project_path, LSTM)
        train(project_path, tcache, dataset, config=train_config)
    else:
        model = LSTM(model_config)
        tcache = TrainCache.from_model(model)  # type: ignore[var-annotated]
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
    assert ae is not None
    conf = ChimeraConfig(lstm_config, ae_config=ae)
    model = Chimera(conf)  # type: ignore[var-annotated]
    tcache = TrainCache.from_model(model)  # type: ignore[var-annotated]
    train(project_path, tcache, dataset, train_config)


def visualize_train_histories(project_path: Path):
    plot_dir_path = project_path / "train_history"
    plot_dir_path.mkdir(exist_ok=True)

    all_result_paths = TrainCache.filter_result_paths(project_path, None, None)
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


def visualize_variational_autoencoder(project_path: Path):

    tcache = TrainCache.load(project_path, VariationalAutoEncoder)
    vae = tcache.best_model
    assert vae is not None

    save_dir_path = project_path / "autoencoder_result"
    save_dir_path.mkdir(exist_ok=True)

    for axis in range(vae.config.n_bottleneck):
        images = vae.get_latent_axis_images(axis)
        assert ImageSequenceClip is not None, "check if your moviepy is properly installed"
        clip = ImageSequenceClip([im.numpy() for im in images], fps=20)
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


def visualize_lstm_propagation(project_path: Path, propagator: Propagator, n_prop: int):
    bundle = EpisodeBundle.load(project_path).get_untouch_bundle()
    save_dir_path = project_path / "lstm_result"
    save_dir_path.mkdir(exist_ok=True)

    image_encoder = load_default_image_encoder(project_path)

    for idx, edata in enumerate(bundle):
        episode_data = bundle[idx]

        image_type = None
        for key, encoder in propagator.encoding_rule.items():
            if issubclass(key, ImageBase):
                image_type = key
        assert image_type is not None

        n_feed = 20
        feed_avs = episode_data.get_sequence_by_type(AngleVector)[:n_feed]
        feed_images = episode_data.get_sequence_by_type(image_type)[:n_feed]

        # set context if necessary
        if propagator.require_static_context:
            context = image_encoder.forward(feed_images[0])
            propagator.set_static_context(context)

        use_gripper = GripperState in propagator.encoding_rule

        if use_gripper:
            fed_grippers = episode_data.get_sequence_by_type(GripperState)[:n_feed]

        print("start lstm propagation")
        for i in range(n_feed):
            elem_dict = ElementDict([feed_avs[i], feed_images[i]])
            if use_gripper:
                elem_dict[GripperState] = fed_grippers[i]
            propagator.feed(elem_dict)
        print("finish lstm propagation")

        elem_dict_list = propagator.predict(n_prop)
        pred_images = [elem_dict[image_type] for elem_dict in elem_dict_list]
        pred_flags = [elem_dict[TerminateFlag].numpy().item() for elem_dict in elem_dict_list]
        pred_avs = [elem_dict[AngleVector].numpy() for elem_dict in elem_dict_list]
        if use_gripper:
            pred_gss = [elem_dict[GripperState].numpy() for elem_dict in elem_dict_list]

        n_av_dim = bundle.spec.type_shape_table[AngleVector][0]
        n_gs_dim = bundle.spec.type_shape_table[GripperState][0] if use_gripper else 0
        fig, axs = plt.subplots(n_av_dim + n_gs_dim, 1)

        # plot angle vectors
        av_seq_gt = episode_data.get_sequence_by_type(AngleVector)
        np_av_seq_gt = np.array([av.numpy() for av in av_seq_gt])
        np_av_seq_pred = np.concatenate((np_av_seq_gt[:n_feed], np.array(pred_avs)), axis=0)

        i_dim = 0
        for i_av_dim in range(n_av_dim):
            axs[i_dim].plot(np_av_seq_gt[:, i_av_dim], color="blue", lw=1)
            axs[i_dim].plot(np_av_seq_pred[:, i_av_dim], color="red", lw=1)

            # determine axes min max
            conc = np.hstack((np_av_seq_gt[:, i_av_dim], np_av_seq_pred[:, i_av_dim]))
            y_min = np.min(conc)
            y_max = np.max(conc)
            diff = y_max - y_min
            axs[i_dim].set_ylim([y_min - diff * 0.1, y_max + diff * 0.1])
            axs[i_dim].set_title("AngleVector dim {}".format(i_av_dim), fontsize=5, pad=0.0)
            i_dim += 1

        if use_gripper:
            gs_seq_gt = episode_data.get_sequence_by_type(GripperState)
            np_gs_seq_gt = np.array([gs.numpy() for gs in gs_seq_gt])
            np_gs_seq_pred = np.concatenate((np_gs_seq_gt[:n_feed], np.array(pred_gss)), axis=0)
            for i_gs_dim in range(n_gs_dim):
                axs[i_dim].plot(np_gs_seq_gt[:, i_gs_dim], color="blue", lw=1)
                axs[i_dim].plot(np_gs_seq_pred[:, i_gs_dim], color="red", lw=1)

                # determine axes min max
                conc = np.hstack((np_gs_seq_gt[:, i_gs_dim], np_gs_seq_pred[:, i_gs_dim]))
                y_min = np.min(conc)
                y_max = np.max(conc)
                diff = y_max - y_min
                axs[i_dim].set_ylim([y_min - diff * 0.1, y_max + diff * 0.1])
                axs[i_dim].set_title("GripperState dim {}".format(i_gs_dim), fontsize=5, pad=0.0)
                i_dim += 1

        for ax in axs:
            ax.grid()

        image_path = save_dir_path / "seq-{}{}.png".format(AngleVector.__name__, idx)
        fig.savefig(str(image_path), format="png", dpi=300)
        print("saved to {}".format(image_path))

        # save gif image
        print("adding text to images...")
        fed_images_with_text = [
            add_text_to_image(image, "fed (original) image)", "blue") for image in feed_images
        ]
        clamp = lambda x: max(min(x, 1.0), 0.0)  # noqa
        pred_images_with_text = [
            add_text_to_image(
                image, "predicted image (prob-terminated={:.2f})".format(clamp(flag)), "green"
            )
            for image, flag in zip(pred_images, pred_flags)
        ]

        images_with_text = fed_images_with_text + pred_images_with_text

        image_path = save_dir_path / "result-image{}.gif".format(idx)
        assert ImageSequenceClip is not None, "check if your moviepy is properly installed"
        clip = ImageSequenceClip(images_with_text, fps=20)
        clip.write_gif(str(image_path), fps=20)
