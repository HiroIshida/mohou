import os
import numpy as np
import re
import random
import pickle
import time
import logging
from logging import Logger
from typing import Type, Optional
import matplotlib.pyplot as plt

try:
    from moviepy.editor import ImageSequenceClip
except Exception:
    ImageSequenceClip = None

from mohou.types import ImageBase, AngleVector, TerminateFlag, ElementDict, GripperState
from mohou.types import MultiEpisodeChunk
from mohou.model import AutoEncoder
from mohou.model import AutoEncoderConfig
from mohou.model import LSTM
from mohou.model import LSTMConfig
from mohou.model import AutoEncoderBase
from mohou.dataset import AutoEncoderDataset
from mohou.dataset import AutoEncoderDatasetConfig
from mohou.dataset import AutoRegressiveDataset
from mohou.dataset import AutoRegressiveDatasetConfig
from mohou.propagator import Propagator
from mohou.file import get_project_dir, get_subproject_dir
from mohou.trainer import TrainCache, TrainConfig, train
from mohou.embedding_rule import EmbeddingRule
from mohou.utils import canvas_to_ndarray

logger = logging.getLogger(__name__)


def create_default_logger(project_name: str, prefix: str) -> Logger:
    timestr = "_" + time.strftime("%Y%m%d%H%M%S")
    log_dir = os.path.join(get_project_dir(project_name), 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file_name = os.path.join(log_dir, (prefix + timestr + '.log'))
    FORMAT = '[%(levelname)s] %(asctime)s %(name)s: %(message)s'
    logging.basicConfig(filename=log_file_name, format=FORMAT)
    logger = logging.getLogger('mohou')
    logger.setLevel(level=logging.INFO)

    log_sym_name = os.path.join(log_dir, ('latest_' + prefix + '.log'))
    logger.info('create log symlink :{0} => {1}'.format(log_file_name, log_sym_name))
    if os.path.islink(log_sym_name):
        os.unlink(log_sym_name)
    os.symlink(log_file_name, log_sym_name)
    return logger


def train_autoencoder(
        project_name: str,
        image_type: Type[ImageBase],
        model_config: AutoEncoderConfig,
        dataset_config: AutoEncoderDatasetConfig,
        train_config: TrainConfig,
        ae_type: Type[AutoEncoderBase] = AutoEncoder,
        chunk: Optional[MultiEpisodeChunk] = None,
        warm_start: bool = False):

    if chunk is None:
        chunk = MultiEpisodeChunk.load(project_name)

    dataset = AutoEncoderDataset.from_chunk(chunk, image_type, dataset_config)
    if warm_start:
        logger.info('warm start')
        tcache = TrainCache.load(project_name, ae_type)
        train(tcache, dataset, model=None, config=train_config)
    else:
        tcache = TrainCache(project_name)  # type: ignore[var-annotated]
        model = ae_type(model_config)  # type: ignore
        train(tcache, dataset, model=model, config=train_config)


def train_lstm(
        project_name: str,
        embedding_rule: EmbeddingRule,
        model_config: LSTMConfig,
        dataset_config: AutoRegressiveDatasetConfig,
        train_config: TrainConfig,
        chunk: Optional[MultiEpisodeChunk] = None,
        warm_start: bool = False):

    if chunk is None:
        chunk = MultiEpisodeChunk.load(project_name)

    dataset = AutoRegressiveDataset.from_chunk(chunk, embedding_rule, dataset_config)
    if warm_start:
        logger.info('warm start')
        tcache = TrainCache.load(project_name, LSTM)
        train(tcache, dataset, model=None, config=train_config)
    else:
        tcache = TrainCache(project_name)  # type: ignore[var-annotated]
        model = LSTM(model_config)
        train(tcache, dataset, model=model, config=train_config)


def visualize_train_histories(project_name: str):
    project_dir = get_project_dir(project_name)
    fnames = os.listdir(project_dir)

    plot_dir = get_subproject_dir(project_name, 'train_history')
    for fname in fnames:
        m = re.match(r'.*TrainCache.*', fname)
        if m is not None:
            pickle_file = os.path.join(project_dir, fname)

            with open(pickle_file, 'rb') as f:
                tcache: TrainCache = pickle.load(f)
                fig, ax = plt.subplots()
                tcache.visualize((fig, ax))
                image_file = os.path.join(plot_dir, fname + '.png')
                fig.savefig(image_file)
                print('saved to {}'.format(image_file))


def visualize_image_reconstruction(
        project_name: str, n_vis: int = 5, ae_type: Type[AutoEncoderBase] = AutoEncoder):

    chunk = MultiEpisodeChunk.load(project_name)
    chunk_intact = chunk.get_intact_chunk()
    chunk_not_intact = chunk.get_not_intact_chunk()

    tcache = TrainCache.load(project_name, ae_type)
    assert tcache.best_model is not None
    image_type = tcache.best_model.image_type  # type: ignore[union-attr]
    no_aug = AutoEncoderDatasetConfig(0)  # to feed not randomized image
    dataset_intact = AutoEncoderDataset.from_chunk(chunk_intact, image_type, no_aug)
    dataset_not_intact = AutoEncoderDataset.from_chunk(chunk_not_intact, image_type, no_aug)

    for dataset, postfix in zip([dataset_intact, dataset_not_intact], ['intact', 'not_intact']):
        idxes = list(range(len(dataset)))
        random.shuffle(idxes)
        idxes_test = idxes[:min(n_vis, len(dataset))]

        for i, idx in enumerate(idxes_test):

            image_torch = dataset[idx].unsqueeze(dim=0)
            image_torch_reconstructed = tcache.best_model(image_torch)

            img = dataset.image_type.from_tensor(image_torch.squeeze(dim=0))
            img_reconstructed = dataset.image_type.from_tensor(image_torch_reconstructed.squeeze(dim=0))

            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle('left: original, right: reconstructed')
            ax1.imshow(img.to_rgb()._data)
            ax2.imshow(img_reconstructed.to_rgb()._data)
            save_dir = get_subproject_dir(project_name, 'autoencoder_result')

            full_file_name = os.path.join(save_dir, 'result-{}-{}.png'.format(postfix, i))
            plt.savefig(full_file_name)


def add_text_to_image(image: ImageBase, text: str, color: str):
    fig = plt.figure(tight_layout={'pad': 0})
    ax = plt.subplot(1, 1, 1)
    ax.axis('off')
    ax.imshow(image.to_rgb()._data)
    bbox = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax.text(7, 1, text, fontsize=15, color=color, verticalalignment='top', bbox=bbox)
    fig.canvas.draw()
    fig.canvas.flush_events()
    return canvas_to_ndarray(fig)


def visualize_lstm_propagation(project_name: str, propagator: Propagator, n_prop: int):

    chunk = MultiEpisodeChunk.load(project_name).get_intact_chunk()
    save_dir = get_subproject_dir(project_name, 'lstm_result')

    for idx, edata in enumerate(chunk):
        episode_data = chunk[idx]

        image_type = None
        for key, embedder in propagator.embed_rule.items():
            if issubclass(key, ImageBase):
                image_type = key
        assert image_type is not None

        n_feed = 20
        fed_avs = episode_data.get_sequence_by_type(AngleVector)[:n_feed]
        fed_images = episode_data.get_sequence_by_type(image_type)[:n_feed]

        use_gripper = GripperState in propagator.embed_rule

        if use_gripper:
            fed_grippers = episode_data.get_sequence_by_type(GripperState)[:n_feed]

        print("start lstm propagation")
        for i in range(n_feed):
            elem_dict = ElementDict([fed_avs[i], fed_images[i]])
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

        n_av_dim = chunk.spec.type_shape_table[AngleVector][0]
        n_gs_dim = chunk.spec.type_shape_table[GripperState][0] if use_gripper else 0
        fig, axs = plt.subplots(n_av_dim + n_gs_dim, 1)

        # plot angle vectors
        av_seq_gt = episode_data.get_sequence_by_type(AngleVector)
        np_av_seq_gt = np.array([av.numpy() for av in av_seq_gt])
        np_av_seq_pred = np.concatenate((np_av_seq_gt[:n_feed], np.array(pred_avs)), axis=0)

        i_dim = 0
        for i_av_dim in range(n_av_dim):
            axs[i_dim].plot(np_av_seq_gt[:, i_av_dim], color='blue', lw=1)
            axs[i_dim].plot(np_av_seq_pred[:, i_av_dim], color='red', lw=1)

            # determine axes min max
            conc = np.hstack((np_av_seq_gt[:, i_av_dim], np_av_seq_pred[:, i_av_dim]))
            y_min = np.min(conc)
            y_max = np.max(conc)
            diff = y_max - y_min
            axs[i_dim].set_ylim([y_min - diff * 0.1, y_max + diff * 0.1])
            axs[i_dim].set_title('AngleVector dim {}'.format(i_av_dim), fontsize=5, pad=0.0)
            i_dim += 1

        if use_gripper:
            gs_seq_gt = episode_data.get_sequence_by_type(GripperState)
            np_gs_seq_gt = np.array([gs.numpy() for gs in gs_seq_gt])
            np_gs_seq_pred = np.concatenate((np_gs_seq_gt[:n_feed], np.array(pred_gss)), axis=0)
            for i_gs_dim in range(n_gs_dim):
                axs[i_dim].plot(np_gs_seq_gt[:, i_gs_dim], color='blue', lw=1)
                axs[i_dim].plot(np_gs_seq_pred[:, i_gs_dim], color='red', lw=1)

                # determine axes min max
                conc = np.hstack((np_gs_seq_gt[:, i_gs_dim], np_gs_seq_pred[:, i_gs_dim]))
                y_min = np.min(conc)
                y_max = np.max(conc)
                diff = y_max - y_min
                axs[i_dim].set_ylim([y_min - diff * 0.1, y_max + diff * 0.1])
                axs[i_dim].set_title('GripperState dim {}'.format(i_gs_dim), fontsize=5, pad=0.0)
                i_dim += 1

        for ax in axs:
            ax.grid()

        filename = os.path.join(save_dir, 'seq-{}{}.png'.format(AngleVector.__name__, idx))
        fig.savefig(filename, format='png', dpi=300)
        print('saved to {}'.format(filename))

        # save gif image
        print("adding text to images...")
        fed_images_with_text = [add_text_to_image(image, 'fed (original) image)', 'blue') for image in fed_images]
        clamp = lambda x: max(min(x, 1.0), 0.0)  # noqa
        pred_images_with_text = [
            add_text_to_image(image, 'predicted image (prob-terminated={:.2f})'.format(clamp(flag)), 'green')
            for image, flag in zip(pred_images, pred_flags)]

        images_with_text = fed_images_with_text + pred_images_with_text

        full_file_name = os.path.join(save_dir, 'result-image{}.gif'.format(idx))
        assert ImageSequenceClip is not None, 'check if your moviepy is properly installed'
        clip = ImageSequenceClip(images_with_text, fps=20)
        clip.write_gif(full_file_name, fps=20)
