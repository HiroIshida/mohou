import numpy as np
import torch
from typing import List

from mohou.constant import CONTINUE_FLAG_VALUE
from mohou.types import ElementDict, AngleVector
from mohou.model import AutoEncoder, LSTM
from mohou.embedder import IdenticalEmbedder
from mohou.embedding_rule import EmbeddingRule, create_embedding_rule
from mohou.trainer import TrainCache


class Propagator:
    lstm: LSTM
    embed_rule: EmbeddingRule
    fed_state_list: List[np.ndarray]  # eatch state is equipped with flag

    def __init__(self, lstm: LSTM, embed_rule: EmbeddingRule):
        self.lstm = lstm
        self.embed_rule = embed_rule
        self.fed_state_list = []

    def feed(self, elem_dict: ElementDict):
        state = self.embed_rule.apply(elem_dict)
        state_with_flag = np.hstack((state, CONTINUE_FLAG_VALUE))
        self.fed_state_list.append(state_with_flag)

    def predict(self, n_prop: int) -> List[ElementDict]:
        pred_state_list = self._predict(n_prop)
        elem_dict_list = []
        for pred_state in pred_state_list:
            elem_dict = self.embed_rule.inverse_apply(pred_state)
            elem_dict_list.append(elem_dict)
        return elem_dict_list

    def _predict(self, n_prop: int) -> List[np.ndarray]:
        pred_state_list: List[np.ndarray] = []

        for i in range(n_prop):
            states = np.vstack(self.fed_state_list + pred_state_list)
            states_torch = torch.from_numpy(states).float().unsqueeze(dim=0)
            state_pred_torch: torch.Tensor = self.lstm(states_torch)[:, -1]
            state_pred = state_pred_torch.squeeze().detach().numpy()

            # force override
            state_pred[-1] = CONTINUE_FLAG_VALUE

            pred_state_list.append(state_pred)

        return pred_state_list


def create_default_propagator(project_name: str, n_angle_vector: int):
    tcache_autoencoder = TrainCache.load(project_name, AutoEncoder)
    tcach_lstm = TrainCache.load(project_name, LSTM)

    image_embed_func = tcache_autoencoder.best_model.get_embedder()
    av_idendical_func = IdenticalEmbedder(AngleVector, n_angle_vector)

    embed_rule = create_embedding_rule(image_embed_func, av_idendical_func)
    propagator = Propagator(tcach_lstm.best_model, embed_rule)

    return propagator
