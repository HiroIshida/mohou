from abc import ABC, abstractmethod
import copy
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Type

from mohou.file import dump_object
from mohou.file import load_object
from mohou.types import ElementDict
from mohou.types import VectorBase
from mohou.types import HasAList
from mohou.propagator import Propagator
from mohou.default import create_default_propagator


_execution_data_path = Path("execution_data")


def get_execution_data_path() -> Path: 
    # to prevend modification 
    return copy.deepcopy(_execution_data_path)


@dataclass
class ExecutionData:
    real_seq: List[ElementDict]

    def dump(self, project_name: str):
        dump_object(self, project_name, subpath=_execution_data_path)

    @classmethod
    def load(self, project_name: str):
        load_object(self, project_name, subpath=_execution_data_path)


@dataclass
class ExecutionDataPlotter:
    propagator: Propagator
    real_seq: List[ElementDict]

    @classmethod
    def load(
        cls,
        project_name: str,
        propagator: Optional[Propagator] = None) -> 'ExecutionDataPlotter':

        if propagator is None:
            create_default_propagator(project_name=project_name)

    def get_shape(self, elem_type: Type[VectorBase]) -> Tuple[int, ...]:
        assert len(self.real_seq) > 0
        elem_dict = self.real_seq[0]
        return elem_dict[elem_type].shape

    def plot_prediction(
            self,
            elem_type: Type[VectorBase],
            n_horizon: int = 10) -> None:
        assert issubclass(elem_type, VectorBase)

        pred_seq_seq = []
        for real in self.real_seq:
            self.propagator.feed(real)
            pred_seq = self.propagator.predict(n_horizon)
            pred_seq_seq.append(pred_seq)

        real_seq_arr2d = self.seq_to_2dimarray(self.real_seq, elem_type)

        pred_seqs_arr3d = np.array(
            [self.seq_to_2dimarray(seq, elem_type) for seq in pred_seq_seq])


        n_dim = self.get_shape(elem_type)[0]
        fig, axes = plt.subplots(n_dim, 1)

        for i in range(n_dim):
            ax = axes[i]
            scalar_real_seq = real_seq_arr2d[:, i]
            ax.plot(scalar_real_seq)

        for ax in axes:
            ax.grid()

        plt.show()

    @staticmethod
    def seq_to_2dimarray(seq: List[ElementDict], elem_type: Type[VectorBase]):
        return np.array([e[elem_type].numpy() for e in seq])
