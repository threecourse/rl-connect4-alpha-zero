from asyncio import Future
from dataclasses import dataclass
from typing import Any, Optional, Dict

import numpy as np

from c4_zero.env.c4_util import CounterKey, MOVES
from c4_zero.env.data import InferenceData


@dataclass
class QueueItem:
    data: InferenceData
    future: Future


@dataclass
class ThoughtItem:
    """ある盤面でactionを決定したときの情報

    action: 選択した手
    ary_p: P値
    ary_q: Q値
    ary_n: N値
    """
    action: Optional[int]
    ary_p: np.array  # (BH * BW,)
    ary_q: np.array  # (BH * BW,)
    ary_n: np.array  # (BH * BW,)


@dataclass
class EvaluationItem:
    """ある盤面を評価した情報

    curr_policy: ある盤面のポリシーネットワークの値
    curr_value: ある盤面のバリューネットワークの値
    values_after_move: 一手進めたあとのそれぞれの盤面のバリューネットワークの値
    """
    curr_policy: np.array
    curr_value: float
    values_after_move: np.array


@dataclass
class CallbackInMCTS:
    per_sim: int
    callback: Any


@dataclass
class MCTSInfo:
    var_n: Dict[CounterKey, np.ndarray]
    var_w: Dict[CounterKey, np.ndarray]
    var_p: Dict[CounterKey, np.ndarray]


class AlphaZeroPlayerUtil:

    @classmethod
    def normalize_by_temparature(cls, p: np.array, temperature: float = 1) -> np.array:
        """値を温度で標準化する"""
        pp = np.power(p, temperature)
        return pp / np.sum(pp)

    @classmethod
    def dirichlet_noise_of_mask(cls, moves_array, alpha):
        """ディリクレ分布によるノイズを与える"""
        num_1 = np.sum(moves_array)
        noise = list(np.random.dirichlet([alpha] * num_1))
        ret_list = []
        for i in range(MOVES):
            if moves_array[i] == 1:
                ret_list.append(noise.pop(0))
            else:
                ret_list.append(0)
        return np.array(ret_list)
