from logging import getLogger
from typing import Optional, List

import numpy as np

logger = getLogger(__name__)

from c4_zero.env.c4_util import Player, Winner
from c4_zero.env.c4_util import CounterKey
from c4_zero.env.data import MoveData, InferenceData


class Env:
    """環境

    盤面、手番などの情報を保持する
    """
    board: 'Board'
    next_player: Player
    turn: int
    done: bool
    winner: Winner

    def __init__(self):
        self.board = None
        self.next_player = None
        self.turn = 0
        self.done = False
        self.winner = None

    def copy(self) -> 'Env':
        """コピーメソッド"""
        raise NotImplementedError

    def reset(self) -> None:
        """初期盤面を作成する"""
        raise NotImplementedError

    def step(self, action: Optional[int]) -> None:
        """次の手を指定して盤面を進める"""
        raise NotImplementedError

    def render(self) -> None:
        """盤面を表示する"""
        raise NotImplementedError

    def get_env_black(self) -> "Env":
        """黒番の盤面を取得する。そうでない場合、反転して黒番とした盤面を取得する"""
        raise NotImplementedError

    def counter_key(self) -> CounterKey:
        """環境を表現するキーを取得する"""
        raise NotImplementedError

    def another_side_counter_key(self) -> CounterKey:
        """反転した盤面を表現するキーを取得する"""
        raise NotImplementedError

    def moves_data(self, policy) -> List[MoveData]:
        """学習用データを作成する"""
        raise NotImplementedError

    def legal_moves(self) -> np.array:
        """合法手を出力する"""
        raise NotImplementedError

    def to_inference(self, **kwargs) -> InferenceData:
        """変換を考慮して推論に使うデータとする"""
        raise NotImplementedError

    def leaf_p_restore_transform(self, leaf_p, **kwargs) -> np.array:
        """変換を考慮したP値を出力する"""
        raise NotImplementedError
