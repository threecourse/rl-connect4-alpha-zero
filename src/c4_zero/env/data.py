from datetime import datetime
from typing import Optional, List

import numpy as np
from sklearn.externals import joblib

from c4_zero.env.c4_util import BH, BW, BLACK, WHITE
from c4_zero.env.c4_util import Player


class InferenceData:
    """推論用のデータ（1つの状態をnp.arrayの形式にしたもの）"""
    s_ary: np.array

    def __init__(self, s_ary: np.array):
        # データの個数は1つとする
        assert (s_ary.shape == (2, BH, BW))
        self.s_ary = s_ary

    @classmethod
    def to_x(cls, data_list: List['InferenceData']) -> np.array:
        x = np.array([data.s_ary for data in data_list])
        return x


class InferenceResult:
    """推論の結果"""

    p_ary: np.array
    v_ary: np.array

    def __init__(self, p_ary: np.array, v_ary: np.array):
        L = p_ary.shape[0]
        assert p_ary.shape == (L, BH * BW)
        assert v_ary.shape == (L, 1)
        self.p_ary = p_ary
        self.v_ary = v_ary


class TrainingData:
    """学習用のデータ（複数の状態および結果をnp.arrayの形式にしたもの）"""

    s_ary: np.array
    p_ary: np.array
    z_ary: np.array

    def __init__(self, s_ary: np.array, p_ary: np.array, z_ary: np.array):
        L = s_ary.shape[0]
        # print(s_ary.shape, p_ary.shape, z_ary.shape)
        assert (s_ary.shape == (L, 2, BH, BW))
        assert (p_ary.shape == (L, BH * BW))
        assert (z_ary.shape == (L,))

        self.s_ary = s_ary
        self.p_ary = p_ary
        self.z_ary = z_ary

    def __len__(self):
        return self.s_ary.shape[0]

    @classmethod
    def concatenate(cls, training_data_list: List['TrainingData']) -> Optional['TrainingData']:
        s_ary_list, p_ary_list, z_ary_list = [], [], []
        for training_data in training_data_list:
            s_ary_list.append(training_data.s_ary)
            p_ary_list.append(training_data.p_ary)
            z_ary_list.append(training_data.z_ary)

        if s_ary_list:
            s_ary = np.concatenate(s_ary_list)
            p_ary = np.concatenate(p_ary_list)
            z_ary = np.concatenate(z_ary_list)
            return TrainingData(s_ary, p_ary, z_ary)
        else:
            return None


class MoveData:
    """学習用のデータ（一手を表す）

    プレイヤーが進行中に学習用データを保存し、ワーカーがゲーム終了時に収集する
    """

    state: np.array
    policy: np.array
    z: Optional[float]

    def __init__(self, state: np.array, policy: np.array, z: Optional[float] = None):
        assert (state.shape == (BH * BW,))
        assert (policy.shape == (BH * BW,))
        self.state = state
        self.policy = policy
        self.z = z

    @property
    def is_draw(self) -> bool:
        return self.z == 0

    @property
    def is_not_draw(self) -> bool:
        return self.z != 0

    @classmethod
    def write_to_file(cls, path: str, moves_data: List['MoveData']) -> None:
        """データの書き込みを行う"""
        joblib.dump(moves_data, path, compress=True)

    @classmethod
    def read_from_file(cls, path: str) -> List['MoveData']:
        """データの読み込みを行う"""
        return joblib.load(path)

    @classmethod
    def convert_to_training_data(cls, moves_data: List['MoveData']) -> TrainingData:
        state_list = []
        policy_list = []
        z_list = []
        for move_data in moves_data:
            state = move_data._converted_state()
            state_list.append(state)
            policy_list.append(move_data.policy)
            z_list.append(move_data.z)
        training_data = TrainingData(np.array(state_list), np.array(policy_list), np.array(z_list))
        return training_data

    def _converted_state(self) -> List[np.array]:
        """学習用データに変換する"""
        cell_ary = np.array(self.state, dtype=int)
        black_ary, white_ary = cell_ary == BLACK, cell_ary == WHITE
        state = [black_ary.reshape((BH, BW)), white_ary.reshape((BH, BW))]
        return state


class MoveGGFHistory:
    """自己対戦における、棋譜の作成用の過去の手や思考の履歴の保持"""
    moves: List[str]

    def __init__(self):
        self.moves = []

    def add_move(self, env: 'Env', action_with_evaluation: 'ActionWithEvaluation') -> None:
        """手による情報を保存する

        :param Env env:
        :param ActionWithEvaluation action_with_evaluation:
        :return:
        """
        if action_with_evaluation.action is None:
            return  # resigned

        # 調整のパスを設定する
        if len(self.moves) % 2 == 0:
            if env.next_player == Player.white:
                self.moves.append(GGF.convert_action_to_move(None))
        else:
            if env.next_player == Player.black:
                self.moves.append(GGF.convert_action_to_move(None))

        # 手を追加する
        move = f"{GGF.convert_action_to_move(action_with_evaluation.action)}/{action_with_evaluation.q * 10}/{action_with_evaluation.n}"
        self.moves.append(move)

    def to_ggf_string(self, black_name: str = None, white_name: str = None) -> str:
        """ggf形式の棋譜に変換する"""
        return GGF.make_ggf_string(black_name=black_name, white_name=white_name, moves=self.moves)


class GGF:
    """棋譜の形式"""

    @classmethod
    def convert_action_to_move(cls, action: Optional[int]) -> str:
        """actionを文字列に変換する"""
        if action is None:
            return "PA"
        y = action // BH
        x = action % BW
        return chr(ord("A") + y) + str(x + 1)

    @classmethod
    def make_ggf_string(cls, black_name: Optional[str] = None, white_name: Optional[str] = None,
                        dt: Optional[datetime] = None,
                        moves: List[str] = None, result: Optional[str] = None, think_time_sec: int = 60):
        """棋譜を作成する"""
        ggf = '(;GM[C4]PC[C4AZ]DT[%(datetime)s]PB[%(black_name)s]PW[%(white_name)s]RE[%(result)s]TI[%(time)s]%(move_list)s;)'
        dt = dt or datetime.utcnow()

        move_list = []
        for i, move in enumerate(moves or []):
            if i % 2 == 0:
                move_list.append(f"B[{move}]")
            else:
                move_list.append(f"W[{move}]")

        params = dict(
            black_name=black_name or "black",
            white_name=white_name or "white",
            result=result or '?',
            datetime=dt.strftime("%Y.%m.%d_%H:%M:%S.%Z"),
            time=f"{think_time_sec // 60}:{think_time_sec % 60}",
            move_list="".join(move_list),
        )
        return ggf % params
