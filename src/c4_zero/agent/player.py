from logging import getLogger
from typing import List, Optional

from c4_zero.agent.player_util import EvaluationItem
from c4_zero.agent.player_util import ThoughtItem, MCTSInfo
from c4_zero.env.c4_util import ActionWithEvaluation
from c4_zero.env.data import MoveData
from c4_zero.env.environment import Env

logger = getLogger(__name__)


class BasePlayer:
    """プレイヤーの基本クラス"""

    def __init__(self):
        pass

    def action(self, env_black: Env, callback_in_mtcs=None) -> int:
        """黒番の盤面が与えられたときの行動を返す"""
        raise NotImplementedError

    def action_with_evaluation(self, env_black: Env, callback_in_mtcs=None) -> ActionWithEvaluation:
        """黒番の盤面が与えられたときの行動と評価を返す"""
        raise NotImplementedError

    def ask_evaluation(self, env_black: Env) -> EvaluationItem:
        """現時点のP、Vと次の局面のそれぞれのVを出力する（デバッグ用）"""
        raise NotImplementedError

    def ask_thought(self, env_black: Env) -> Optional[ThoughtItem]:
        """その局面についての思考の経過を返す"""
        raise NotImplementedError

    def finish_game(self, z: float) -> None:
        """ゲームの終了処理"""
        raise NotImplementedError

    @property
    def moves_data(self) -> List[MoveData]:
        raise NotImplementedError

    @property
    def resigned(self) -> bool:
        raise NotImplementedError

    @staticmethod
    def create_mtcs_info() -> MCTSInfo:
        """MCTSの情報を作成する"""
        raise NotImplementedError
