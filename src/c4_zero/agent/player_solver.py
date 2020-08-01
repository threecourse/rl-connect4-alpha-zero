import asyncio
from _asyncio import Future
from asyncio.queues import Queue
from collections import defaultdict
from logging import getLogger
from typing import Any, Dict, List, Optional, Set

import numpy as np

from c4_zero.agent.api import InferenceAPIBase, InferenceSimpleAPI
from c4_zero.agent.model.model import C4Model
from c4_zero.agent.player import BasePlayer
from c4_zero.agent.player_util import AlphaZeroPlayerUtil
from c4_zero.agent.player_util import QueueItem, ThoughtItem, CallbackInMCTS, MCTSInfo, EvaluationItem
from c4_zero.config import Config, PlayerConfig
from c4_zero.env.c4_util import MOVES, Player, Winner, CounterKey, ActionWithEvaluation
from c4_zero.env.data import MoveData, InferenceData
from c4_zero.env.environment import Env
from c4_zero.env.solver.solver import SolverServer, SolverClient

logger = getLogger(__name__)


class SolverPlayer:
    """ソルバーを利用したプレイヤー"""
    config: Config
    player_config: PlayerConfig
    solver: SolverClient

    def __init__(self, config: Config, player_config: PlayerConfig, solver: SolverClient):
        super().__init__()
        self.config = config
        self.player_config = player_config
        self.solver = solver or self._create_solver()
        self._moves_data = []

    def _create_solver(self):
        solver_server = SolverServer()
        solver_server.start_serve()
        solver = solver_server.get_api_client()
        return solver

    def action(self, env_black: Env, callback_in_mtcs=None) -> int:
        """黒番の盤面が与えられたときの行動を返す"""
        assert (env_black.next_player == Player.black)
        action_with_eval = self.action_with_evaluation(env_black, callback_in_mtcs=callback_in_mtcs)
        return action_with_eval.action

    def action_with_evaluation(self, env_black: Env, callback_in_mtcs=None) -> ActionWithEvaluation:
        """黒番の盤面が与えられたときの行動と評価を返す"""
        assert (env_black.next_player == Player.black)
        v, policy_best, policy_random = self.solver.solve_move(env_black)
        random_ratio = self.player_config.solver_player_random_ratio
        if np.random.random() < random_ratio:
            # ランダム選択
            action = int(np.random.choice(range(MOVES), p=policy_random))
        else:
            # ベストな手を選ぶ
            action = int(np.random.choice(range(MOVES), p=policy_best))
        return ActionWithEvaluation(action=action, n=999, q=v)

    def ask_evaluation(self, env_black: Env) -> EvaluationItem:
        """現時点のP、Vと次の局面のそれぞれのVを出力する（デバッグ用）"""
        return None

    def ask_thought(self, env_black: Env) -> Optional[ThoughtItem]:
        """その局面についての思考の経過を返す"""
        return None

    def finish_game(self, z: float) -> None:
        # self-playに利用しない前提では不要
        raise NotImplementedError

    @property
    def moves_data(self) -> List[MoveData]:
        # self-playに利用しない前提では不要
        raise NotImplementedError

    @property
    def resigned(self) -> bool:
        # self-playに利用しない前提では不要
        raise NotImplementedError

    @staticmethod
    def create_mtcs_info() -> MCTSInfo:
        """MCTSの情報を作成する"""
        raise NotImplementedError
