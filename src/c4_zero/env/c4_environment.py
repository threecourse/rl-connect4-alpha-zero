from logging import getLogger
from typing import Optional, List

import numpy as np
import pyximport

from c4_zero.env.c4_util import BLACK, WHITE, BLACK_STR, WHITE_STR, EMPTY_STR
from c4_zero.env.c4_util import Player, another_player, CounterKey, BH, BW, MOVES, A
from c4_zero.env.c4_util import Winner
from c4_zero.env.data import MoveData, InferenceData
from c4_zero.env.environment import Env

pyximport.install(setup_args={'include_dirs': [np.get_include()]})
from c4_zero.env.board import Board, create_board_from_array2d
from c4_zero.env.solver.solver import SolverClient, SolverServer
from c4_zero.config import EnvironmentConfig

logger = getLogger(__name__)


class C4Env(Env):
    """connect4の環境

    スレッドごとにコンフィグの設定・サーバの初期化を行う必要があるので注意が必要
    """
    env_config: EnvironmentConfig
    initialized: bool
    env_config = None
    initialized = False

    @classmethod
    def initialize(cls, env_config: EnvironmentConfig):
        cls.env_config = env_config
        if cls.env_config.env_solver_turn:
            solver_server = SolverServer()
            solver_server.start_serve()
            cls.solver = solver_server.get_api_client()
        else:
            cls.solver = None
        cls.initialized = True

    def __init__(self):
        super().__init__()

    @classmethod
    def create_init(cls) -> Env:
        """初期状態の作成を行う"""
        game = cls()
        game.reset()
        return game

    @classmethod
    def create(cls, board: Board, next_player: Player, turn: int) -> Env:
        """盤面・ターン・手番を指定して状態の作成を行う"""
        game = cls()
        game.board = board.copy()
        game.next_player = next_player
        game.turn = turn
        game.done = False
        game.winner = None
        return game

    def copy(self) -> Env:
        """コピーメソッド"""
        return C4Env.create(self.board, self.next_player, self.turn)

    def reset(self) -> None:
        """初期盤面を作成する"""
        self.board = Board()
        self.next_player = Player.black
        self.turn = 0
        self.done = False
        self.winner = None

    def step(self, action: Optional[int]) -> None:
        """次の手を指定して盤面を進める

        :param int|None action: move pos=0 ~ 63 (0=top left, 7 top right, 63 bottom right), None is resign
        """
        assert action is None or 0 <= action < MOVES, f"Illegal action={action}"
        if not self.initialized:
            raise Exception("environment not initialized")

        if action is None:
            # 投了の場合
            win_player = another_player(self.next_player)
            if win_player == Player.black:
                self.winner = Winner.black
            else:
                self.winner = Winner.white
            self._game_over()
            return

        # 投了以外の場合
        y, x = action // BW, action % BW
        if self.next_player == Player.black:
            self.board.set_value(y, x, BLACK)
        else:
            self.board.set_value(y, x, WHITE)
        self.turn += 1
        self._change_to_next_player()

        # 局面評価
        black_count, white_count, moves = self.board.count_arrays()
        if self.turn != moves:
            raise Exception("popcountの計算に不整合がある")

        if black_count[A] > 0:
            self.winner = Winner.black
            self._game_over()
            return
        elif white_count[A] > 0:
            self.winner = Winner.white
            self._game_over()
            return
        else:
            if moves == BH * BW:
                self.winner = Winner.draw
                self._game_over()
                return

        # 勝敗が決定していない局面について、ソルバーによる勝敗判定を行う
        if self.solver and self.turn >= self.env_config.env_solver_turn:
            v = self.solver.solve_v(self)
            if (self.next_player == Player.black and v == 1) or \
                    (self.next_player == Player.white and v == -1):
                self.winner = Winner.black
                self._game_over()
            elif (self.next_player == Player.black and v == -1) or \
                    (self.next_player == Player.white and v == 1):
                self.winner = Winner.white
                self._game_over()
            else:
                self.winner = Winner.draw
                self._game_over()

    def render(self) -> None:
        """盤面を表示する"""
        print(f"next={self.next_player.name} turn={self.turn}")
        print(self.to_string())

    def get_env_black(self) -> "Env":
        """黒番の盤面を取得する。そうでない場合、反転して黒番とした盤面を取得する"""

        if self.next_player == Player.black:
            return C4Env.create(self.board, Player.black, self.turn)
        else:
            return self._get_env_flip()

    def _get_env_flip(self) -> "Env":
        """反転した盤面を取得する

        手番と盤面をそれぞれ反転させる
        """
        player = another_player(self.next_player)
        board = self.board.reversed_copy()
        return C4Env.create(board, player, self.turn)

    def counter_key(self) -> CounterKey:
        """環境を表現するキーを取得する"""
        return CounterKey(self.board.black, self.board.white, self.next_player.value)

    def another_side_counter_key(self) -> CounterKey:
        """反転した盤面を表現するキーを取得する"""
        return CounterKey(self.board.white, self.board.black, another_player(self.next_player).value)

    def moves_data(self, policy) -> List[MoveData]:
        """学習用データを作成する"""

        # 黒番のみから学習データを作成するものとする
        assert (self.next_player == Player.black)
        moves = []

        # 通常のデータ
        cell_ary = self.board.to_array1d()
        policy = policy.reshape((MOVES,))
        moves.append(MoveData(cell_ary, policy))

        # 左右をflipしたデータ
        cell_ary_filplr = np.fliplr(cell_ary.reshape((BH, BW))).reshape(MOVES)
        policy_filplr = np.fliplr(policy.reshape((BH, BW))).reshape(MOVES)
        moves.append(MoveData(cell_ary_filplr, policy_filplr))

        return moves

    def legal_moves(self) -> np.array:
        """合法手を出力する"""
        return self.board.legal_moves()

    def to_inference(self, **kwargs) -> InferenceData:
        """変換を考慮して推論に使うデータとする"""

        # predictをするための構造に変換する
        cells_ary = self.board.to_array1d().reshape((BH, BW)).astype(int)
        black_ary, white_ary = cells_ary == BLACK, cells_ary == WHITE
        state = [black_ary, white_ary] if self.next_player == Player.black else [white_ary, black_ary]
        data = InferenceData(np.array(state))
        return data

    def leaf_p_restore_transform(self, leaf_p, **kwargs) -> np.array:
        """変換を考慮したP値を出力する"""
        return leaf_p

    def _game_over(self) -> None:
        """ゲーム終了時の処理"""
        self.done = True

    def _change_to_next_player(self) -> None:
        """手番の入替"""
        self.next_player = another_player(self.next_player)

    @classmethod
    def from_string(cls, input_str):
        """文字列から盤面を作成する"""

        # 枠つきのBoardを入力とする
        lines = input_str.split("\n")
        assert (len(lines) == BH + 2)

        next_player_char = lines[0][1]
        assert (next_player_char in ["W", "B"])
        if next_player_char == "B":
            next_player = Player.black
        else:
            next_player = Player.white
        turn = int(lines[0][2:4])

        offset_y = 1
        offset_x = 1

        cell_ary = np.zeros((BH, BW), dtype=int)
        for y in range(BH):
            for x in range(BW):
                c = lines[y + offset_y][x + offset_x]
                if c == BLACK_STR:
                    cell_ary[y, x] = BLACK
                elif c == WHITE_STR:
                    cell_ary[y, x] = WHITE

        return cls.create(create_board_from_array2d(cell_ary), next_player, turn)

    def to_string(self):
        """盤面を文字列にする

        完全にはfrom_stringと対になっていない
        """
        ret = ""
        ret += "#"
        if self.next_player == Player.black:
            ret += "B"
        else:
            ret += "W"
        ret += "#" * (BW) + "\n"

        for y in range(BH):
            ret += "#"
            for x in range(BW):
                v = self.board.get_value(y, x)
                if v == BLACK:
                    ret += BLACK_STR
                elif v == WHITE:
                    ret += WHITE_STR
                else:
                    ret += EMPTY_STR
            ret += "#"
            ret += "\n"
        ret += "#" * (BW + 2) + "\n"
        return ret
