import os
import subprocess
from logging import getLogger
from multiprocessing import Pipe, connection
from subprocess import PIPE
from threading import Thread
from typing import List
from typing import NoReturn, Optional, Tuple

import numpy as np

from c4_zero.env.c4_util import MOVES

BH = 6
BW = 7
MOVE_NONE = 0
MOVE_WIN = 1
MOVE_ILLEGAL = 2
CP = 0  # current
OP = 1  # opponent

logger = getLogger(__name__)


class SolverServer:
    # 非同期対応は行っていない
    cache_max = 1000000

    def __init__(self):
        self.connections = []
        self.cache = {}
        script_dir = os.path.join(os.path.dirname(__file__), "solver-pascal")
        solver_path = "./c4solver"
        self.proc = subprocess.Popen(solver_path, cwd=script_dir, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE,
                                     universal_newlines=True)

    def get_api_client(self) -> 'SolverClient':
        me, you = Pipe()
        self.connections.append(me)
        return SolverClient(you)

    def start_serve(self) -> None:
        # logger.info("start solver server")
        prediction_worker = Thread(target=self._worker, name="worker")
        prediction_worker.daemon = True
        prediction_worker.start()

    def _worker(self) -> NoReturn:
        while True:
            # サーバに送られたデータを確認する
            ready_conns = connection.wait(self.connections, timeout=0.001)  # type: List[connection.Connection]
            if not ready_conns:
                continue

            # それぞれ計算して返す
            for conn in ready_conns:
                try:
                    x = conn.recv()
                    r = self._calc_cache(x)
                    conn.send(r)
                except EOFError:
                    self.connections.remove(conn)
                    logger.debug(f"solver client connection removed")

    def _calc_cache(self, s: str) -> Optional[int]:
        # 大きすぎる場合はキャッシュをクリアする
        if len(self.cache) > self.cache_max:
            self.cache = {}
        if not s in self.cache:
            ret = self._calc(s)
            self.cache[s] = ret
        return self.cache[s]

    def _calc(self, s: str) -> Optional[int]:
        self.proc.stdin.write(s + "\n")
        self.proc.stdin.flush()
        self.proc.stdout.flush()
        out = self.proc.stdout.readline().rstrip()
        if len(out) == 0:
            return None  # error
        else:
            return int(out.split(' ')[1])


class SolverClient:
    def __init__(self, conn: connection.Connection):
        self.connection = conn

    def send(self, x: str) -> Optional[int]:
        self.connection.send(x)
        score = self.connection.recv()
        return score

    def solve_v(self, env: 'C4Env') -> int:
        """盤面のスコアを返す

        手番ベースの盤面とし、その手番が勝っているか否かを返す
        終局前の盤面という前提
        """
        s_ary = env.to_inference().s_ary
        state_str = self._to_state_str(s_ary)
        score = self.send(state_str)
        if score is None:
            raise Exception("illegal solver input")
        score_binary = int(np.sign(score))
        assert (score_binary in [-1, 0, 1])
        return score_binary

    def solve_move(self, env: 'C4Env') -> Tuple[int, np.array, np.array]:
        """盤面のスコアおよび各手を指すべき確率を返す

        手番ベースの盤面とし、その手番が勝っているか否かと各手を指すべき確率を返す
        終局前の盤面という前提
        """
        s_ary = env.to_inference().s_ary
        actions, is_wins = ActionChecker.get_action_candidates(s_ary)
        assert (len(actions) > 0)

        scores = []
        for action, is_wins in zip(actions, is_wins):
            if is_wins:
                # その手で勝利する場合
                score = 1
            else:
                state_str = self._to_state_str_action(s_ary, action)
                score_raw_opponent = self.send(state_str)
                if score_raw_opponent is None:
                    raise Exception("illegal solver input")
                score_raw = score_raw_opponent * (-1)
                score = int(np.sign(score_raw))
            scores.append(score)

        v = max(scores)
        best_actions = [action for action, score in zip(actions, scores) if score == v]
        policy_best = np.zeros(MOVES)
        policy_random = np.zeros(MOVES)
        for action in best_actions:
            policy_best[action] = 1.0 / len(best_actions)
        for action in actions:
            policy_random[action] = 1.0 / len(actions)

        return v, policy_best, policy_random

    def _to_state_str(self, s_ary: np.array) -> str:
        lst = ['B']
        for y in range(BH):
            for x in range(BW):
                if s_ary[CP, y, x] == 1:
                    lst.append('o')
                elif s_ary[OP, y, x] == 1:
                    lst.append('x')
                else:
                    lst.append('.')
        state_str = "".join(lst)
        return state_str

    def _to_state_str_action(self, s_ary: np.array, action: int) -> str:
        lst = ['B']
        for y in range(BH):
            for x in range(BW):
                if y * BW + x == action:
                    lst.append('x')
                elif s_ary[CP, y, x] == 1:
                    lst.append('x')  # 相手の手番にするので、逆にする
                elif s_ary[OP, y, x] == 1:
                    lst.append('o')  # 相手の手番にするので、逆にする
                else:
                    lst.append('.')
        state_str = "".join(lst)
        return state_str


class ActionChecker:

    @classmethod
    def get_action_candidates(cls, s_ary: np.array) -> Tuple[List[int], List[bool]]:
        """合法手およびその手で勝利するか否かを求める"""
        actions: List[int] = []
        is_wins: List[bool] = []
        for x in range(BW):
            y, result = cls._action_result(s_ary, x)
            if result != MOVE_ILLEGAL:
                actions.append(y * BW + x)
                is_wins.append(result == MOVE_WIN)
        return actions, is_wins

    @classmethod
    def _action_result(cls, s_ary: np.array, x: int) -> Tuple[int, int]:
        """その局面でのxに対応するyおよび結果を求める"""

        # set y
        y = BH - 1
        while True:
            if y < 0 or (s_ary[CP, y, x] == 0 and s_ary[OP, y, x] == 0):
                break
            else:
                y -= 1

        # check is-legit or not
        if y < 0:
            return -1, MOVE_ILLEGAL

        # check is-win or not
        assert (s_ary[CP, y, x] == 0 and s_ary[OP, y, x] == 0)
        sv = cls._check_series(s_ary, y, x, 1, 0) + cls._check_series(s_ary, y, x, -1, 0)
        sh = cls._check_series(s_ary, y, x, 0, 1) + cls._check_series(s_ary, y, x, 0, -1)
        sd1 = cls._check_series(s_ary, y, x, 1, 1) + cls._check_series(s_ary, y, x, -1, -1)
        sd2 = cls._check_series(s_ary, y, x, 1, -1) + cls._check_series(s_ary, y, x, -1, 1)
        if sv >= 3 or sh >= 3 or sd1 >= 3 or sd2 >= 3:
            return y, MOVE_WIN
        else:
            return y, MOVE_NONE

    @classmethod
    def _check_series(cls, s_ary: np.array, y: int, x: int, dy: int, dx: int) -> int:
        """ある点から特定の方向に動かしたときのcurrent playerの連続について"""
        c = 0
        while True:
            y += dy
            x += dx
            if x < 0 or x >= BW or y < 0 or y >= BH:
                break
            if s_ary[CP, y, x] == 0:
                break
            c += 1
        return c
