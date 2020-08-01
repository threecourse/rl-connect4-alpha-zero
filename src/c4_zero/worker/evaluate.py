import os
import time
from dataclasses import dataclass
from datetime import datetime
from logging import getLogger
from typing import NoReturn

from c4_zero.agent.model.model_util import ModelUtil
from c4_zero.agent.player import BasePlayer
from c4_zero.agent.player_creator import create_player
from c4_zero.config import Config
from c4_zero.env.c4_environment import C4Env
from c4_zero.env.c4_util import Player, Winner, BW
from c4_zero.env.solver.solver import SolverServer
from c4_zero.lib import tensorflow_util
from c4_zero.lib.tensorboard_logger import TensorBoardLogger
from c4_zero.worker.optimize import OptimizeStatus

logger = getLogger(__name__)


def start(config: Config):
    tensorflow_util.set_session_config(per_process_gpu_memory_fraction=0.2)
    return EvaluateWorker(config).start()


@dataclass
class EvaluateResult:
    games_n: int
    random_rate: int  # [0, 10, 30, 100]
    win_rate: float
    win_rate_black: float
    win_rate_white: float


class EvaluateWorker:
    """評価を行う

    元々はモデル同士の対戦での評価によるモデルの入替えだが、ここではソルバーとの勝率を単純に評価するのみとしている
    """

    def __init__(self, config: Config):
        self.config = config
        self.kifu_path = None
        self.tensor_board_logger = TensorBoardLogger(self.config.resource.evaluate_log_dir)
        self.last_step = 0
        self.current_step = 0

        # ソルバーは使い回すことにする
        solver_server = SolverServer()
        solver_server.start_serve()
        self.solver = solver_server.get_api_client()

    def start(self) -> NoReturn:
        # envの初期化
        C4Env.initialize(self.config.env)

        # 定期的に実行する
        while True:
            self.current_step = self.get_step()
            if self.current_step >= self.last_step + self.config.evaluate.evaluate_per_steps:
                self.evaluate_models()
                self.last_step = self.current_step
            else:
                sleep_seconds_evaluate = 10
                time.sleep(sleep_seconds_evaluate)

    def get_step(self) -> int:
        optimize_status = OptimizeStatus.try_load(self.config.resource.optimize_status_file)
        return optimize_status.total_steps

    def evaluate_models(self):
        assert (self.config.evaluate.player2.player_type == "solver")

        # モデルの読込
        self.best_model = ModelUtil.load_model(self.config, self.config.evaluate.player1.use_newest)

        # ソルバーのレベルを変えて対戦させる
        random_rates = self.config.evaluate.evaluate_random_rates
        for random_rate in random_rates:
            self.evaluate_model(random_rate)

    def evaluate_model(self, random_rate: int = 0):
        rc = self.config.resource
        self.config.evaluate.player2.solver_player_random_ratio = random_rate / 100.0
        game_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        self.kifu_path = os.path.join(rc.eval_ggf_data_dir, rc.ggf_filename_tmpl % game_id)

        p1_wins = []
        p1_is_blacks = []
        winning_rate = 0
        game_num = self.config.evaluate.game_num
        for game_idx in range(game_num):
            player1 = create_player(self.config, self.config.evaluate.player1, model=self.best_model)
            player2 = create_player(self.config, self.config.evaluate.player2, model=None, solver=self.solver)
            p1_win, p1_is_black, score = self.play_game(player1, player2, game_idx)
            if p1_win is not None:
                p1_wins.append(p1_win)
                p1_is_blacks.append(p1_is_black)
                winning_rate = sum(p1_wins) / len(p1_wins)
            logger.debug(f"game {game_idx}: p1_win={p1_win} p1_is_black={p1_is_black} score={score} "
                         f"winning rate {winning_rate * 100:.1f}%")

        wins_black = sum([p1_win for p1_win, p1_is_black in zip(p1_wins, p1_is_blacks) if p1_is_black])
        wins_white = sum([p1_win for p1_win, p1_is_black in zip(p1_wins, p1_is_blacks) if not p1_is_black])
        games_black = sum([1 for p1_win, p1_is_black in zip(p1_wins, p1_is_blacks) if p1_is_black])
        games_white = sum([1 for p1_win, p1_is_black in zip(p1_wins, p1_is_blacks) if not p1_is_black])
        winning_rate = sum(p1_wins) / len(p1_wins)
        winning_rate_black = wins_black / games_black
        winning_rate_white = wins_white / games_white

        result = EvaluateResult(game_num, random_rate, winning_rate, winning_rate_black, winning_rate_white)
        self.write_log(result)

        logger.debug(
            f"winning rate {winning_rate * 100:.1f}% - black: {wins_black}/{games_black}, white: {wins_white}/{games_white}")

    def write_log(self, result: EvaluateResult):
        random_rate = result.random_rate
        log_info = {
            f"evaluate/win_rate": result.win_rate,
            f"evaluate/win_rate_white": result.win_rate_white,
            f"evaluate/win_rate_black": result.win_rate_black,
        }
        log_dir_suffix = f"random{random_rate}"
        self.tensor_board_logger.log_scaler(log_dir_suffix, log_info, self.current_step)

    def play_game(self, player1: BasePlayer, player2: BasePlayer, game_idx: int):
        env = C4Env.create_init()

        # 偶数番／奇数番で先手後手を定める
        p1_is_black = game_idx % 2 == 0
        if p1_is_black:
            black, white = player1, player2
        else:
            black, white = player2, player1
        actions = []

        while not env.done:
            env_black = env.get_env_black()
            if env.next_player == Player.black:
                action = black.action(env_black)
            else:
                action = white.action(env_black)
            actions.append(action)

            env.step(action)

        if env.winner == Winner.black:
            if p1_is_black:
                p1_win = 1
            else:
                p1_win = 0
        elif env.winner == Winner.white:
            if p1_is_black:
                p1_win = 0
            else:
                p1_win = 1
        else:
            p1_win = 0.5

        SimpleKifu.save_kifu(self.kifu_path, "p1" if p1_is_black else "p2", "p2" if p1_is_black else "p1",
                             actions, env.winner)

        score = 0
        return p1_win, p1_is_black, score


class SimpleKifu:

    @classmethod
    def save_kifu(cls, kifu_path, player_black, player_white, actions, winner):
        st = cls.to_kifu(player_black, player_white, actions, winner)
        with open(kifu_path, "at") as f:
            f.write(st + "\n")

    @classmethod
    def to_kifu(cls, player_black, player_white, actions, winner):
        move_list = []
        for i, action in enumerate(actions):
            y, x = action // BW, action % BW
            move_list.append(str(x))
        move_string = "".join(move_list)
        result_string = 'D'
        if winner == Winner.black:
            result_string = 'B'
        elif winner == Winner.white:
            result_string = 'W'

        return f"[{player_black}:{player_white}]:{result_string}:{move_string}"
