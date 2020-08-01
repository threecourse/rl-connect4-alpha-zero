import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from logging import getLogger
from multiprocessing import Manager
from random import random
from time import time
from traceback import print_stack
from typing import List, Optional

import numpy as np

from c4_zero.agent.api import InferenceAPIServer, InferenceAPIBase
from c4_zero.agent.player import BasePlayer
from c4_zero.agent.player_alphazero import AlphaZeroPlayer
from c4_zero.agent.player_creator import create_player
from c4_zero.config import Config, PlayerConfig
from c4_zero.env.c4_environment import C4Env
from c4_zero.env.c4_util import Winner, Player
from c4_zero.env.data import MoveGGFHistory, MoveData
from c4_zero.env.data_helper import get_game_data_filenames
from c4_zero.env.environment import Env
from c4_zero.lib import tensorflow_util
from c4_zero.lib.tensorboard_logger import TensorBoardLogger

logger = getLogger(__name__)


def start(config: Config):
    """自己対戦を開始する

    * 推論用APIサーバを起動する
    * プロセスプールを用いた非同期実行により、複数個のSelfPlayWorkerを起動する
    """
    tensorflow_util.set_session_config(per_process_gpu_memory_fraction=0.3)
    api_server = InferenceAPIServer(config, config.self_play.player)
    api_server.start_serve()

    process_num = config.self_play.multi_process_num

    with Manager() as manager:
        self_play_status = SelfPlayStatus.try_load(config.resource.self_play_status_file)
        shared_var = SharedVar(manager, game_idx=self_play_status.game_idx)
        if config.self_play.use_self_play_worker:
            with ProcessPoolExecutor(max_workers=process_num) as executor:
                futures = []
                for i in range(process_num):
                    play_worker = SelfPlayWorker(config, env=C4Env.create_init(), api=api_server.get_api_client(),
                                                 shared_var=shared_var, worker_index=i)
                    futures.append(executor.submit(play_worker.start))
        else:
            play_worker = SelfPlayWorker(config, env=C4Env.create_init(), api=api_server.get_api_client(),
                                         shared_var=shared_var, worker_index=0)
            play_worker.start()


class SharedVar:
    """multiprocessの共有変数"""

    def __init__(self, manager, game_idx: int):
        """

        :param Manager manager:
        :param int game_idx:
        """
        self._lock = manager.Lock()
        self._game_idx = manager.Value('i', game_idx)  # type: multiprocessing.managers.ValueProxy

    @property
    def game_idx(self):
        return self._game_idx.value

    def incr_game_idx(self, n=1):
        with self._lock:
            self._game_idx.value += n
            return self._game_idx.value

from dataclasses import dataclass
import json

@dataclass
class SelfPlayStatus:
    game_idx: int

    @classmethod
    def _initial_status(cls) -> 'SelfPlayStatus':
        return SelfPlayStatus(game_idx=0)

    @classmethod
    def try_load(cls, path: str) -> 'SelfPlayStatus':
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            status = SelfPlayStatus(game_idx=data["game_idx"])
            return status
        else:
            return cls._initial_status()

    def save(self, path: str):
        status = {"game_idx": self.game_idx}
        with open(path, "w") as f:
            json.dump(status, f)

class SelfPlayWorker:
    """自己対戦を行うWorker

    Attributes:
        config: コンフィグ
        player_config: コンフィグ
        env: 環境
        api: 推論用API
        shared_var: 共有変数
        worker_index: workerのindex
        black: 先手番プレイヤー
        white: 後手番プレイヤー
        moves_data: 学習用の手の履歴（プレイヤーからの視点のデータを集めたもの）
        move_ggf_history: 棋譜用の過去の手や評価の履歴（ゲームマネージャからの視点）
        tensor_board_logger: TensorBoardのロガー

        # 投了関係
        false_positive_count_of_resign: 投了を無しにしたゲームの数のうち、誤った投了を行ったもの
        resign_test_game_count: 投了を無しにしたゲームの数

        # resign_threshold（投了の閾値）は、直接player_configを入替えているので注意
    """

    config: Config
    player_config: PlayerConfig
    env: Env
    api: Optional[InferenceAPIBase]
    shared_var: SharedVar
    worker_index: int
    black: BasePlayer
    white: BasePlayer
    moves_data: List[MoveData]
    move_ggf_history: 'MoveGGFHistory'
    tensor_board_logger: TensorBoardLogger
    false_positive_count_of_resign: int
    resign_test_game_count: int

    def __init__(self, config: Config, env: Env, api: Optional[InferenceAPIBase],
                 shared_var: SharedVar, worker_index: int = 0):
        """コンストラクタ"""
        self.config = config
        self.player_config = config.self_play.player
        self.env = env
        self.api = api
        self.shared_var = shared_var
        self.moves_data = []
        self.false_positive_count_of_resign = 0
        self.resign_test_game_count = 0
        self.worker_index = worker_index

    @property
    def _false_positive_rate(self):
        if self.resign_test_game_count == 0:
            return 0
        return self.false_positive_count_of_resign / self.resign_test_game_count

    def start(self) -> None:
        """自己対戦を行う"""
        try:
            self._start()
        except Exception as e:
            print(repr(e))
            import traceback
            print(traceback.format_exc())
            print("----------")
            print_stack()

    def _start(self) -> None:
        """自己対戦を行う

        * mtcs_infoはWorker内でのみ共有している（configの現在の設定では毎ゲーム削除している）
        """
        # envの初期化
        C4Env.initialize(self.config.env)

        # 初期化
        logger.debug("SelfPlayWorker#start()")
        np.random.seed(None)
        worker_name = f"worker{self.worker_index:03d}"
        self.tensor_board_logger = TensorBoardLogger(os.path.join(self.config.resource.self_play_log_dir, worker_name))
        self.moves_data = []
        mtcs_info = None
        local_idx = 0

        while True:
            np.random.seed(None)
            local_idx += 1
            game_idx = self.shared_var.game_idx

            start_time = time()
            if mtcs_info is None and self.config.self_play.share_mtcs_info_in_self_play:
                mtcs_info = AlphaZeroPlayer.create_mtcs_info()

            # ゲームの開始
            env = self.start_game(local_idx, game_idx, mtcs_info)
            game_idx = self.shared_var.incr_game_idx()

            # ロギング
            end_time = time()
            time_spent = end_time - start_time
            logger.debug(f"play game {game_idx} time={time_spent} sec, "
                         f"turn={env.turn}:{env.winner}")

            # TensorBoardのロギング
            prefix = "self-play"
            log_info = {f"{prefix}/time": time_spent, f"{prefix}/turn": env.turn}
            if mtcs_info:
                log_info[f"{prefix}/mcts_buffer_size"] = len(mtcs_info.var_p)
            self.tensor_board_logger.log_scaler(None, log_info, game_idx)

            # 一定ゲームごとにMCTSの情報をリセットする
            if self.config.self_play.reset_mtcs_info_per_game and local_idx % self.config.self_play.reset_mtcs_info_per_game == 0:
                logger.debug("reset MCTS info")
                mtcs_info = None

            # game_idxを記録しておく
            SelfPlayStatus(game_idx).save(self.config.resource.self_play_status_file)

    def start_game(self, local_idx: int, last_game_idx: int, mtcs_info: 'MCTSInfo') -> Env:
        """
        ゲームを開始する

        :param local_idx: ??
        :param last_game_idx: 最後のゲームのindex
        :param mtcs_info: MCTS情報
        :return 終了時の環境

        * simulation_num_per_moveは最後のゲームのindexから動的に定める
        * 投了の可否は確率的に定める
        """

        # 初期設定
        self.env.reset()
        enable_resign = self.player_config.disable_resignation_rate <= random()
        self.player_config.simulation_num_per_move = self._decide_simulation_num_per_move(last_game_idx)
        logger.debug(f"simulation_num_per_move = {self.player_config.simulation_num_per_move}")
        self.black = self._create_player(enable_resign=enable_resign, mtcs_info=mtcs_info)
        self.white = self._create_player(enable_resign=enable_resign, mtcs_info=mtcs_info)
        if not enable_resign:
            logger.debug("Resignation is disabled in the next game.")
        self.move_ggf_history = MoveGGFHistory()

        # ゲームループ
        while not self.env.done:
            env_black = self.env.get_env_black()
            if self.env.next_player == Player.black:
                action_with_evaluation = self.black.action_with_evaluation(env_black)
            else:
                action_with_evaluation = self.white.action_with_evaluation(env_black)
            self.move_ggf_history.add_move(self.env, action_with_evaluation)
            self.env.step(action_with_evaluation.action)

        # 終了処理
        self._finish_game(resign_enabled=enable_resign)

        # ゲームデータの処理
        self._save_play_data(write=local_idx % self.config.self_play.nb_game_in_file == 0)
        self._remove_play_data()

        # 棋譜の出力
        if self.config.self_play.enable_ggf_data:
            is_write = local_idx <= 5
            self._save_ggf_data(write=is_write)

        return self.env

    def _create_player(self, enable_resign=None, mtcs_info=None):
        """プレイヤーの作成"""
        return create_player(self.config, self.player_config, None,
                             enable_resign=enable_resign, mtcs_info=mtcs_info, api=self.api)

    def _save_play_data(self, write: bool = True) -> None:
        """ゲームデータの保存

        :param write: 書き込むかどうか

        * ゲームデータをバッファに保存する。また、一定間隔ごとにファイルとして保存する
        """

        # ゲームデータのバッファへの保存
        # 一定の確率で引分のゲームは除外する
        if self.black.moves_data[0].is_not_draw or self.config.self_play.drop_draw_game_rate <= np.random.random():
            data = self.black.moves_data + self.white.moves_data
            self.moves_data += data

        # 書き込まない場合は終了
        if not write or not self.moves_data:
            return

        # ゲームデータのファイルへの保存
        rc = self.config.resource
        game_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        path = os.path.join(rc.play_data_dir, rc.play_data_filename_tmpl % game_id)
        logger.info(f"save play data to {path}")
        MoveData.write_to_file(path, self.moves_data)
        self.moves_data = []

    def _save_ggf_data(self, write: bool):
        """棋譜の保存"""
        if not write or not self.move_ggf_history:
            return
        rc = self.config.resource
        game_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        path = os.path.join(rc.self_play_ggf_data_dir, rc.ggf_filename_tmpl % game_id)
        with open(path, "wt") as f:
            f.write(self.move_ggf_history.to_ggf_string("RAZ", "RAZ") + "\n")

    def _remove_play_data(self):
        """フォルダ内のゲームデータが多すぎる場合に削除する"""
        files = get_game_data_filenames(self.config.resource)
        if len(files) < self.config.self_play.max_file_num:
            return
        try:
            for i in range(len(files) - self.config.self_play.max_file_num):
                os.remove(files[i])
        except:
            pass

    def _finish_game(self, resign_enabled=True):
        """ゲームの終了処理"""

        if self.env.winner == Winner.black:
            black_win = 1
            false_positive_of_resign = self.black.resigned
        elif self.env.winner == Winner.white:
            black_win = -1
            false_positive_of_resign = self.white.resigned
        else:
            black_win = 0
            false_positive_of_resign = self.black.resigned or self.white.resigned

        self.black.finish_game(black_win)
        self.white.finish_game(-black_win)

        # 投了関係の処理
        if not resign_enabled:
            self.resign_test_game_count += 1
            if false_positive_of_resign:
                self.false_positive_count_of_resign += 1
                logger.debug("false positive of resignation happened")
            self._check_and_update_resignation_threshold()

    def _check_and_update_resignation_threshold(self):
        """投了関連の変数のアップデート"""

        # 評価に十分なサンプルがない場合はスキップ
        if self.resign_test_game_count < 100 or self.player_config.resign_threshold is None:
            return

        # 誤投了率に応じて、投了する評価の閾値を動かす
        old_threshold = self.player_config.resign_threshold
        if self._false_positive_rate >= self.player_config.false_positive_threshold:
            self.player_config.resign_threshold -= self.player_config.resign_threshold_delta
        else:
            self.player_config.resign_threshold += self.player_config.resign_threshold_delta
        logger.debug(f"update resign_threshold: {old_threshold} -> {self.player_config.resign_threshold}")

        # 計測値をリセット
        self.false_positive_count_of_resign = 0
        self.resign_test_game_count = 0

    def _decide_simulation_num_per_move(self, idx) -> int:
        """手の探索におけるシミュレーション回数を決定する"""
        ret = -1  # dummy
        for min_idx, num in self.player_config.schedule_of_simulation_num_per_move:
            if idx >= min_idx:
                ret = num
        return ret
