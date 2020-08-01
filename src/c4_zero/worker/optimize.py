import json
import os
from collections import Counter
from dataclasses import dataclass
from logging import getLogger
from time import sleep, time
from typing import Any, Dict, List, Optional, Set

import keras.backend as K
import tensorflow as tf
from keras.callbacks import Callback
from keras.optimizers import SGD

# from c4_zero.agent.evaluator.problem_evaluator import ProblemEvaluator
from c4_zero.agent.model.model import C4Model
from c4_zero.agent.model.model import objective_function_for_policy, \
    objective_function_for_value
from c4_zero.agent.model.model_util import ModelUtil
from c4_zero.config import Config
from c4_zero.env.c4_environment import C4Env
from c4_zero.env.data import MoveData, TrainingData
from c4_zero.env.data_helper import get_game_data_filenames
from c4_zero.lib import tensorflow_util
from c4_zero.lib.tensorboard_logger import TensorBoardLogger

logger = getLogger(__name__)


def start(config: Config):
    """モデルの学習を行うWorkerを起動する"""

    tensorflow_util.set_session_config(per_process_gpu_memory_fraction=0.65)
    return OptimizeWorker(config).start()


@dataclass
class OptimizeStatus:
    total_steps: int

    @classmethod
    def _initial_status(cls) -> 'OptimizeStatus':
        return OptimizeStatus(total_steps=0)

    @classmethod
    def try_load(cls, path: str) -> 'OptimizeStatus':
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            status = OptimizeStatus(total_steps=data["total_steps"])
            return status
        else:
            return cls._initial_status()

    def save(self, path: str):
        status = {"total_steps": self.total_steps}
        with open(path, "w") as f:
            json.dump(status, f)


class OptimizeWorker:
    """モデルの学習を行うWorker"""
    config: Config
    status: OptimizeStatus
    model: C4Model
    loaded_filenames: Set[str]
    loaded_data: Dict[str, TrainingData]
    training_count_of_files: Counter
    dataset: Optional[TrainingData]
    optimizer: Any

    def __init__(self, config: Config):
        self.config = config
        self.loaded_filenames = set()
        self.loaded_data = {}
        self.dataset = None
        self.training_count_of_files = Counter()
        self.tensor_board = TensorBoardLogger(self.config.resource.optimize_log_dir)
        self.save_model_count = 0

    def start(self):
        # envの初期化
        C4Env.initialize(self.config.env)
        self.status = OptimizeStatus.try_load(self.config.resource.optimize_status_file)
        self.model = self._load_model()
        self.training()

    def training(self):
        """モデルの学習を行う

        * プロセスを立ち上げっぱなしの状態にしておく
        * 1バッチの学習=1ステップ
        * 一定時間ごとに学習データをロードし、一定サイズ以上であれば学習する
        * 一定ステップごとにモデルの保存、テストケースによる評価、TensorBoardへのロギングを行う
        """
        self._compile_model()

        save_model_callback = PerStepCallback(self.config.optimize.save_model_steps, self._save_current_model,
                                              self.config.optimize.wait_after_save_model_ratio,
                                              initial_step=self.status.total_steps)
        save_evaluation_callback = PerStepCallback(self.config.optimize.save_model_steps, self._save_problem_evaluation,
                                                   self.config.optimize.wait_after_save_model_ratio,
                                                   initial_step=self.status.total_steps)
        callbacks = [save_model_callback, save_evaluation_callback]  # type: List[Callback]
        tb_callback = None  # type: TensorBoardStepCallback

        if self.config.optimize.use_tensorboard:
            tb_callback = TensorBoardStepCallback(
                log_dir=self.config.resource.tensorboard_log_dir,
                logging_per_steps=self.config.optimize.logging_per_steps,
                step=self.status.total_steps,
            )
            callbacks.append(tb_callback)

        while True:
            self._load_play_data()
            if self._dataset_size < self.config.optimize.min_data_size_to_learn:
                logger.info(
                    f"dataset_size={self._dataset_size} is less than {self.config.optimize.min_data_size_to_learn}")
                sleep(10)
                continue
            self._update_learning_rate(self.status.total_steps)
            steps = self.train_epoch(self.config.optimize.epoch_to_checkpoint, callbacks)
            self.status.total_steps += steps
            self._count_up_training_count_and_delete_self_play_data_files()

            # ステータスを保存する
            self.status.save(self.config.resource.optimize_status_file)
            if self.save_model_count >= self.config.optimize.max_save_model_count:
                break

        if tb_callback:  # This code is never reached. But potentially this is required.
            tb_callback.close()

    def train_epoch(self, epochs, callbacks):
        tc = self.config.optimize
        self.model.fit(self.dataset.s_ary, [self.dataset.p_ary, self.dataset.z_ary],
                       batch_size=tc.batch_size,
                       callbacks=callbacks,
                       epochs=epochs)
        # ステップは、学習を行ったバッチの数
        steps = (len(self.dataset) // tc.batch_size) * epochs
        return steps

    def _compile_model(self):
        self.optimizer = SGD(lr=1e-2, momentum=0.9)
        losses = [objective_function_for_policy, objective_function_for_value]
        self.model.compile(optimizer=self.optimizer, loss=losses)

    def _update_learning_rate(self, total_steps):
        # The deepmind paper says
        # ~400k: 1e-2
        # 400k~600k: 1e-3
        # 600k~: 1e-4

        lr = self._decide_learning_rate(total_steps)
        if lr:
            K.set_value(self.optimizer.lr, lr)
            logger.debug(f"total step={total_steps}, set learning rate to {lr}")

    def _decide_learning_rate(self, total_steps):
        ret = None

        if os.path.exists(self.config.resource.force_learing_rate_file):
            try:
                with open(self.config.resource.force_learing_rate_file, "rt") as f:
                    ret = float(str(f.read()).strip())
                    if ret:
                        logger.debug(f"loaded lr from force learning rate file: {ret}")
                        return ret
            except ValueError:
                pass

        for step, lr in self.config.optimize.lr_schedules:
            if total_steps >= step:
                ret = lr
        return ret

    def _save_current_model(self, step):
        ModelUtil.save_model(self.model, save_newest=True)
        self.save_model_count += 1

    def _save_problem_evaluation(self, step):
        # テストケースによる評価を削除
        pass

        """
        evaluate_df = ProblemEvaluator.evaluate(self.model, debug=True)

        evaluate_df_base = evaluate_df[evaluate_df["problem"] != "Test_L1_R51"]
        evaluate_df_r1 = evaluate_df[evaluate_df["problem"] == "Test_L1_R1"]
        evaluate_df_r2 = evaluate_df[evaluate_df["problem"] == "Test_L1_R2"]
        evaluate_df_r3 = evaluate_df[evaluate_df["problem"] == "Test_L1_R3"]
        evaluate_df_r51 = evaluate_df[evaluate_df["problem"] == "Test_L1_R51"]

        log_dir_suffix, log_info = None, {f"optimize/v_rmse": evaluate_df_base["rmse"].mean(),
                                          f"optimize/v_acc_wo_draw": evaluate_df_base["acc_wo_draw"].mean()}
        self.tensor_board.log_scaler(log_dir_suffix, log_info, step)
        log_dir_suffix, log_info = "all", {f"optimize/p_acc": evaluate_df_base["move_score_acc"].mean()}
        self.tensor_board.log_scaler(log_dir_suffix, log_info, step)
        log_dir_suffix, log_info = "L1_R1", {f"optimize/p_acc": evaluate_df_r1["move_score_acc"].mean()}
        self.tensor_board.log_scaler(log_dir_suffix, log_info, step)
        log_dir_suffix, log_info = "L1_R2", {f"optimize/p_acc": evaluate_df_r2["move_score_acc"].mean()}
        self.tensor_board.log_scaler(log_dir_suffix, log_info, step)
        log_dir_suffix, log_info = "L1_R3", {f"optimize/p_acc": evaluate_df_r3["move_score_acc"].mean()}
        self.tensor_board.log_scaler(log_dir_suffix, log_info, step)
        log_dir_suffix, log_info = "L1_R51", {f"optimize/p_acc": evaluate_df_r51["move_score_acc"].mean()}
        self.tensor_board.log_scaler(log_dir_suffix, log_info, step)
        """

    def _collect_all_loaded_data(self) -> Optional[TrainingData]:
        return TrainingData.concatenate(list(self.loaded_data.values()))

    @property
    def _dataset_size(self):
        if self.dataset is None:
            return 0
        return len(self.dataset)

    def _load_model(self):
        """モデルの読込を行う"""
        model = ModelUtil.load_model(self.config, use_newest=True)
        return model

    def _load_play_data(self) -> None:
        """対象フォルダから学習データを読み込む

        * 既に読み込まれている学習データについてはスキップする
        * 既に読み込まれている学習データがフォルダから無くなっていた場合、削除する
        """
        filenames = get_game_data_filenames(self.config.resource)
        updated = False
        for filename in filenames:
            if filename in self.loaded_filenames:
                continue
            self.load_data_from_file(filename)
            updated = True

        for filename in (self.loaded_filenames - set(filenames)):
            self._unload_data_of_file(filename)
            updated = True

        if updated:
            logger.debug("updating training dataset")
            self.dataset = self._collect_all_loaded_data()

    def load_data_from_file(self, filename) -> None:
        try:
            logger.debug(f"loading data from {filename}")
            moves_data = MoveData.read_from_file(filename)
            self.loaded_data[filename] = MoveData.convert_to_training_data(moves_data)
            self.loaded_filenames.add(filename)
        except Exception as e:
            logger.warning(f"failed load_data_from_file: {str(e)}")

    def _unload_data_of_file(self, filename) -> None:
        logger.debug(f"removing data about {filename} from training set")
        self.loaded_filenames.remove(filename)
        if filename in self.loaded_data:
            del self.loaded_data[filename]
        if filename in self.training_count_of_files:
            del self.training_count_of_files[filename]

    def _count_up_training_count_and_delete_self_play_data_files(self) -> None:
        limit = self.config.optimize.delete_self_play_after_number_of_training
        if not limit:
            return

        for filename in self.loaded_filenames:
            self.training_count_of_files[filename] += 1
            if self.training_count_of_files[filename] >= limit:
                if os.path.exists(filename):
                    try:
                        logger.debug(f"remove {filename}")
                        os.remove(filename)
                    except Exception as e:
                        logger.warning(e)


class PerStepCallback(Callback):
    def __init__(self, per_step, callback, wait_after_save_model_ratio=None, initial_step=0):
        super().__init__()
        self.per_step = per_step
        self.step = initial_step
        self.callback = callback
        self.wait_after_save_model_ratio = wait_after_save_model_ratio
        self.last_wait_time = time()

    def on_batch_end(self, batch, logs=None):
        self.step += 1
        if self.step % self.per_step == 0:
            self.callback(step=self.step)
            self.wait()

    def wait(self):
        if self.wait_after_save_model_ratio:
            time_spent = time() - self.last_wait_time
            logger.debug(f"start sleeping {time_spent} seconds")
            sleep(time_spent * self.wait_after_save_model_ratio)
            logger.debug(f"finish sleeping")
            self.last_wait_time = time()


class TensorBoardStepCallback(Callback):
    """keras用のコールバック

    バッチ終了時にtensorflowにログを出力する
    """

    def __init__(self, log_dir, logging_per_steps=100, step=0):
        super().__init__()
        self.step = step
        self.logging_per_steps = logging_per_steps
        self.writer = tf.summary.FileWriter(log_dir)

    def on_batch_end(self, batch, logs=None):
        self.step += 1

        if self.step % self.logging_per_steps > 0:
            return

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, self.step)
        self.writer.flush()

    def close(self):
        self.writer.close()
