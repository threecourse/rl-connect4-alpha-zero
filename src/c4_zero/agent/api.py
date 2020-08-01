from logging import getLogger
from multiprocessing import Pipe, connection
from threading import Thread
from time import time
from typing import List, NoReturn

import numpy as np
import tensorflow as tf

from c4_zero.agent.model.model import C4Model
from c4_zero.agent.model.model_util import ModelUtil
from c4_zero.config import Config, PlayerConfig
from c4_zero.env.c4_util import BH, BW
from c4_zero.env.data import InferenceData, InferenceResult

logger = getLogger(__name__)


class InferenceAPIBase:
    """推論を行うAPIのベースクラス"""

    def predict(self, x: np.array) -> InferenceResult:
        return self._do_predict(x)

    def predict_inference_data(self, data: InferenceData) -> InferenceResult:
        x = data.s_ary
        return self.predict(x)

    def predict_inference_data_list(self, data_list: List[InferenceData]) -> InferenceResult:
        x = InferenceData.to_x(data_list)
        return self.predict(x)

    def _do_predict(self, x: np.array) -> InferenceResult:
        raise NotImplementedError


class InferenceSimpleAPI(InferenceAPIBase):
    """推論を行うAPI"""

    def __init__(self, config: Config, model: 'C4Model') -> None:
        self.config = config
        self.model = model

    def _do_predict(self, x: np.array) -> InferenceResult:
        p_ary, v_ary = self.model.predict_on_batch(x)
        return InferenceResult(p_ary, v_ary)


class InferenceModelAPIClient(InferenceAPIBase):
    """推論を行うAPI（サーバに推論の依頼を行う）"""

    def __init__(self, config: Config, conn: connection.Connection):
        self.config = config
        self.connection = conn

    def _do_predict(self, x: np.array) -> InferenceResult:
        self.connection.send(x)
        p_ary, v_ary = self.connection.recv()
        return InferenceResult(p_ary, v_ary)


class DummyAPI(InferenceAPIBase):
    """ダミー値を返すAPIサーバ"""

    def _do_predict(self, x: np.array) -> InferenceResult:
        L = x.shape[0]
        p_ary, v_ary = np.ones((L, BH * BW)), np.zeros((L))
        return InferenceResult(p_ary, v_ary)


class InferenceAPIServer:
    """推論を行うAPIのサーバ

    * サーバにデータが送付されていれば、それらをまとめて推論する
    * 一定間隔でモデルの再読込を行う

    （参考）https://github.com/Akababa/Chess-Zero/blob/nohistory/src/chess_zero/agent/api_chess.py
    """

    def __init__(self, config: Config, player_config: PlayerConfig):
        self.config = config
        self.player_config = player_config
        self.model = None  # type: C4Model
        self.connections = []  # type: List[connection.Connection]

    def get_api_client(self) -> InferenceModelAPIClient:
        """クライアントを作成する"""
        me, you = Pipe()
        self.connections.append(me)
        return InferenceModelAPIClient(self.config, you)

    def start_serve(self) -> None:
        """サーバを起動する"""
        save_newest = self.player_config.use_newest
        model = ModelUtil.try_load_model(self.config, save_newest)
        if model is None:
            self.model = ModelUtil.initial_model(self.config)
            ModelUtil.save_model(self.model, save_newest=save_newest)
        else:
            self.model = model

        # threading workaround: https://github.com/keras-team/keras/issues/5640
        self.model.make_predict_function()
        self.graph = tf.get_default_graph()

        prediction_worker = Thread(target=self._prediction_worker, name="prediction_worker")
        prediction_worker.daemon = True
        prediction_worker.start()

    def _prediction_worker(self) -> NoReturn:
        """サーバの実行内容"""
        logger.debug("prediction_worker started")
        average_prediction_size = []
        last_model_check_time = time()
        while True:
            # 前回のモデルの読み込みより一定時間が経過していた場合
            if last_model_check_time + 60 < time():
                # モデルの再読込を試行する
                ModelUtil.reload_model_if_changed(self.model, self.player_config.use_newest)
                last_model_check_time = time()
                logger.debug(f"average_prediction_size={np.average(average_prediction_size)}")
                average_prediction_size = []

            # サーバに送られたデータを確認する
            ready_conns = connection.wait(self.connections, timeout=0.001)  # type: List[connection.Connection]
            if not ready_conns:
                continue

            # サーバに送られたデータを結合する
            data = []
            size_list = []
            for conn in ready_conns:
                x = conn.recv()
                data.append(x)  # shape: (k, 2, 8, 8)
                size_list.append(x.shape[0])  # save k
            average_prediction_size.append(np.sum(size_list))
            array = np.concatenate(data, axis=0)

            # 推論を行う
            policy_ary, value_ary = self.model.predict_on_batch(array)
            idx = 0

            # サーバに送られた単位に戻して返す
            for conn, s in zip(ready_conns, size_list):
                conn.send((policy_ary[idx:idx + s], value_ary[idx:idx + s]))
                idx += s
