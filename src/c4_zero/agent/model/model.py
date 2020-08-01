import hashlib
import json
import os
from logging import getLogger
from typing import Tuple

import keras.backend as K
# noinspection PyPep8Naming
import numpy as np
from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.losses import mean_squared_error
from keras.regularizers import l2

from c4_zero.config import Config
from c4_zero.env.c4_util import BH, BW

logger = getLogger(__name__)


class C4Model:
    """ニューラルネットモデル

    kerasのモデルをラップしたもの
    """
    config: Config
    digest: str
    _model: Model

    def __init__(self, config: Config):
        self.config = config
        self.digest = None

    def compile(self, **kwargs):
        return self._model.compile(**kwargs)

    def fit(self, x, y, **kwargs) -> None:
        assert x.ndim == 4  # 4のみを受け付けるようにする
        L = x.shape[0]
        assert x.shape == (L, 2, BH, BW)
        policy_ary, value_ary = y
        assert policy_ary.shape == (L, BH * BW)
        assert value_ary.shape == (L,)
        self._model.fit(x, y, **kwargs)

    def predict(self, x: np.array, **kwargs) -> Tuple[np.array, np.array]:
        assert x.ndim == 4  # 4のみを受け付けるようにする
        L = x.shape[0]
        assert x.shape == (L, 2, BH, BW)
        policy_ary, value_ary = self._model.predict(x, **kwargs)
        assert policy_ary.shape == (L, BH * BW)
        assert value_ary.shape == (L, 1)
        return policy_ary, value_ary

    def predict_on_batch(self, x: np.array) -> Tuple[np.array, np.array]:
        assert x.ndim == 4  # 4のみを受け付けるようにする
        L = x.shape[0]
        assert x.shape == (L, 2, BH, BW)
        policy_ary, value_ary = self._model.predict_on_batch(x)
        assert policy_ary.shape == (L, BH * BW)
        assert value_ary.shape == (L, 1)
        return policy_ary, value_ary

    def make_predict_function(self):
        """threading workaroundに必要"""
        return self._model._make_predict_function()

    def build(self):
        """モデルの構築を行う

        最初に作成する場合のみ必要
        """

        mc = self.config.model
        in_x = x = Input((2, BH, BW))  # [own(8x8), enemy(8x8)]

        # (batch, channels, height, width)
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size, padding="same",
                   data_format="channels_first", kernel_regularizer=l2(mc.l2_reg))(x)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)

        for _ in range(mc.res_layer_num):
            x = self._build_residual_block(x)

        res_out = x
        # for policy output
        x = Conv2D(filters=2, kernel_size=1, data_format="channels_first", kernel_regularizer=l2(mc.l2_reg))(res_out)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)
        x = Flatten()(x)
        # no output for 'pass'
        policy_out = Dense(BH * BW, kernel_regularizer=l2(mc.l2_reg), activation="softmax", name="policy_out")(x)

        # for value output
        x = Conv2D(filters=1, kernel_size=1, data_format="channels_first", kernel_regularizer=l2(mc.l2_reg))(res_out)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)
        x = Flatten()(x)
        x = Dense(mc.value_fc_size, kernel_regularizer=l2(mc.l2_reg), activation="relu")(x)
        value_out = Dense(1, kernel_regularizer=l2(mc.l2_reg), activation="tanh", name="value_out")(x)

        self._model = Model(in_x, [policy_out, value_out], name="c4_model")

    @staticmethod
    def fetch_digest(weight_path) -> str:
        """対象パスのモデルのdigestを計算する"""
        if os.path.exists(weight_path):
            m = hashlib.sha256()
            with open(weight_path, "rb") as f:
                m.update(f.read())
            return m.hexdigest()

    def load(self, config_path, weight_path) -> bool:
        """対象パスのモデルの読込を行う"""
        if os.path.exists(config_path) and os.path.exists(weight_path):
            logger.debug(f"loading model from {config_path}")
            with open(config_path, "rt") as f:
                self._model = Model.from_config(json.load(f))
            self._model.load_weights(weight_path)
            self.digest = self.fetch_digest(weight_path)
            logger.debug(f"loaded model digest = {self.digest}")
            return True
        else:
            logger.debug(f"model files does not exist at {config_path} and {weight_path}")
            return False

    def save(self, config_path, weight_path):
        """対象パスにモデルを保存する"""
        logger.debug(f"save model to {config_path}")
        with open(config_path, "wt") as f:
            json.dump(self._model.get_config(), f)
            self._model.save_weights(weight_path)
        self.digest = self.fetch_digest(weight_path)
        logger.debug(f"saved model digest {self.digest}")

    def _build_residual_block(self, x):
        mc = self.config.model
        in_x = x
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size, padding="same",
                   data_format="channels_first", kernel_regularizer=l2(mc.l2_reg))(x)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size, padding="same",
                   data_format="channels_first", kernel_regularizer=l2(mc.l2_reg))(x)
        x = BatchNormalization(axis=1)(x)
        x = Add()([in_x, x])
        x = Activation("relu")(x)
        return x


def objective_function_for_policy(y_true, y_pred):
    # can use categorical_crossentropy??
    return K.sum(-y_true * K.log(y_pred + K.epsilon()), axis=-1)


def objective_function_for_value(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)
