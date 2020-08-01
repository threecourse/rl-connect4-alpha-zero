import os
from datetime import datetime
from logging import getLogger
from time import sleep
from typing import Optional

import keras.backend as K

from c4_zero.agent.model.model import C4Model
from c4_zero.config import Config
from c4_zero.env.data_helper import get_next_generation_model_dirs

logger = getLogger(__name__)


class ModelUtil:

    @classmethod
    def initial_model(cls, config: Config) -> C4Model:
        model = C4Model(config)
        model.build()
        return model

    @classmethod
    def load_model(cls, config: Config, use_newest: bool) -> C4Model:
        """モデルを読み込む"""
        model = cls.try_load_model(config, use_newest)
        if model is None:
            raise RuntimeError("No models found!")
        return model

    @classmethod
    def try_load_model(cls, config: Config, use_newest: bool) -> Optional[C4Model]:
        """モデルを読み込む（読み込めなかった場合はNone）"""

        # 最新もしくはベストのモデルを読み込む
        if use_newest:
            model = C4Model(config)
            loaded = cls._load_newest_model(model)
            return model if loaded else None
        else:
            model = C4Model(config)
            loaded = cls._load_best_model_weight(model)
            return model if loaded else None

    @classmethod
    def reload_model_if_changed(cls, model: C4Model, use_newest: bool) -> None:
        """モデルを再度読み込む"""
        try:
            if use_newest:
                cls._reload_newest_model_if_changed(model, clear_session=True)
            else:
                cls._reload_best_model_if_changed(model, clear_session=True)
        except Exception as e:
            logger.error(e)

    @classmethod
    def save_model(cls, model: C4Model, save_newest: bool) -> None:
        if save_newest:
            cls._save_newest_model(model)
        else:
            cls._save_as_best_model(model)

    @classmethod
    def _save_as_best_model(cls, model: C4Model) -> None:
        """最も良いモデルのウェイトを保存する"""
        model.save(model.config.resource.model_best_config_path, model.config.resource.model_best_weight_path)

    @classmethod
    def _save_newest_model(cls, model: C4Model) -> None:
        """最新のモデルのウェイトを保存する"""
        rc = model.config.resource
        model_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        model_dir = os.path.join(rc.next_generation_model_dir, rc.next_generation_model_dirname_tmpl % model_id)
        os.makedirs(model_dir, exist_ok=True)
        config_path = os.path.join(model_dir, rc.next_generation_model_config_filename)
        weight_path = os.path.join(model_dir, rc.next_generation_model_weight_filename)
        model.save(config_path, weight_path)

    @classmethod
    def _load_best_model_weight(cls, model: C4Model, clear_session=True) -> bool:
        """最も良いモデルのウェイトを読み込む"""
        if clear_session:
            K.clear_session()
        return model.load(model.config.resource.model_best_config_path, model.config.resource.model_best_weight_path)

    @classmethod
    def _reload_best_model_if_changed(cls, model: C4Model, clear_session=True) -> bool:
        """もしモデルのウェイトが変更になっていたら再読込する"""
        logger.debug(f"start reload the best model if changed")
        digest = model.fetch_digest(model.config.resource.model_best_weight_path)
        if digest != model.digest:
            return cls._load_best_model_weight(model, clear_session=clear_session)
        else:
            logger.debug(f"the best model is not changed")
            return False

    @classmethod
    def _load_newest_model(cls, model: C4Model, clear_session=True) -> bool:
        """最も良いモデルのウェイトを読み込む"""
        rc = model.config.resource
        dirs = get_next_generation_model_dirs(rc)
        if not dirs:
            logger.debug("No next generation model exists.")
            return False
        model_dir = dirs[-1]
        config_path = os.path.join(model_dir, rc.next_generation_model_config_filename)
        weight_path = os.path.join(model_dir, rc.next_generation_model_weight_filename)
        logger.debug(f"Loading weight from {model_dir}")
        if clear_session:
            K.clear_session()
        for _ in range(5):
            try:
                return model.load(config_path, weight_path)
            except Exception as e:
                logger.warning(f"error in load model: #{e}")
                sleep(3)
        # could not load model
        raise RuntimeError("Cannot Load Model!")

    @classmethod
    def _reload_newest_model_if_changed(cls, model: C4Model, clear_session=True) -> bool:
        """最新のモデルのウェイトが変更になっていたら再読込する"""

        rc = model.config.resource
        dirs = get_next_generation_model_dirs(rc)
        if not dirs:
            logger.debug("No next generation model exists.")
            return False
        model_dir = dirs[-1]
        config_path = os.path.join(model_dir, rc.next_generation_model_config_filename)
        weight_path = os.path.join(model_dir, rc.next_generation_model_weight_filename)
        digest = model.fetch_digest(weight_path)
        if digest and digest != model.digest:
            return cls._load_newest_model(model, clear_session)
        else:
            logger.debug(f"The newest model is not changed: digest={digest}")
            return False
