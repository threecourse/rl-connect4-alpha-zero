import os
from glob import glob
from logging import getLogger

from c4_zero.config import ResourceConfig

logger = getLogger(__name__)


def get_game_data_filenames(rc: ResourceConfig):
    """ResourceConfigで指定されたディレクトリにある、指定された条件を満たすファイル名のリストを取得する"""
    pattern = os.path.join(rc.play_data_dir, rc.play_data_filename_tmpl % "*")
    files = list(sorted(glob(pattern)))
    return files


def get_next_generation_model_dirs(rc: ResourceConfig):
    """ResourceConfigで指定されたディレクトリにある、指定された条件を満たすディレクトリ名のリストを取得する"""
    dir_pattern = os.path.join(rc.next_generation_model_dir, rc.next_generation_model_dirname_tmpl % "*")
    dirs = list(sorted(glob(dir_pattern)))
    return dirs
