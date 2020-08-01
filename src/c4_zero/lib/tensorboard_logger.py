import os
from typing import Optional

import tensorflow as tf


class TensorBoardLogger:
    """TensorBoardのLogger"""

    def __init__(self, log_dir_base):
        self.log_dir_base = log_dir_base
        self.writers = {}

    def log_scaler(self, log_dir_suffix: Optional[str], info: dict, step: int) -> None:
        if log_dir_suffix is None:
            log_dir = self.log_dir_base
        else:
            log_dir = os.path.join(self.log_dir_base, log_dir_suffix)

        if log_dir not in self.writers:
            writer = tf.summary.FileWriter(log_dir)
            self.writers[log_dir_suffix] = writer
        writer = self.writers[log_dir_suffix]

        for tag, value in info.items():
            summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
            writer.add_summary(summary, step)
        writer.flush()
