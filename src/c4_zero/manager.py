import argparse
from logging import getLogger

import yaml
from moke_config import create_config

from .config import Config
from .lib.logger import setup_logger
import time

logger = getLogger(__name__)

CMD_LIST = ['self', 'opt', 'eval', 'play_gui', 'opt-workaround']


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", help="what to do", choices=CMD_LIST)
    parser.add_argument("-c", help="specify config yaml", dest="config_file")
    return parser


def setup(config: Config, args):
    config.resource.create_directories()
    setup_logger(config.resource.main_log_path)


def start():
    parser = create_parser()
    args = parser.parse_args()

    if args.config_file:
        with open(args.config_file, "rt") as f:
            config = create_config(Config, yaml.load(f))
    else:
        config = create_config(Config)
    setup(config, args)

    if args.cmd != "nboard":
        logger.info(f"config type: {config.type}")

    if args.cmd == "self":
        from .worker import self_play
        return self_play.start(config)
    elif args.cmd == 'opt':
        from .worker import optimize
        return optimize.start(config)
    elif args.cmd == 'eval':
        from .worker import evaluate
        return evaluate.start(config)
    elif args.cmd == 'opt-workaround':
        import subprocess
        while True:
            cmd = f"python src/c4_zero/run.py opt -c {args.config_file}"
            logger.info(f"start cmd: {cmd}")
            subprocess.check_call(cmd, shell=True)
            sleep_seconds = 5
            time.sleep(sleep_seconds)
    elif args.cmd == 'play_gui':
        from .worker import gui_app
        return gui_app.start(config)
