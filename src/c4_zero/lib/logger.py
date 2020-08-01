from logging import StreamHandler, basicConfig, DEBUG, getLogger, Formatter


def setup_logger(log_filename):
    """ログの設定を行う"""
    format_str = '%(asctime)s@%(name)s %(levelname)s # %(message)s'
    basicConfig(filename=log_filename, level=DEBUG, format=format_str)
    stream_handler = StreamHandler()
    stream_handler.setFormatter(Formatter(format_str))
    getLogger().addHandler(stream_handler)
