# idea from http://eyalarubas.com/python-subproc-nonblock.html
from logging import getLogger
from queue import Queue, Empty
from threading import Thread

logger = getLogger(__name__)


class NonBlockingStreamReader:
    """NonBlockingな他プロセスへの標準入出力の接続

    現在は使用していない
    """

    def __init__(self, stream):
        self._stream = stream
        self._queue = Queue()
        self._thread = None
        self.closed = True

    def start(self, push_callback=None):
        def _worker():
            while True:
                line = self._stream.readline()
                if line:
                    if push_callback:
                        push_callback(line)
                    self._queue.put(line)
                else:
                    logger.debug("the stream may be closed")
                    break
            self.closed = True

        self._thread = Thread(target=_worker)
        self._thread.setDaemon(True)
        self._thread.setName("NonBlockingStreamReader of %s" % repr(self._stream))
        self.closed = False
        self._thread.start()

    def readline(self, timeout=None):
        try:
            return self._queue.get(block=timeout is not None, timeout=timeout)
        except Empty:
            return None
