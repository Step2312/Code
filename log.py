import logging


class Logger():
    def __int__(self, level):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level)
