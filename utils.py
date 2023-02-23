import json
import logging
import sys

import pandas as pd


def readJSON(filepath, encoding='utf8'):
    """

    :param filepath: json文件路径
    :param encoding: 指定读取编码
    :return: json对象
    """
    with open(filepath, encoding=encoding) as f:
        return json.load(f)


def get_logger(logger_name, filename, level='info', mode='a', encoding='utf8'):
    """

    :param logger_name:
    :param filename:
    :param level:
    :param mode:
    :param encoding:
    :return:
    """
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }
    console_filter = logging.Filter()
    file_filter = logging.Filter()
    console_filter.filter = lambda record: record.levelno < level_relations[level]
    file_filter.filter = lambda record: record.levelno >= level_relations[level]
    logger = logging.getLogger(logger_name)
    file_handler = logging.FileHandler(filename, mode=mode, encoding=encoding)
    console_handler = logging.StreamHandler(sys.stdout)
    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s -%(lineno)d - %(message)s",
                                       datefmt='%Y-%m-%d %H:%M:%S')
    console_formater = logging.Formatter("%(asctime)s - %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    file_handler.addFilter(file_filter)
    console_handler.addFilter(console_filter)
    console_handler.setFormatter(console_formater)
    # file_handler.setLevel(logging.INFO)
    logger.setLevel(logging.DEBUG)
    if not logger.hasHandlers():
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    return logger


def preprocess(path, sheet_name):
    """
    对心总病数据进行预处理
    :param path: 心总病文件路径
    :param sheet_name: sheet的名称
    :return: X,y
    """
    xzb = pd.read_excel(path, sheet_name=sheet_name)
    xzb.drop(columns=['病案号'], inplace=True)
    xzb.drop(labels=[1742, 1741], axis=0, inplace=True)
    xzb.drop(labels=xzb[xzb['性别'].isna()].index, inplace=True)
    xzb = xzb.astype(int)
    y = xzb['证名']
    X = xzb.drop(labels=['证名', '性别', '年龄'], axis=1)
    return X, y
