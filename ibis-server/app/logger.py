import logging
from functools import wraps

from app.config import get_config

logging.basicConfig(level=get_config().log_level)


def get_logger(name):
    return logging.getLogger(name)


def log_dto(f):
    logger = get_logger("app.routers.ibis")

    @wraps(f)
    def wrapper(*args, **kwargs):
        logger.debug(f'DTO: {kwargs["dto"]}')
        return f(*args, **kwargs)

    return wrapper


def log_rewritten(f):
    logger = get_logger("app.mdl.rewriter")

    @wraps(f)
    def wrapper(*args, **kwargs):
        rs = f(*args, **kwargs)
        logger.debug(f"Rewritten SQL: {rs}")
        return rs

    return wrapper
