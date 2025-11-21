import logging
import os
import atexit

# Global registry to track initialized loggers
_initialized_loggers = set()

def setup_logger(logger_name: str, log_file: str = '',
                 level: int = logging.WARNING) -> None:
    """
    :param logger_name: name to give to logger
    :param log_file: file to save log to
    :param level: which base level of importance to set logger to
    :return: *None*
    """
    # Check if logger has already been set up
    if logger_name in _initialized_loggers:
        return

    log = logging.getLogger(logger_name)

    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)

    if log_file == '':
        log_file = f"{logger_name}.log"

    log_file_path = os.path.join('logs', log_file)

    formatter = logging.Formatter(
        fmt="%(name)s - %(levelname)s: %(asctime)-15s %(message)s")
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    log.setLevel(level)
    if not log.hasHandlers():
        log.addHandler(file_handler)
        log.addHandler(stream_handler)

    # Mark this logger as initialized
    _initialized_loggers.add(logger_name)

    # Register cleanup function to close handlers on exit
    atexit.register(_cleanup_logger, logger_name)

def _cleanup_logger(logger_name: str) -> None:
    """
    Clean up logger handlers on program exit.

    :param logger_name: name of the logger to clean up
    """
    log = logging.getLogger(logger_name)
    handlers = log.handlers[:]
    for handler in handlers:
        handler.close()
        log.removeHandler(handler)

