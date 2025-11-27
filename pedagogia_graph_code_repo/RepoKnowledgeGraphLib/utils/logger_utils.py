import logging
import os
import sys
import atexit

# Global registry to track initialized loggers
_initialized_loggers = set()

# Get log level from environment variable (default to INFO for visibility in docker logs)
DEFAULT_LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
LOG_TO_FILE = os.getenv('LOG_TO_FILE', 'false').lower() == 'true'

def setup_logger(logger_name: str, log_file: str = '',
                 level: int = None) -> None:
    """
    :param logger_name: name to give to logger
    :param log_file: file to save log to
    :param level: which base level of importance to set logger to (defaults to LOG_LEVEL env var)
    :return: *None*
    """
    # Check if logger has already been set up
    if logger_name in _initialized_loggers:
        return

    log = logging.getLogger(logger_name)

    # Determine log level from parameter, env var, or default
    if level is None:
        level = getattr(logging, DEFAULT_LOG_LEVEL, logging.INFO)

    formatter = logging.Formatter(
        fmt="%(name)s - %(levelname)s: %(asctime)-15s %(message)s")

    # Always add stream handler for stdout (docker logs visibility)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)

    log.setLevel(level)
    if not log.hasHandlers():
        log.addHandler(stream_handler)
        
        # Optionally add file handler if LOG_TO_FILE is enabled
        if LOG_TO_FILE:
            os.makedirs('logs', exist_ok=True)
            if log_file == '':
                log_file = f"{logger_name}.log"
            log_file_path = os.path.join('logs', log_file)
            file_handler = logging.FileHandler(log_file_path, mode='w')
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level)
            log.addHandler(file_handler)

    # Prevent log propagation to avoid duplicate logs
    log.propagate = False

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

