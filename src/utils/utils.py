"""
General utility functions.
"""

# 3rd party
import traceback

def log_info(message, logger=None, include_traceback=False):
    """Log information either to stdout or Python Logger if `logger` is not `None`.

    :param str message: log message
    :param Python Logger logger: logger. If `None`, message is printed to stdout
    :param bool include_traceback: if True, includes traceback (requires being called under and try/exception block). Defaults to False
    """
    
    if include_traceback:
        message += "\n" + traceback.format_exc()
        
    if logger:
        logger.info(message)
    else:
        print(message)
        