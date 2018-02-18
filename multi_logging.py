import logging

formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')  # Format of log messages


# Allows for logging to be performed throughout the system and enables multiple logging destinations
def setup_logger(name, log_file, level=logging.INFO):
    # File handler
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    # Logger setup
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger
