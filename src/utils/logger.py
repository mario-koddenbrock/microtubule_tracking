import datetime
import logging
import logging.handlers
import os


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
LOG_LEVEL = logging.INFO

# --- Setup ---
def setup_logging(logger_name = 'mt', log_dir:str = os.path.abspath(".logs")):
    """Configures the application's logging."""

    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()
    log_file = os.path.join(log_dir, f"{logger_name}_{timestamp}.log")

    print(f"Logging to {log_file}")

    # 1. Get the logger
    logger = logging.getLogger(logger_name)
    # Set the lowest-level to be captured. DEBUG will capture everything.
    logger.setLevel(logging.DEBUG)

    # 2. Create a formatter
    formatter = logging.Formatter(LOG_FORMAT)

    # 3. Create and configure the console handler (for development)
    # This handler prints logs to the console (stdout).
    console_handler = logging.StreamHandler()
    # You can set a different level for the console. For example, INFO.
    # The level can be controlled by an environment variable for flexibility.
    console_log_level_str = os.environ.get('LOG_LEVEL', 'ERROR').upper()
    console_log_level = getattr(logging, console_log_level_str, logging.INFO)
    console_handler.setLevel(console_log_level)
    console_handler.setFormatter(formatter)

    # 4. Create and configure the rotating file handler (for production/history)
    # This handler writes logs to a file, with automatic rotation.
    file_handler = logging.FileHandler(log_file)

    # The file handler should typically log everything (DEBUG and up).
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(formatter)

    # 5. Add handlers to the logger
    # Avoid adding handlers if they already exist (e.g., in interactive sessions)
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    # 6. Log the successful configuration
    logger.info("Logging has been configured successfully. Console level: %s", console_log_level_str)

    return logger

