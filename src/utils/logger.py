import logging
import logging.handlers
import os

# --- Configuration ---
# 1. Set the name for the application's logger.
# Using a specific name avoids interfering with other libraries' loggers.
APP_LOGGER_NAME = 'my_project'

# 2. Define the log format.
# This format includes timestamp, log level, module name, and the message.
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"

# 3. Define the log file location and rotation settings.
# It's good practice to have a dedicated 'logs' directory.
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
os.makedirs(LOG_DIR, exist_ok=True) # Ensure the log directory exists
LOG_FILENAME = os.path.join(LOG_DIR, 'app.log')
# Rotate logs when the file reaches 2MB, keep 5 backup files.
MAX_BYTES = 2 * 1024 * 1024
BACKUP_COUNT = 5
# --- End of Configuration ---

# --- Setup ---
def setup_logging():
    """Configures the application's logging."""

    # 1. Get the logger
    logger = logging.getLogger(APP_LOGGER_NAME)
    # Set the lowest-level to be captured. DEBUG will capture everything.
    logger.setLevel(logging.DEBUG)

    # 2. Create a formatter
    formatter = logging.Formatter(LOG_FORMAT)

    # 3. Create and configure the console handler (for development)
    # This handler prints logs to the console (stdout).
    console_handler = logging.StreamHandler()
    # You can set a different level for the console. For example, INFO.
    # The level can be controlled by an environment variable for flexibility.
    console_log_level_str = os.environ.get('LOG_LEVEL', 'INFO').upper()
    console_log_level = getattr(logging, console_log_level_str, logging.INFO)
    console_handler.setLevel(console_log_level)
    console_handler.setFormatter(formatter)

    # 4. Create and configure the rotating file handler (for production/history)
    # This handler writes logs to a file, with automatic rotation.
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILENAME,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT
    )
    # The file handler should typically log everything (DEBUG and up).
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # 5. Add handlers to the logger
    # Avoid adding handlers if they already exist (e.g., in interactive sessions)
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    # 6. Log the successful configuration
    logger.info("Logging has been configured successfully. Console level: %s", console_log_level_str)

    return logger

