import logging
import sys
import os
from datetime import datetime
from typing import Optional

class Logger:
    """
    A static logger class (Singleton pattern) that ensures a single 
    instance of the application logger is used throughout the application.

    It outputs logs to both the console and a daily, date-stamped log file.
    """
    _logger: Optional[logging.Logger] = None
    
    # --- Configuration ---
    APP_NAME = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    LOG_DIR = "log"
    CONSOLE_LEVEL = logging.INFO
    FILE_LEVEL = logging.DEBUG

    
    @staticmethod
    def initialize():
        """
        Initializes the logger instance. This method should be called once 
        at the start of the application.
        """
        if Logger._logger is not None:
            # Logger is already initialized, prevent re-initialization
            return

        # 1. Create the log directory if it doesn't exist
        try:
            os.makedirs(Logger.LOG_DIR, exist_ok=True)
        except OSError as e:
            # Fallback: if directory creation fails, only use a basic console logger
            print(f"Error creating directory {Logger.LOG_DIR}: {e}. Falling back to console-only logging.")
            Logger._logger = logging.getLogger(Logger.APP_NAME)
            Logger._logger.setLevel(Logger.CONSOLE_LEVEL)
            return

        # 2. Define the log filename based on current date
        # Format: app_name_yyyy_mm_dd.log
        today_date = datetime.now().strftime("%Y_%m_%d")
        log_filename = f"{Logger.APP_NAME}_{today_date}.log"
        log_filepath = os.path.join(Logger.LOG_DIR, log_filename)

        # 3. Get or create the logger instance
        logger = logging.getLogger(Logger.APP_NAME)
        logger.setLevel(logging.DEBUG) # Set lowest level to allow all handlers to filter
        logger.propagate = False # Prevent logs from propagating to the root logger

        # 4. Define the formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 5. Setup Console Handler (StreamHandler)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(Logger.CONSOLE_LEVEL) 
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 6. Setup File Handler (FileHandler, mode='a' for appending)
        file_handler = logging.FileHandler(log_filepath, mode='a', encoding='utf-8')
        file_handler.setLevel(Logger.FILE_LEVEL) # Captures everything (DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        Logger._logger = logger

    @staticmethod
    def _get_logger() -> logging.Logger:
        """Internal method to ensure the logger is initialized before use."""
        if Logger._logger is None:
            # Automatically initialize if a method is called before initialize()
            Logger.initialize()
            if Logger._logger is None:
                # Should not happen if initialize is successful, but for safety
                raise RuntimeError("Logger failed to initialize.")
        return Logger._logger

    @staticmethod
    def debug(message: str):
        """Logs a message at DEBUG level (File only)."""
        Logger._get_logger().debug(message)

    @staticmethod
    def info(message: str):
        """Logs a message at INFO level (Console and File)."""
        Logger._get_logger().info(message)

    @staticmethod
    def warning(message: str):
        """Logs a message at WARNING level (Console and File)."""
        Logger._get_logger().warning(message)

    @staticmethod
    def error(message: str):
        """Logs a message at ERROR level (Console and File)."""
        Logger._get_logger().error(message)

    @staticmethod
    def exception(message: str):
        """
        Logs a message and the current exception traceback at ERROR level 
        (Console and File). Must be called inside an except block.
        """
        Logger._get_logger().exception(message)