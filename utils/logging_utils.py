"""
Logging Utilities

Setup logging for the framework.
"""

import logging


def setup_logging(level: str = "INFO"):
    """
    Setup logging configuration.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )