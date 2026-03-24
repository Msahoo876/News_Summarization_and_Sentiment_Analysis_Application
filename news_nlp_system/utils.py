"""
Utility functions for logging, error handling, and text chunking
"""
import logging
from functools import wraps

def get_logger(name):
    """Creates a configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

def handle_exceptions(logger):
    """Decorator to catch exceptions and log them."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                raise e
        return wrapper
    return decorator

def chunk_text(text, chunk_size=700, overlap=50):
    """
    Splits text into chunks of maximum token length (approximate).
    This is useful for transformers with max context windows (e.g., 1024 for BART).
    Using simple word splitting as an approximation (1 word ~ 1.3 tokens).
    overlap is used to not cut off sentences abruptly.
    """
    words = text.split()
    chunks = []
    
    if len(words) <= chunk_size:
        return [text]

    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        # Shift start by chunk_size - overlap to keep context
        start += (chunk_size - overlap)
        
    return chunks
