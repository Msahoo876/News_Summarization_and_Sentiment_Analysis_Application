"""
Configuration parameters for the NLP pipeline
"""

# Supported Huggingface Models
SUMMARIZATION_MODEL = "facebook/bart-large-cnn"
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

# Summarization constraints
MAX_SUMMARY_LENGTH = 150
MIN_SUMMARY_LENGTH = 50

# App Constraints
MAX_INPUT_LENGTH = 10000  # Max character length for inputs to prevent OOM
