"""
Text preprocessing for NLP pipeline
"""
import re
import string
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from utils import get_logger, handle_exceptions

logger = get_logger(__name__)

class TextPreprocessor:
    def __init__(self):
        # We ensure stopwords and tokenizers are downloaded
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
            
        self.stop_words = set(stopwords.words('english'))
        logger.info("TextPreprocessor initialized.")

    @handle_exceptions(logger)
    def clean_text(self, text):
        """
        Applies necessary cleaning steps to the raw text.
        """
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    @handle_exceptions(logger)
    def extract_sentences(self, text):
        """Sentence tokenization."""
        return sent_tokenize(text)

    @handle_exceptions(logger)
    def remove_stopwords_and_lowercase(self, text):
        """
        Basic NLP normalization. Useful for classic bag-of-words or rule-based models.
        (Transformer models generally prefer raw/clean text, so this is selectively used.)
        """
        words = word_tokenize(text.lower())
        filtered = [w for w in words if w not in self.stop_words and w not in string.punctuation]
        return " ".join(filtered)
