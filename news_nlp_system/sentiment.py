"""
Sentiment analysis module using TextBlob and DistilBERT
"""
from textblob import TextBlob
from transformers import pipeline
import config
from utils import get_logger, handle_exceptions

logger = get_logger(__name__)

class SentimentAnalyzer:
    def __init__(self, model_name=config.SENTIMENT_MODEL):
        logger.info(f"Loading sentiment model: {model_name}")
        device = getattr(config, 'DEVICE', 'cpu')
        device_id = 0 if device == "cuda" else -1
        
        self.classifier = pipeline(
            "sentiment-analysis", 
            model=model_name,
            device=device_id
        )
        logger.info("Sentiment model loaded successfully.")

    @handle_exceptions(logger)
    def analyze_textblob(self, text):
        """
        Rule-based sentiment analysis using TextBlob.
        """
        if not text or not text.strip():
            return {"label": "NEUTRAL", "score": 0.0}

        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            label = "POSITIVE"
        elif polarity < -0.1:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"
            
        # Normalize score to 0-1 for consistency with transformer
        score = abs(polarity) 
        
        return {
            "label": label,
            "score": round(score, 4)
        }

    @handle_exceptions(logger)
    def analyze_transformer(self, text):
        """
        Transformer-based sentiment analysis.
        Note: DistilBERT usually has max token length of 512. 
        For long texts, we truncate to the first 400 words approximately.
        """
        if not text or not text.strip():
            return {"label": "NEUTRAL", "score": 0.0}

        # Truncate to avoid index errors on very long documents
        words = text.split()
        if len(words) > 400:
            truncated_text = " ".join(words[:400])
        else:
            truncated_text = text

        result = self.classifier(truncated_text)[0]
        
        # Models often return labels like 'POSITIVE' or 'NEGATIVE'
        return {
            "label": str(result['label']).upper(),
            "score": round(result['score'], 4)
        }

    @handle_exceptions(logger)
    def analyze_all(self, text):
        """
        Returns results from both approaches for comparison.
        """
        return {
            "textblob": self.analyze_textblob(text),
            "transformer": self.analyze_transformer(text)
        }
