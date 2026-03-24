"""
End-to-end NLP workflow pipeline
"""
from preprocessing import TextPreprocessor
from summarizer import TransformerSummarizer
from sentiment import SentimentAnalyzer
from utils import get_logger, handle_exceptions

logger = get_logger(__name__)

class NLPPipeline:
    def __init__(self):
        logger.info("Initializing NLP Pipeline...")
        self.preprocessor = TextPreprocessor()
        self.summarizer = TransformerSummarizer()
        self.sentiment_analyzer = SentimentAnalyzer()
        logger.info("NLP Pipeline initialized successfully.")

    @handle_exceptions(logger)
    def process_article(self, text, max_summary_length=None, min_summary_length=None):
        """
        Executes the full pipeline:
        1. Preprocess/clean text
        2. Generate summary
        3. Analyze sentiment of the summary (or original text - here we use original for better context)
        4. Return structured dictionary
        """
        logger.info("Starting article processing")
        
        # 1. Clean Text
        cleaned_text = self.preprocessor.clean_text(text)
        
        if not cleaned_text:
            raise ValueError("Text is empty after cleaning.")

        # 2. Extract Sentences just to provide some stats
        sentences = self.preprocessor.extract_sentences(cleaned_text)
        stats = {
            "char_count": len(cleaned_text),
            "word_count": len(cleaned_text.split()),
            "sentence_count": len(sentences)
        }

        # 3. Summarization
        logger.info("Generating summary...")
        summary = self.summarizer.summarize(
            cleaned_text,
            max_length=max_summary_length,
            min_length=min_summary_length
        )

        # 4. Sentiment Analysis
        # We perform sentiment analysis on the raw (cleaned) text to capture the full context
        logger.info("Analyzing sentiment...")
        sentiment_results = self.sentiment_analyzer.analyze_all(cleaned_text)

        logger.info("Processing complete.")
        
        return {
            "original_stats": stats,
            "cleaned_text": cleaned_text,
            "summary": summary,
            "sentiment": sentiment_results
        }
