"""
Transformer-based summarization module
"""
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import config
from utils import get_logger, handle_exceptions, chunk_text

logger = get_logger(__name__)

class TransformerSummarizer:
    def __init__(self, model_name=config.SUMMARIZATION_MODEL):
        logger.info(f"Loading summarization model: {model_name}")
        
        # Load tokenizer and model directly to bypass pipeline registration issues
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Setup device
        device_str = getattr(config, 'DEVICE', 'cpu')
        self.device = torch.device('cuda' if device_str == "cuda" and torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.max_length = config.MAX_SUMMARY_LENGTH
        self.min_length = config.MIN_SUMMARY_LENGTH
        logger.info("Summarization model loaded successfully.")

    @handle_exceptions(logger)
    def summarize(self, text, max_length=None, min_length=None):
        """
        Summarizes the given text. Handles long texts by chunking.
        """
        if not text or not text.strip():
            return ""

        max_len = max_length or self.max_length
        min_len = min_length or self.min_length

        # Chunk the text to handle max input limits (max 1024 for BART)
        chunks = chunk_text(text, chunk_size=700, overlap=50)
        
        summary_chunks = []
        for chunk in chunks:
            chunk_words = len(chunk.split())
            if chunk_words < min_len:
                summary_chunks.append(chunk)
                continue
                
            current_max = min(max_len, int(chunk_words * 0.8)) 
            current_min = min(min_len, int(chunk_words * 0.2))
            
            if current_max <= current_min:
                current_max = current_min + 1

            # Tokenize direct
            inputs = self.tokenizer(
                [chunk], 
                max_length=1024, 
                truncation=True, 
                return_tensors="pt"
            ).to(self.device)
            
            # Generate summary
            summary_ids = self.model.generate(
                inputs["input_ids"], 
                max_length=current_max, 
                min_length=current_min, 
                do_sample=False
            )
            
            # Decode resulting tokens
            summary_text = self.tokenizer.decode(
                summary_ids[0], 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
            
            summary_chunks.append(summary_text)

        final_summary = " ".join(summary_chunks)
        return final_summary
