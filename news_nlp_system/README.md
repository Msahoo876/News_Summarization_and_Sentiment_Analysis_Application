# NewsInsights Pro: Summarization & Sentiment Analysis

A production-grade, local-first Natural Language Processing (NLP) system designed to summarize long news articles and determine their underlying sentiment. This system strictly utilizes open-source Hugging Face transformer models and rule-based fallback mechanisms without any reliance on paid or external APIs.

## 🚀 Features
* **Abstractive Summarization:** Powered by `facebook/bart-large-cnn`, easily condensing long text into meaningful summaries.
* **Dual-Engine Sentiment Analysis:** Compares predictions from:
  * **DistilBERT** (`distilbert-base-uncased-finetuned-sst-2-english`) for robust, context-aware analysis.
  * **TextBlob** for a lightweight rule-based baseline.
* **Robust Text Preprocessing:** Includes URL removal, formatting normalizations, and NLTK-based cleaning.
* **Memory Safe Pipeline:** Implemented text chunking to safely process articles that exceed standard model token limits without crashing.
* **Interactive UI:** A highly responsive Streamlit application for end-to-end user interaction.

## 🛠️ Technology Stack
* **Language:** Python 3.9+
* **Core:** Hugging Face `transformers`, `torch`
* **NLP Utilities:** `nltk`, `textblob`, `spacy`
* **Frontend UI:** `streamlit`

## 📁 Project Structure
```
/news_nlp_system
│
├── app.py                    # Streamlit UI
├── summarizer.py             # Transformer-based summarization (BART)
├── sentiment.py              # Sentiment analysis (DistilBERT + TextBlob)
├── preprocessing.py          # Text cleaning & normalization
├── pipeline.py               # End-to-end workflow coordinating all parts
├── config.py                 # Constants & model configurations
├── utils.py                  # Logging and chunking helper functions
├── data/
│   └── sample_news.txt       # Example article for testing
├── requirements.txt
└── README.md
```

## ⚙️ Setup Instructions

### 1. Requirements Installation
Ensure you have Python installed. The system requires PyTorch and Transformers. To install the dependencies:
```bash
pip install -r requirements.txt
```

*(Note: NLTK dependencies like `punkt` and `stopwords` are automatically downloaded when the code runs).*

### 2. Running the Application
Launch the Streamlit server from the project directory:
```bash
streamlit run app.py
```
This will start a local server, and a web interface will automatically open in your default browser at `http://localhost:8501`.

*(Note: On the first run, Hugging Face models will be downloaded and cached locally. Subsequent runs will use the cached models).*

## 🧠 System Architecture Explanation

1. **Input Stage:** Raw text is submitted via the Streamlit UI. Configuration bounds (like summary lengths) are also read.
2. **Preprocessing (`preprocessing.py`):** The text undergoes sanitization (HTML cleanup, URL purging, and whitespace normalizing).
3. **Pipeline Orchestrator (`pipeline.py`):** The `NLPPipeline` routes the cleaned text.
4. **Summarization (`summarizer.py`):** If the text is extremely long, it applies a chunking algorithm to divide the document into transformer-safe blocks (~700 tokens), summarizes each chunk independently, and joins them seamlessly.
5. **Sentiment Analysis (`sentiment.py`):** Evaluates the entire cleaned text through DistilBERT (scaled down to 400 words to prevent context overflows) and parses the full text via TextBlob.
6. **Output Stage:** The Streamlit UI unpacks the structured JSON-like response, rendering statistics, summary, and multiple sentiment confidence gauges in real time.
