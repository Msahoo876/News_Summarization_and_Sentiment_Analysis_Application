"""
Streamlit Web UI for the News Summarization System
"""
import streamlit as st
import pandas as pd
from pipeline import NLPPipeline
import config

# Set page configs
st.set_page_config(
    page_title="News Insights Pro",
    page_icon="🗞️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize pipeline only once utilizing st.cache_resource
@st.cache_resource
def load_pipeline():
    return NLPPipeline()

def main():
    st.title("🗞️ NewsInsights Pro: Summarization & Sentiment")
    st.markdown("""
    **An intelligent, local-first NLP engine. No external APIs used.**
    This tool uses:
    * **BART/T5** for Abstractive Summarization
    * **DistilBERT & TextBlob** for Sentiment Analysis
    """)

    # Sidebar for parameters
    st.sidebar.header("⚙️ Pipeline Configuration")
    
    max_length = st.sidebar.slider(
        "Max Summary Length", 
        min_value=50, 
        max_value=500, 
        value=config.MAX_SUMMARY_LENGTH, 
        step=10
    )
    
    min_length = st.sidebar.slider(
        "Min Summary Length", 
        min_value=10, 
        max_value=200, 
        value=config.MIN_SUMMARY_LENGTH, 
        step=10
    )

    if min_length >= max_length:
        st.sidebar.error("Min length must be strictly less than max length.")

    # Main Interface
    st.subheader("Input News Article")
    
    # Optional sample text loading
    if st.button("Load Sample Article"):
        try:
            with open("data/sample_news.txt", "r") as f:
                st.session_state.text_input = f.read()
        except FileNotFoundError:
            st.warning("Sample file not found. Please paste text manually.")

    input_text = st.text_area(
        "", 
        height=300, 
        placeholder="Paste your news article here...",
        value=st.session_state.get('text_input', "")
    )
    
    # Character count limit
    if len(input_text) > config.MAX_INPUT_LENGTH:
        st.warning(f"Text is too long ({len(input_text)} chars). Please truncate to {config.MAX_INPUT_LENGTH} characters.")
        
    if st.button("🚀 Analyze Article", type="primary"):
        if not input_text.strip():
            st.error("Please enter some text to analyze.")
            return
            
        if min_length >= max_length:
            st.error("Invalid configuration: Min length must be strictly less than Max length.")
            return

        with st.spinner("Processing... Loading models might take a minute on first run."):
            try:
                pipeline = load_pipeline()
                results = pipeline.process_article(
                    input_text, 
                    max_summary_length=max_length, 
                    min_summary_length=min_length
                )
                
                # Render Results
                st.success("Analysis Complete!")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("📝 Summary")
                    st.info(results["summary"])
                    
                    st.subheader("📊 Text Statistics")
                    stats = results['original_stats']
                    st.write(f"**Word Count:** {stats['word_count']} | **Sentence Count:** {stats['sentence_count']}")
                
                with col2:
                    st.subheader("🎭 Sentiment Analysis")
                    
                    transformer_sent = results['sentiment']['transformer']
                    textblob_sent = results['sentiment']['textblob']
                    
                    st.markdown("### DistilBERT (Transformer)")
                    if transformer_sent['label'] == 'POSITIVE':
                        st.success(f"**Label:** {transformer_sent['label']}")
                    elif transformer_sent['label'] == 'NEGATIVE':
                        st.error(f"**Label:** {transformer_sent['label']}")
                    else:
                        st.warning(f"**Label:** {transformer_sent['label']}")
                    st.progress(transformer_sent['score'], text=f"Confidence: {transformer_sent['score']:.2f}")
                    
                    st.markdown("---")
                    
                    st.markdown("### TextBlob (Rule-based)")
                    if textblob_sent['label'] == 'POSITIVE':
                        st.success(f"**Label:** {textblob_sent['label']}")
                    elif textblob_sent['label'] == 'NEGATIVE':
                        st.error(f"**Label:** {textblob_sent['label']}")
                    else:
                        st.warning(f"**Label:** {textblob_sent['label']}")
                    st.progress(textblob_sent['score'], text=f"Polarity Score: {textblob_sent['score']:.2f}")

            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")

if __name__ == "__main__":
    main()
