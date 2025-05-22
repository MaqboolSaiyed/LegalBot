import nltk
import logging

def download_nltk_data():
    """Download required NLTK data."""
    try:
        # Download required NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('stopwords', quiet=True)
        logging.info("Successfully downloaded NLTK data")
    except Exception as e:
        logging.error(f"Error downloading NLTK data: {e}")
        raise 