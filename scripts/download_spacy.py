import spacy
from spacy.cli.download import download
import sys

try:
    print("Attempting to load en_core_web_sm...")
    nlp = spacy.load("en_core_web_sm")
    print("Success! Model already exists.")
except OSError:
    print("Model not found. Downloading...")
    try:
        download("en_core_web_sm")
        print("Download complete.")
    except Exception as e:
        print(f"Download failed: {e}")
        # Build a minimal blank pipeline if download fails strictly for demo purposes
        # This prevents the whole system from crashing
