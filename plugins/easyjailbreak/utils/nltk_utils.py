import nltk
import os
from pathlib import Path

from nltk.data import find
from nltk.corpus import words as nltk_words

NLTK_DATA_PATH = (
    os.environ.get("NLTK_DATA_PATH")
    or ("/opt/airflow/nltk_data" if os.path.isdir("/opt/airflow") else None)
    or str(Path.home() / "nltk_data")
)
os.makedirs(NLTK_DATA_PATH, exist_ok=True)
nltk.data.path.append(NLTK_DATA_PATH)
try:
    find("corpora/words")
except LookupError:
    nltk.download("words", download_dir=NLTK_DATA_PATH, quiet=True)
NWORDS = set(w.lower() for w in nltk_words.words())

__all__ = ["NWORDS", "NLTK_DATA_PATH"]
