"""
data.py — Data loading, sampling, train/test split, and BERT encoding
MLOps Assignment 2 | IIT Jodhpur PGD AI Programme

Usage:
    python data.py
"""

import gzip
import json
import pickle
import random
import requests

from transformers import DistilBertTokenizerFast

from utils import build_label_maps

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "distilbert-base-cased"
MAX_LENGTH = 512
SAMPLE_SIZE = 2000      # reviews to sample per genre from the first HEAD rows
HEAD = 10000            # max rows to stream per genre
TRAIN_PER_GENRE = 800   # reviews per genre used for training
TEST_PER_GENRE = 200    # reviews per genre used for testing
RANDOM_SEED = 42

GENRE_URL_DICT = {
    "poetry":                 "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz",
    "comics_graphic":         "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz",
    "fantasy_paranormal":     "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz",
    "history_biography":      "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz",
    "mystery_thriller_crime": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz",
    "romance":                "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz",
    "young_adult":            "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_reviews(url, head=HEAD, sample_size=SAMPLE_SIZE, seed=RANDOM_SEED):
    """
    Stream reviews from a remote gzipped JSON-lines file.

    Args:
        url:         Direct HTTPS URL to a .json.gz file.
        head:        Maximum number of rows to read from the stream.
        sample_size: Number of reviews to randomly sample from `head` rows.
        seed:        Random seed for reproducibility.

    Returns:
        List of review text strings.
    """
    random.seed(seed)
    reviews = []
    count = 0

    response = requests.get(url, stream=True)
    response.raise_for_status()

    with gzip.open(response.raw, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            text = d.get("review_text", "").strip()
            if text:
                reviews.append(text)
            count += 1
            if head is not None and count >= head:
                break

    return random.sample(reviews, min(sample_size, len(reviews)))


def load_all_genres(genre_url_dict=GENRE_URL_DICT, cache_path="genre_reviews_dict.pickle"):
    """
    Download (or load from cache) reviews for all genres.

    Returns:
        genre_reviews_dict: {genre_name: [review_text, ...]}
    """
    try:
        print(f"Loading cached data from {cache_path} ...")
        genre_reviews_dict = pickle.load(open(cache_path, "rb"))
        print("Cache loaded successfully.")
        return genre_reviews_dict
    except FileNotFoundError:
        pass

    genre_reviews_dict = {}
    for genre, url in genre_url_dict.items():
        print(f"Downloading reviews for genre: {genre}")
        genre_reviews_dict[genre] = load_reviews(url)

    pickle.dump(genre_reviews_dict, open(cache_path, "wb"))
    print(f"Data cached to {cache_path}.")
    return genre_reviews_dict


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def split_data(genre_reviews_dict, train_per_genre=TRAIN_PER_GENRE,
               test_per_genre=TEST_PER_GENRE, seed=RANDOM_SEED):
    """
    Produce flat train/test lists of texts and string labels.

    Args:
        genre_reviews_dict: {genre: [review, ...]}
        train_per_genre:    Reviews per genre for training.
        test_per_genre:     Reviews per genre for testing.
        seed:               Random seed.

    Returns:
        train_texts, train_labels, test_texts, test_labels
    """
    random.seed(seed)
    train_texts, train_labels = [], []
    test_texts,  test_labels  = [], []

    total_per_genre = train_per_genre + test_per_genre

    for genre, reviews in genre_reviews_dict.items():
        sampled = random.sample(reviews, min(total_per_genre, len(reviews)))
        for review in sampled[:train_per_genre]:
            train_texts.append(review)
            train_labels.append(genre)
        for review in sampled[train_per_genre: train_per_genre + test_per_genre]:
            test_texts.append(review)
            test_labels.append(genre)

    print(f"Train: {len(train_texts)} samples | Test: {len(test_texts)} samples")
    return train_texts, train_labels, test_texts, test_labels


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_data(train_texts, train_labels, test_texts, test_labels,
                model_name=MODEL_NAME, max_length=MAX_LENGTH):
    """
    Tokenize texts and encode string labels to integers.

    Returns:
        train_encodings, test_encodings,
        train_labels_encoded, test_labels_encoded,
        label2id, id2label, tokenizer
    """
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

    print("Tokenising train texts ...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True,
                                max_length=max_length)
    print("Tokenising test texts ...")
    test_encodings  = tokenizer(test_texts,  truncation=True, padding=True,
                                max_length=max_length)

    label2id, id2label = build_label_maps(train_labels)

    train_labels_encoded = [label2id[y] for y in train_labels]
    test_labels_encoded  = [label2id[y] for y in test_labels]

    print(f"Label map: {label2id}")
    return (train_encodings, test_encodings,
            train_labels_encoded, test_labels_encoded,
            label2id, id2label, tokenizer)


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    genre_reviews_dict = load_all_genres()
    train_texts, train_labels, test_texts, test_labels = split_data(genre_reviews_dict)
    (train_enc, test_enc,
     train_lbl_enc, test_lbl_enc,
     label2id, id2label, tokenizer) = encode_data(train_texts, train_labels,
                                                   test_texts, test_labels)
    print("Data pipeline complete.")
    print(f"  Train encodings keys : {list(train_enc.keys())}")
    print(f"  Unique encoded labels: {set(train_lbl_enc)}")
