"""
train.py — Model loading, Trainer setup with W&B tracking, training loop
MLOps Assignment 2 | IIT Jodhpur PGD AI Programme

Usage:
    python train.py
"""

import os

import torch
import wandb
from transformers import (
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from data import load_all_genres, split_data, encode_data
from utils import MyDataset, compute_metrics

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME  = "distilbert-base-cased"
MAX_LENGTH  = 512
OUTPUT_DIR  = "./results"
LOGGING_DIR = "./logs"
SAVED_MODEL = "distilbert-reviews-genres"

WANDB_PROJECT = "mlops-assignment2"
WANDB_RUN     = "distilbert-run-1"

HYPERPARAMS = {
    "model":         MODEL_NAME,
    "epochs":        3,
    "batch_size":    16,
    "learning_rate": 3e-5,
    "max_length":    MAX_LENGTH,
    "dataset":       "UCSD Goodreads",
    "warmup_steps":  100,
    "weight_decay":  0.01,
}


def get_device():
    """Return the best available device string."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def main():
    device = get_device()
    print(f"Using device: {device}")

    # 1. Data
    genre_reviews_dict = load_all_genres()
    train_texts, train_labels, test_texts, test_labels = split_data(genre_reviews_dict)

    (train_enc, test_enc,
     train_lbl_enc, test_lbl_enc,
     label2id, id2label, tokenizer) = encode_data(train_texts, train_labels,
                                                   test_texts, test_labels)

    train_dataset = MyDataset(train_enc, train_lbl_enc)
    test_dataset  = MyDataset(test_enc,  test_lbl_enc)

    # 2. W&B init
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN,
        config=HYPERPARAMS,
    )

    # 3. Model
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    ).to(device)

    # 4. Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=HYPERPARAMS["epochs"],
        per_device_train_batch_size=HYPERPARAMS["batch_size"],
        per_device_eval_batch_size=32,
        learning_rate=HYPERPARAMS["learning_rate"],
        warmup_steps=HYPERPARAMS["warmup_steps"],
        weight_decay=HYPERPARAMS["weight_decay"],
        logging_dir=LOGGING_DIR,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="wandb",          # single line enables full W&B logging
        run_name=WANDB_RUN,
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # 6. Train
    print("Starting training ...")
    trainer.train()

    # 7. Save model + tokenizer locally
    trainer.save_model(SAVED_MODEL)
    tokenizer.save_pretrained(SAVED_MODEL)
    print(f"Model saved to ./{SAVED_MODEL}")

    wandb.finish()
    return trainer, tokenizer, label2id, id2label, test_dataset, test_labels


if __name__ == "__main__":
    main()
