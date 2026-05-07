"""
train.py — Model loading, Trainer setup with W&B tracking, training loop
MLOps Assignment 2 | IIT Jodhpur PGD AI Programme

Usage:
    python train.py
"""

import argparse
import os

import torch
import wandb
from huggingface_hub import login
from transformers import (
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from data import load_all_genres, split_data, encode_data, TRAIN_PER_GENRE, TEST_PER_GENRE
from utils import MyDataset, compute_metrics

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME  = "distilbert-base-cased"
MAX_LENGTH  = 512
OUTPUT_DIR  = "./results"
LOGGING_DIR = "./logs"
SAVED_MODEL = "distilbert-reviews-genres"

WANDB_PROJECT = "distilbert-goodreads-genres"
WANDB_RUN     = "distilbert-run-1"
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "charancoursera9/distilbert-goodreads-genres")

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


def parse_args():
    parser = argparse.ArgumentParser(description="Train the DistilBERT Goodreads genre classifier")
    parser.add_argument("--quick", action="store_true",
                        help="Run a short low-resource training pass for CPU environments")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Per-device training batch size")
    parser.add_argument("--train-per-genre", type=int, default=None,
                        help="Number of samples per genre for training")
    parser.add_argument("--test-per-genre", type=int, default=None,
                        help="Number of samples per genre for testing")
    parser.add_argument("--wandb-run", type=str, default=WANDB_RUN,
                        help="W&B run name")
    parser.add_argument("--hf-model-repo", type=str,
                        default=os.getenv("HF_MODEL_REPO", HF_MODEL_REPO),
                        help="Hugging Face repository to push model/tokenizer to")
    parser.add_argument("--hf-token", type=str,
                        default=os.getenv("HF_TOKEN"),
                        help="Hugging Face token used to log in and push the model")
    return parser.parse_args()


def get_device():
    """Return the best available device string."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def push_model_to_hub(model, tokenizer, repo_name, hf_token=None):
    """Push the trained model and tokenizer to Hugging Face Hub."""
    hf_token = hf_token or os.getenv("HF_TOKEN")
    if not hf_token:
        print("HF_TOKEN is not set; skipping push to Hugging Face Hub.")
        return None

    print(f"Logging in to Hugging Face and pushing model/tokenizer to {repo_name} ...")
    login(token=hf_token)
    model.push_to_hub(repo_name)
    tokenizer.push_to_hub(repo_name)
    return f"https://huggingface.co/{repo_name}"


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    if args.quick:
        print("Quick mode enabled: reducing dataset size and training duration.")

    train_per_genre = args.train_per_genre if args.train_per_genre is not None else (
        20 if args.quick else TRAIN_PER_GENRE)
    test_per_genre = args.test_per_genre if args.test_per_genre is not None else (
        5 if args.quick else TEST_PER_GENRE)
    num_epochs = args.epochs if args.epochs is not None else (
        1 if args.quick else HYPERPARAMS["epochs"])
    batch_size = args.batch_size if args.batch_size is not None else (
        8 if args.quick else HYPERPARAMS["batch_size"])

    # 1. Data
    genre_reviews_dict = load_all_genres()
    train_texts, train_labels, test_texts, test_labels = split_data(
        genre_reviews_dict,
        train_per_genre=train_per_genre,
        test_per_genre=test_per_genre,
    )

    (train_enc, test_enc,
     train_lbl_enc, test_lbl_enc,
     label2id, id2label, tokenizer) = encode_data(train_texts, train_labels,
                                                   test_texts, test_labels)

    train_dataset = MyDataset(train_enc, train_lbl_enc)
    test_dataset  = MyDataset(test_enc,  test_lbl_enc)

    # 2. W&B init
    config = HYPERPARAMS.copy()
    config.update({
        "epochs": num_epochs,
        "batch_size": batch_size,
        "train_per_genre": train_per_genre,
        "test_per_genre": test_per_genre,
        "quick_mode": args.quick,
    })

    wandb.init(
        project=WANDB_PROJECT,
        name=args.wandb_run,
        config=config,
    )

    # 3. Model (Task 3: Load a pretrained Hugging Face model)
    # Load the tokenizer and model from Hugging Face with the correct number of output labels.
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    ).to(device)

    # 4. Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
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
        run_name=args.wandb_run,
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # 6. Train (Task 4: Run training with W&B tracking and evaluation)
    print("Starting training ...")
    trainer.train()

    # 7. Save model + tokenizer locally
    trainer.save_model(SAVED_MODEL)
    tokenizer.save_pretrained(SAVED_MODEL)
    print(f"Model saved to ./{SAVED_MODEL}")

    # 8. Push model to Hugging Face Hub (Task 6)
    hf_model_url = push_model_to_hub(model, tokenizer,
                                    args.hf_model_repo,
                                    hf_token=args.hf_token)
    if hf_model_url is not None:
        wandb.run.summary["huggingface_model"] = hf_model_url
        print(f"Logged Hugging Face model URL to W&B summary: {hf_model_url}")

    wandb.finish()
    return trainer, tokenizer, label2id, id2label, test_dataset, test_labels


if __name__ == "__main__":
    main()
