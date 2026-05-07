"""
eval.py — Evaluation, metrics computation, saving results, and W&B artifact upload
MLOps Assignment 2 | IIT Jodhpur PGD AI Programme

Usage:
    python eval.py
    (Run after train.py, or pass --retrain to trigger training first.)
"""

import argparse
import json
import os
import pickle

import wandb
from sklearn.metrics import classification_report
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

from data import load_all_genres, split_data, encode_data
from utils import MyDataset, build_label_maps, compute_metrics

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SAVED_MODEL     = "distilbert-reviews-genres"
EVAL_REPORT_PATH = "eval_report.json"
WANDB_PROJECT   = "mlops-assignment2"
WANDB_RUN       = "distilbert-eval"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_trained_model_and_tokenizer(saved_model_dir=SAVED_MODEL):
    """Load the locally saved model and tokenizer."""
    print(f"Loading model from {saved_model_dir} ...")
    model     = DistilBertForSequenceClassification.from_pretrained(saved_model_dir)
    tokenizer = DistilBertTokenizerFast.from_pretrained(saved_model_dir)
    return model, tokenizer


def build_test_dataset(tokenizer, label2id):
    """Rebuild the test dataset using the same data pipeline."""
    genre_reviews_dict = load_all_genres()
    _, _, test_texts, test_labels = split_data(genre_reviews_dict)

    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)
    test_labels_encoded = [label2id[y] for y in test_labels]
    test_dataset = MyDataset(test_encodings, test_labels_encoded)
    return test_dataset, test_labels


def make_minimal_trainer(model, test_dataset):
    """Create a Trainer with minimal args just for prediction/evaluation."""
    args = TrainingArguments(
        output_dir="./eval_tmp",
        per_device_eval_batch_size=32,
        report_to=[],
    )
    return Trainer(
        model=model,
        args=args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )


# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------

def main():
    # 1. W&B init
    run = wandb.init(project=WANDB_PROJECT, name=WANDB_RUN, job_type="eval")

    # 2. Load model
    model, tokenizer = load_trained_model_and_tokenizer()
    id2label = model.config.id2label
    label2id = model.config.label2id

    # 3. Rebuild test data
    test_dataset, test_labels = build_test_dataset(tokenizer, label2id)

    # 4. Trainer for evaluation
    trainer = make_minimal_trainer(model, test_dataset)

    # 5. Run evaluation
    print("Running evaluation ...")
    eval_results = trainer.evaluate()
    print("Evaluation results:", eval_results)

    # 6. Log final metrics to W&B
    wandb.log({
        "final/loss":     eval_results.get("eval_loss", None),
        "final/accuracy": eval_results.get("eval_accuracy", None),
        "final/f1":       eval_results.get("eval_f1", None),
    })

    # 7. Full classification report
    preds = trainer.predict(test_dataset).predictions.argmax(-1)
    predicted_str = [id2label[i] for i in preds]

    report = classification_report(
        test_labels,
        predicted_str,
        target_names=list(id2label.values()),
        output_dict=True,
    )

    print("\nClassification Report:")
    print(classification_report(test_labels, predicted_str,
                                target_names=list(id2label.values())))

    # 8. Save classification report to JSON
    with open(EVAL_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Classification report saved to {EVAL_REPORT_PATH}")

    # 9. Upload report as a versioned W&B Artifact
    artifact = wandb.Artifact("eval-report", type="evaluation")
    artifact.add_file(EVAL_REPORT_PATH)
    wandb.log_artifact(artifact)
    print("Artifact uploaded to W&B.")

    wandb.finish()


if __name__ == "__main__":
    main()
