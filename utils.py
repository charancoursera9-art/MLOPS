"""
utils.py — Shared helpers: label maps, dataset class, compute_metrics
MLOps Assignment 2 | IIT Jodhpur PGD AI Programme
"""

import torch
from sklearn.metrics import accuracy_score, f1_score


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class MyDataset(torch.utils.data.Dataset):
    """Custom PyTorch Dataset that wraps HuggingFace tokenizer encodings."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# ---------------------------------------------------------------------------
# Label utilities
# ---------------------------------------------------------------------------

def build_label_maps(label_list):
    """
    Build label2id and id2label dictionaries from a flat list of string labels.

    Args:
        label_list: iterable of string labels (may contain duplicates)

    Returns:
        label2id (dict), id2label (dict)
    """
    unique_labels = sorted(set(label_list))          # sorted for determinism
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label  = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(pred):
    """
    HuggingFace Trainer-compatible compute_metrics function.
    Returns accuracy and weighted F1 score.
    """
    labels = pred.label_ids
    preds  = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1":       f1_score(labels, preds, average="weighted"),
    }
