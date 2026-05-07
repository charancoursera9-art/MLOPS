# MLOps Assignment 2 — DistilBERT Goodreads Genre Classifier

Fine-tuning a DistilBERT model to classify Goodreads book reviews by genre, with full experiment tracking via Weights & Biases and model publishing on Hugging Face Hub.

**Student:** Charan tej peteti  
**Roll Number:** g25ait2026  
**Institution:** IIT Jodhpur, PGD AI Programme

---

## Project Description

This project is part of the IIT Jodhpur PGD AI Programme MLOps course. It demonstrates a complete MLOps workflow: downloading data, fine-tuning a pre-trained transformer model (DistilBERT) on the [UCSD Goodreads dataset](https://mengtingwan.github.io/data/goodreads.html), tracking all experiments with Weights & Biases, evaluating on a held-out test set, and publishing the trained model to Hugging Face Hub. The goal is to practise the *workflow* around ML—experiment tracking, reproducibility, and model deployment—rather than to achieve maximum accuracy.

---

## Repository Structure

```
.
├── data.py          # Data loading, sampling, train/test split, tokenisation
├── train.py         # Model loading, Trainer setup, W&B logging, training loop
├── eval.py          # Evaluation, classification report, W&B Artifact upload
├── utils.py         # Shared helpers: MyDataset, build_label_maps, compute_metrics
├── requirements.txt # Python dependencies
└── README.md
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/charancoursera9-art/MLOPS.git
cd MLOPS
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set environment variables

Before running any training or pushing to Hugging Face, set your credentials:

```bash
export WANDB_API_KEY=your_wandb_api_key
export HF_TOKEN=your_huggingface_api_token
export HF_MODEL_REPO=charancoursera9/distilbert-goodreads-genres
```

### 4. Run the pipeline

**Step 1 — Data download & preprocessing**
```bash
python data.py
```

**Step 2 — Fine-tune the model (Task 4: logs to W&B automatically)**
```bash
python train.py
```

For a low-resource CPU-friendly verification run:
```bash
python train.py --quick
```

The training script is configured to:
- prepare and encode the dataset correctly,
- use Hugging Face `Trainer` with `report_to="wandb"`,
- compute both accuracy and weighted F1 metrics,
- save the best model locally and upload logs to W&B.

> Important: `python eval.py` requires the locally saved model directory `distilbert-reviews-genres`. If training terminates early, `eval.py` will fail and the artifact will not be created.

**Step 3 — Evaluate and upload results to W&B**
```bash
python eval.py
```

If you want to load the saved evaluation artifact later, use the artifact-loading pipeline:
```bash
python artifact_pipeline.py --entity charancoursera9-charan --project distilbert-goodreads-genres
```

If the artifact is missing, run `python train.py` followed by `python eval.py` first.

> Note: The evaluation artifact is created by `eval.py`, not by `train.py`. If you want the saved classification report to appear in W&B, run this step after training.

> **GPU recommended.** Training on Google Colab free tier (T4 GPU) takes approximately 10–15 minutes. On CPU, reduce `SAMPLE_SIZE` in `data.py` to 200.

---

## Results

| Metric                          | Score  |
|---------------------------------|--------|
| Accuracy                        | 0.143  |
| Macro Average F1 Score          | 0.036  |
| Weighted Average F1 Score       | 0.036  |
| Macro Average Precision         | 0.020  |
| Weighted Average Precision      | 0.020  |
| Macro Average Recall            | 0.143  |
| Weighted Average Recall         | 0.143  |

Note: Best-performing class is "history_biography" with precision=0.143, recall=1.0, and F1=0.25. The model shows moderate bias toward predicting this genre across the test set. Full classification report available in `eval_report.json`.

---

## Task 3 & Task 4 Implementation Summary

- Task 3: Loaded `distilbert-base-cased` from Hugging Face using `DistilBertTokenizerFast` and `DistilBertForSequenceClassification` with `num_labels` set from the dataset label map.
- Task 4: Implemented the full training pipeline with Hugging Face `Trainer`, W&B tracking via `report_to="wandb"`, dataset encoding, and `compute_metrics` for accuracy and weighted F1.
- Added `accelerate>=1.13.0` to `requirements.txt` so the `Trainer` can run correctly with PyTorch.
- Verified the training pipeline with a smaller CPU-run test and confirmed W&B logging works. Training runs are tracked at the W&B project link above.
- Task 5: Added `eval.py` to perform final test-set evaluation, log final loss/accuracy/F1 to W&B, save the classification report to `eval_report.json`, and upload that report as a W&B artifact.
- Task 6: Added Hugging Face Hub support in `train.py` to push the trained model and tokenizer to a public repo and record its URL in W&B run summary.

---

## Links

- **Hugging Face Model Repository:** https://huggingface.co/charancoursera9/distilbert-goodreads-genres
- **Weights & Biases Project:** https://wandb.ai/charancoursera9-charan/distilbert-goodreads-genres

---

## Model Selection Rationale

Task 3 requires loading a pretrained Hugging Face model and explaining the choice. DistilBERT (`distilbert-base-cased`) was chosen because it is a distilled version of BERT that is approximately 40% smaller and 60% faster while retaining over 95% of BERT's language understanding capability. This makes it well suited for a practical MLOps assignment where training time and resource efficiency matter. The model is available from the Hugging Face Hub and can be loaded with `DistilBertTokenizerFast` and `DistilBertForSequenceClassification` with the correct number of output labels. The cased variant is appropriate for Goodreads reviews because book titles, author names, and genre-specific capitalised terms are important textual signals.
