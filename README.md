# MLOps Assignment 2 — DistilBERT Goodreads Genre Classifier

Fine-tuning a DistilBERT model to classify Goodreads book reviews by genre, with full experiment tracking via Weights & Biases and model publishing on Hugging Face Hub.

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
git clone https://github.com/<your-username>/mlops-assignment2.git
cd mlops-assignment2
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set environment variables

```bash
export WANDB_API_KEY=your_wandb_api_key
export HF_TOKEN=your_huggingface_token
```

### 4. Run the pipeline

**Step 1 — Data download & preprocessing**
```bash
python data.py
```

**Step 2 — Fine-tune the model (logs to W&B automatically)**
```bash
python train.py
```

**Step 3 — Evaluate and upload results to W&B**
```bash
python eval.py
```

> **GPU recommended.** Training on Google Colab free tier (T4 GPU) takes approximately 10–15 minutes. On CPU, reduce `SAMPLE_SIZE` in `data.py` to 200.

---

## Results

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 0.XX   |
| F1 Score  | 0.XX   |
| Eval Loss | 0.XX   |

*(Fill in your actual scores after training.)*

---

## Links

- **Hugging Face Model:** https://huggingface.co/your-username/distilbert-goodreads-genres
- **W&B Dashboard:** https://wandb.ai/your-username/mlops-assignment2

---

## Model Selection Rationale

DistilBERT (`distilbert-base-cased`) was chosen because it is a distilled version of BERT that is approximately 40% smaller and 60% faster while retaining over 95% of BERT's language understanding capability. For this assignment, where training time on free-tier Colab GPUs is a constraint, DistilBERT offers the best trade-off between performance and speed. The cased variant was preferred over the uncased version because book reviews may contain proper nouns (author names, titles) where capitalisation carries meaning.
