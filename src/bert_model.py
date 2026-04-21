import pandas as pd
import torch
from transformers import pipeline

# ================================
# LOAD DATA
# ================================
df = pd.read_csv("../data/processed_reviews.csv")

# Take smaller sample (BERT is slow)
# Note: Neutral reviews are now filtered in preprocessing.py
df = df.sample(1000, random_state=42)

texts = df["processed_text"].astype(str).tolist()


# ================================
# LOAD PRETRAINED BERT MODEL
# ================================
# This uses a pretrained sentiment model with truncation enabled
classifier = pipeline(
    "sentiment-analysis", 
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    truncation=True, 
    max_length=512
)


# ================================
# RUN PREDICTIONS
# ================================
# Pre-clipping text strings to avoid memory overhead and ensuring truncation is active
results = classifier(texts, truncation=True)


# ================================
# FORMAT OUTPUT
# ================================
predictions = []

for res in results:
    label = res["label"]
    
    # Convert labels to match your dataset
    if label == "POSITIVE":
        predictions.append("positive")
    elif label == "NEGATIVE":
        predictions.append("negative")
    else:
        predictions.append("neutral")


df["bert_sentiment"] = predictions


# ================================
# COMPARE WITH ORIGINAL
# ================================
from sklearn.metrics import classification_report

print("\n===== BERT PERFORMANCE =====\n")
print(classification_report(df["sentiment"], df["bert_sentiment"]))