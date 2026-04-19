import pandas as pd
import torch
from transformers import pipeline

# ================================
# LOAD DATA
# ================================
df = pd.read_csv("../data/cleaned_reviews.csv")

# Take smaller sample (BERT is slow)
df = df.sample(1000, random_state=42)

texts = df["clean_text"].astype(str).tolist()


# ================================
# LOAD PRETRAINED BERT MODEL
# ================================
# This uses a pretrained sentiment model
classifier = pipeline("sentiment-analysis")


# ================================
# RUN PREDICTIONS
# ================================
results = classifier(texts)


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