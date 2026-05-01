import pandas as pd
from sklearn.metrics import classification_report
from absa_llm import absa_llm

# Load data
df = pd.read_csv("../data/processed_reviews.csv")
df = df.dropna(subset=["clean_text", "sentiment"])

# Sample 20 reviews (LLM is slow)
sample_df = df.sample(20, random_state=42)

y_true = sample_df["sentiment"].tolist()
y_pred = []

print("Evaluating LLM-Based ABSA (Llama3) on 20 samples...")
for i, row in sample_df.iterrows():
    print(f"Processing {i+1}/20...")
    try:
        res = absa_llm(row["clean_text"])
        if not res:
            y_pred.append("neutral")
            continue
            
        pos_count = sum(1 for s in res.values() if s.lower() == "positive")
        neg_count = sum(1 for s in res.values() if s.lower() == "negative")
        
        if pos_count > neg_count:
            y_pred.append("positive")
        elif neg_count > pos_count:
            y_pred.append("negative")
        else:
            y_pred.append("neutral")
    except:
        y_pred.append("neutral")

print("\n===== LLM-BASED ABSA PERFORMANCE =====\n")
print(classification_report(y_true, y_pred, digits=4))
