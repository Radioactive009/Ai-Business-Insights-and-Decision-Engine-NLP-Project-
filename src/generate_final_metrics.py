import pandas as pd
import numpy as np
import json
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, auc
import ast
from sentiment_model import predict_logit
from bert_model import predict_bert
from absa import absa_from_pos
from absa_llm import absa_llm

def calculate_metrics(y_true, y_pred, y_scores):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": roc_auc,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist()
    }

def main():
    print("Loading data...")
    df = pd.read_csv("../data/processed_reviews.csv")
    
    # Select a balanced sample for fair evaluation (200 reviews total)
    sample_size = 100 
    pos_sample = df[df["sentiment"] == "positive"].sample(sample_size, random_state=42)
    neg_sample = df[df["sentiment"] == "negative"].sample(sample_size, random_state=42)
    eval_df = pd.concat([pos_sample, neg_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    y_true = eval_df["sentiment"].map({"positive": 1, "negative": 0}).values
    
    results = {}
    
    # 1. Logistic Regression
    print("Evaluating Logistic Regression...")
    lr_preds = []
    lr_scores = []
    for text in eval_df["clean_text"]:
        res = predict_logit(text)
        lr_preds.append(1 if res == "positive" else 0)
        # We'll use 0.9/0.1 for scores since predict_logit only returns labels
        lr_scores.append(0.9 if res == "positive" else 0.1)
    results["Logistic Regression"] = calculate_metrics(y_true, lr_preds, lr_scores)
    
    # 2. BERT
    print("Evaluating BERT...")
    bert_preds = []
    bert_scores = []
    for text in eval_df["clean_text"]:
        res = predict_bert(text).lower()
        bert_preds.append(1 if res == "positive" else 0)
        bert_scores.append(0.85 if res == "positive" else 0.15)
    results["BERT (DistilBERT)"] = calculate_metrics(y_true, bert_preds, bert_scores)
    
    # 3. Rule-Based ABSA
    print("Evaluating Rule-Based ABSA...")
    rule_preds = []
    rule_scores = []
    for tags_str in eval_df["pos_tags"]:
        try:
            tags = ast.literal_eval(tags_str)
            rule_res = absa_from_pos(tags)
            pos_count = sum(1 for s in rule_res.values() if s == "positive")
            total = len(rule_res)
            score = pos_count / total if total > 0 else 0.5
            rule_scores.append(score)
            rule_preds.append(1 if score >= 0.5 else 0)
        except:
            rule_scores.append(0.5)
            rule_preds.append(0)
    results["Rule-Based ABSA"] = calculate_metrics(y_true, rule_preds, rule_scores)
    
    # 4. LLM-Based ABSA
    print("Evaluating LLM-Based ABSA (Llama3)...")
    llm_preds = []
    llm_scores = []
    for i, text in enumerate(eval_df["clean_text"]):
        if i % 10 == 0: print(f"  Processed {i}/{len(eval_df)}")
        try:
            llm_res = absa_llm(text)
            pos_count = sum(1 for s in llm_res.values() if s == "positive")
            total = len(llm_res)
            score = pos_count / total if total > 0 else 0.5
            llm_scores.append(score)
            llm_preds.append(1 if score >= 0.5 else 0)
        except:
            llm_scores.append(0.5)
            llm_preds.append(0)
    results["LLM-Based ABSA (Llama3)"] = calculate_metrics(y_true, llm_preds, llm_scores)
    
    # Save to JSON
    output_path = "../data/model_results.json"
    os.makedirs("../data", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
