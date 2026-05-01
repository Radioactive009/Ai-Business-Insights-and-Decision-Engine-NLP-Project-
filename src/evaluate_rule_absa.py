import pandas as pd
import ast
from sklearn.metrics import classification_report

# Re-implementing the core logic here to avoid importing absa.py which runs on the whole dataset
positive_words = {"good", "great", "amazing", "excellent", "love"}
negative_words = {"bad", "poor", "worst", "blurry", "hate"}
invalid_words = {
    "is", "was", "are", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "can", "could", "should", "would",
    "just", "only", "also", "very", "so", "too",
    "write", "read", "got", "make", "part", "know",
    "right", "main", "all", "of", "and", "but", "this", "that",
    "i", "me", "my", "you", "your", "he", "she", "it", "we", "they",
    "who", "what", "which", "when", "where", "how", "why"
}

def absa_from_pos(pos_tags, window=3):
    results = {}
    for i, (word, tag) in enumerate(pos_tags):
        clean_word = word.strip(".,!?;:\"'()[]").lower()
        if tag == "NOUN" and len(clean_word) > 2 and clean_word.isalpha():
            if clean_word in invalid_words: continue
            aspect = clean_word
            start = max(0, i - window)
            end = min(len(pos_tags), i + window + 1)
            best_sentiment = None
            min_dist = float('inf')
            for j in range(start, end):
                if i == j: continue
                s_word, s_tag = pos_tags[j]
                s_word_clean = s_word.strip(".,!?;:\"'()[]").lower()
                is_pos = s_word_clean in positive_words
                is_neg = s_word_clean in negative_words
                if is_pos or is_neg:
                    base_sentiment = "positive" if is_pos else "negative"
                    negated = False
                    for k in range(max(0, j-2), j):
                        prev_word = pos_tags[k][0].strip(".,!?;:\"'()[]").lower()
                        if prev_word in ["not", "no", "never", "n't"]:
                            negated = True
                            break
                    final_sentiment = base_sentiment
                    if negated:
                        final_sentiment = "negative" if base_sentiment == "positive" else "positive"
                    dist = abs(j - i)
                    if dist < min_dist:
                        min_dist = dist
                        best_sentiment = final_sentiment
            if best_sentiment:
                results[aspect] = best_sentiment
    return results

# Load data
df = pd.read_csv("../data/processed_reviews.csv")
df = df.dropna(subset=["pos_tags", "sentiment"])

# Sample 1000 reviews
sample_df = df.sample(1000, random_state=42)

y_true = sample_df["sentiment"].tolist()
y_pred = []

print("Evaluating Rule-Based ABSA on 1000 samples...")
for i, row in sample_df.iterrows():
    try:
        tags = ast.literal_eval(row["pos_tags"])
        res = absa_from_pos(tags)
        
        pos_count = sum(1 for s in res.values() if s == "positive")
        neg_count = sum(1 for s in res.values() if s == "negative")
        
        if pos_count > neg_count:
            y_pred.append("positive")
        elif neg_count > pos_count:
            y_pred.append("negative")
        else:
            # Fallback to the majority class or a neutral prediction
            # Since the ground truth is binary, neutral will be a miss.
            y_pred.append("neutral")
    except:
        y_pred.append("neutral")

print("\n===== RULE-BASED ABSA PERFORMANCE =====\n")
print(classification_report(y_true, y_pred, digits=4))
