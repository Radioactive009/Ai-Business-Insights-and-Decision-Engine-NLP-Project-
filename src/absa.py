# ============================================
# ASPECT-BASED SENTIMENT ANALYSIS (ABSA)
# ============================================

import pandas as pd
import ast

# ============================================
# LOAD DATA
# ============================================
try:
    df = pd.read_csv("../data/processed_reviews.csv")
    # Convert string representation of lists back to Python lists
    df['pos_tags'] = df['pos_tags'].apply(ast.literal_eval)
except FileNotFoundError:
    print("Error: processed_reviews.csv not found. Please run preprocessing.py first.")
    exit()

# ============================================
# SENTIMENT LEXICON
# ============================================
positive_words = {"good", "great", "amazing", "excellent", "love", "awesome", "perfect", "easy", "clear", "fast"}
negative_words = {"bad", "poor", "worst", "blurry", "hate", "terrible", "broken", "slow", "expensive", "difficult"}
aspect_stopwords = {"this", "that", "all", "just", "only", "also", "can", "of", "and", "but"}

# ============================================
# ABSA CORE FUNCTION
# ============================================
def absa_from_pos(pos_tags, window=3):
    """
    Identifies aspects (NOUNs) and links them to nearby sentiment words.
    Uses proximity-based matching and handles negation.
    """
    results = {}
    
    # 1. Iterate through tokens to find Aspects (NOUNs)
    for i, (word, tag) in enumerate(pos_tags):
        # Only consider NOUNs as potential aspects
        if tag == "NOUN":
            aspect_name = word.lower().strip(".,!?;:\"'()")
            
            # 2. FILTER INVALID ASPECTS
            if len(aspect_name) <= 2 or aspect_name in aspect_stopwords:
                continue
                
            # 3. Search for potential sentiments in window ±3
            start = max(0, i - window)
            end = min(len(pos_tags), i + window + 1)
            
            best_sentiment = None
            min_dist = float('inf')
            
            for j in range(start, end):
                if i == j: continue
                
                curr_word, curr_tag = pos_tags[j]
                clean_word = curr_word.lower().strip(".,!?;:\"'()")
                
                # Check for sentiment words in the lexicon
                is_pos = clean_word in positive_words
                is_neg = clean_word in negative_words
                
                # 4. If a sentiment word or ADJ is found
                if curr_tag == "ADJ" or is_pos or is_neg:
                    # Determine base sentiment (only if it's in our lexicon)
                    sentiment = "positive" if is_pos else "negative" if is_neg else None
                    
                    if not sentiment:
                        continue # Skip if it's just an adjective not in our lexicon
                        
                    # 5. NEGATION HANDLING
                    # Check for "not", "no", etc. in the 2 words preceding the sentiment word
                    negated = False
                    for k in range(max(0, j-2), j):
                        prev_word = pos_tags[k][0].lower().strip(".,!?;:")
                        if prev_word in ["not", "no", "never", "n't"] or pos_tags[k][1] == "NEGATION":
                            negated = True
                            break
                    
                    if negated:
                        sentiment = "negative" if sentiment == "positive" else "positive"
                    
                    # 6. Proximity-aware: track the closest sentiment word
                    dist = abs(j - i)
                    if dist < min_dist:
                        min_dist = dist
                        best_sentiment = sentiment
            
            # 7. STRICT ASSIGNMENT: Only add if a sentiment was found
            if best_sentiment:
                results[aspect_name] = best_sentiment
                
    return results

# ============================================
# APPLY ABSA TO DATAFRAME
# ============================================
print("Performing Aspect-Based Sentiment Analysis...")
df["aspect_sentiment"] = df["pos_tags"].apply(absa_from_pos)

# ============================================
# OUTPUT SAMPLE RESULTS
# ============================================
print("\n===== ABSA RESULTS: POSITIVE EXAMPLES =====\n")
pos_samples = df[df["aspect_sentiment"].apply(lambda x: "positive" in x.values())].head(3)
for _, row in pos_samples.iterrows():
    print(f"Text: {row['review_text'][:80]}...")
    print(f"Result: {row['aspect_sentiment']}")

print("\n===== ABSA RESULTS: NEGATIVE EXAMPLES =====\n")
neg_samples = df[df["aspect_sentiment"].apply(lambda x: "negative" in x.values())].head(3)
for _, row in neg_samples.iterrows():
    print(f"Text: {row['review_text'][:80]}...")
    print(f"Result: {row['aspect_sentiment']}")

# ============================================
# GENERIC TEST CASE
# ============================================
print("\n===== GENERIC TEST CASE =====\n")
# Contrasting sentiments in a single sentence
test_text = "The camera is blurry but the screen is amazing."
# Simulated POS tags (representing the output of our custom tagger)
simulated_pos = [
    ('The', 'DETERMINER'), ('camera', 'NOUN'), ('is', 'VERB'), ('blurry', 'ADJ'), 
    ('but', 'CONJUNCTION'), ('the', 'DETERMINER'), ('screen', 'NOUN'), ('is', 'VERB'), ('amazing', 'ADJ')
]

result = absa_from_pos(simulated_pos)
print(f"Input: '{test_text}'")
print(f"Output: {result}")
