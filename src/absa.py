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
positive_words = {"good", "great", "amazing", "excellent", "love", "awesome", "perfect"}
negative_words = {"bad", "poor", "worst", "blurry", "hate", "terrible", "broken"}

# ============================================
# ABSA CORE FUNCTION
# ============================================
def absa_from_pos(pos_tags, window=3):
    """
    Identifies aspects (NOUNs) and maps them to the CLOSEST sentiment word in a window.
    Includes robust negation handling.
    """
    results = {}
    
    # 1. Iterate through tokens to find Aspects (NOUNs)
    for i, (word, tag) in enumerate(pos_tags):
        if tag in ["NOUN", "PROPER_NOUN"]:
            aspect_name = word.lower().strip(".,!?;:\"'()")
            
            # Skip noise
            if len(aspect_name) < 2 or aspect_name.isdigit() or aspect_name in ["it", "this"]:
                continue
                
            # 2. Search for all potential sentiments in window ±3
            start = max(0, i - window)
            end = min(len(pos_tags), i + window + 1)
            
            potential_sentiments = []
            
            for j in range(start, end):
                if i == j: continue
                
                curr_word, curr_tag = pos_tags[j]
                clean_word = curr_word.lower().strip(".,!?;:\"'()")
                
                is_pos = clean_word in positive_words
                is_neg = clean_word in negative_words
                
                if curr_tag == "ADJECTIVE" or is_pos or is_neg:
                    # Found a candidate adjective or sentiment word
                    # Determine base sentiment
                    sentiment = "positive" if is_pos else "negative" if is_neg else None
                    
                    # If it's just an ADJECTIVE but not in lexicon, we'll mark it as neutral/opinion
                    # but the user said only Positive/Negative, so we only proceed if sentiment is set
                    if not sentiment: continue
                        
                    # 3. Check for NEGATION in vicinity of the sentiment word
                    # Look at 1 or 2 words before the sentiment word
                    negated = False
                    for k in range(max(0, j-2), j):
                        prev_word, prev_tag = pos_tags[k]
                        if prev_tag == "NEGATION" or prev_word.lower().strip(".,!?;:") in ["not", "no", "never", "n't"]:
                            negated = True
                            break
                    
                    if negated:
                        sentiment = "negative" if sentiment == "positive" else "positive"
                    
                    potential_sentiments.append((abs(j - i), sentiment))
            
            # 4. Pick the sentiment with the SMALLEST distance to the aspect
            if potential_sentiments:
                # Sort by distance
                potential_sentiments.sort(key=lambda x: x[0])
                results[aspect_name] = potential_sentiments[0][1]
                
    return results

# ============================================
# APPLY ABSA TO DATAFRAME
# ============================================
print("Performing Aspect-Based Sentiment Analysis...")
df["aspect_sentiment"] = df["pos_tags"].apply(absa_from_pos)

# ============================================
# OUTPUT SAMPLE RESULTS
# ============================================
print("\n===== PROCESSED ABSA RESULTS (SAMPLE) =====\n")
# Show reviews where at least one aspect sentiment was found
sample_df = df[df["aspect_sentiment"].apply(lambda x: len(x) > 0)].head(10)

for idx, row in sample_df.iterrows():
    print(f"Text: {row['clean_text'][:80]}...")
    print(f"ABSA Output: {row['aspect_sentiment']}")
    print("-" * 50)

# ============================================
# MANUAL TEST CASES
# ============================================
print("\n===== MANUAL TEST CASES =====\n")

# Test 1: Standard Mixed Sentiment
test_1 = [
    ('The', 'DETERMINER'), ('camera', 'NOUN'), ('is', 'VERB'), ('blurry', 'ADJECTIVE'), 
    ('but', 'CONJUNCTION'), ('the', 'DETERMINER'), ('screen', 'NOUN'), ('is', 'VERB'), ('amazing', 'ADJECTIVE')
]

# Test 2: Negation Handling
test_2 = [
    ('This', 'DETERMINER'), ('phone', 'NOUN'), ("'s", 'NOUN'), ('display', 'NOUN'), ('is', 'VERB'), ('good', 'ADJECTIVE'), 
    ('but', 'CONJUNCTION'), ('battery', 'NOUN'), ('is', 'VERB'), ('not', 'NEGATION'), ('good', 'ADJECTIVE')
]

print(f"Test 1 Results: {absa_from_pos(test_1)}")
print(f"Test 2 Results: {absa_from_pos(test_2)}")
