# ============================================
# ASPECT-BASED SENTIMENT ANALYSIS (ABSA)
# ============================================

import pandas as pd
import ast

# ============================================
# LOAD DATA
# ============================================
# Loading the processed reviews that already contain POS tags and Entities
try:
    df = pd.read_csv("../data/processed_reviews.csv")
except FileNotFoundError:
    print("Error: processed_reviews.csv not found. Please run preprocessing.py first.")
    exit()

# Since lists are stored as strings in CSV, convert them back to Python objects
print("Parsing POS tags and Entities...")
df['pos_tags'] = df['pos_tags'].apply(ast.literal_eval)
df['entities'] = df['entities'].apply(ast.literal_eval)

# ============================================
# SENTIMENT LEXICON
# ============================================
positive_words = {"good", "great", "amazing", "excellent", "love", "beautiful", "fast", "reliable"}
negative_words = {"bad", "poor", "worst", "blurry", "hate", "slow", "expensive", "broken"}

# ============================================
# ABSA CORE LOGIC
# ============================================
def perform_absa(row, window=3):
    """
    Extracts sentiment for each entity based on nearby adjectives.
    """
    pos_tags = row['pos_tags']
    entities = row['entities']
    
    # Store results here: {entity_name: sentiment}
    results = {}
    
    # Process each entity found in the review
    for entity_tuple in entities:
        entity_name, entity_label = entity_tuple
        
        # 1. Find the position(s) of this entity in the token list
        # Note: Entities can be multi-word, so we check if any part matches
        entity_parts = entity_name.split()
        entity_indices = []
        
        for i, (word, tag) in enumerate(pos_tags):
            if word in entity_parts:
                entity_indices.append(i)
        
        if not entity_indices:
            continue
            
        # 2. Search for the CLOSEST adjective in the window
        best_distance = float('inf')
        found_sentiment = "neutral"
        
        for idx in entity_indices:
            start = max(0, idx - window)
            end = min(len(pos_tags), idx + window + 1)
            
            for j in range(start, end):
                if j == idx: continue # skip the entity itself
                
                word, tag = pos_tags[j]
                word_clean = word.lower().strip(".,!?;:")
                
                # Check if word is a known sentiment word
                is_pos = word_clean in positive_words
                is_neg = word_clean in negative_words
                
                if tag == "ADJECTIVE" or is_pos or is_neg:
                    dist = abs(j - idx)
                    if dist < best_distance:
                        best_distance = dist
                        if is_pos:
                            found_sentiment = "positive"
                        elif is_neg:
                            found_sentiment = "negative"
                        elif tag == "ADJECTIVE":
                            # If it's just an adjective not in our lexicon, we mark it
                            # but don't override a known sentiment word unless it's closer
                            if found_sentiment == "neutral":
                                found_sentiment = "detected_adjective"

        results[entity_name] = found_sentiment
        
    return results

# ============================================
# APPLY ABSA TO DATAFRAME
# ============================================
print("Performing Aspect-Based Sentiment Analysis...")
df['aspect_sentiment'] = df.apply(perform_absa, axis=1)

# ============================================
# OUTPUT SAMPLE RESULTS
# ============================================
print("\n===== ABSA RESULTS (SAMPLE) =====\n")
# Filter rows that actually have entities and sentiment detected
sample_df = df[df['aspect_sentiment'].apply(lambda x: len(x) > 0)].head(10)

for idx, row in sample_df.iterrows():
    print(f"Review: {row['clean_text'][:80]}...")
    print(f"Aspect Sentiments: {row['aspect_sentiment']}")
    print("-" * 50)

# ============================================
# MANUAL TEST CASE
# ============================================
print("\n===== MANUAL TEST CASE =====\n")
test_row = {
    'pos_tags': [
        ('The', 'DETERMINER'), ('camera', 'NOUN'), ('was', 'VERB'), ('blurry', 'ADJECTIVE'), 
        ('but', 'CONJUNCTION'), ('the', 'DETERMINER'), ('screen', 'NOUN'), ('is', 'VERB'), ('amazing', 'ADJECTIVE')
    ],
    'entities': [('camera', 'PRODUCT'), ('screen', 'PRODUCT')]
}

result = perform_absa(test_row)
print(f"Test Text: 'The camera was blurry but the screen is amazing.'")
print(f"Result: {result}")
