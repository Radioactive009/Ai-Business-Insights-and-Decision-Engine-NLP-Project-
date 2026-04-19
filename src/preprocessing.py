import pandas as pd
import spacy
import os
from tqdm import tqdm

# ================================
# LOAD SPACY MODEL
# ================================
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("SpaCy model 'en_core_web_sm' not found. Downloading...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


# ================================
# LOAD CLEANED CSV DATA
# ================================
# Ensuring the path works whether run from root or src directory
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(os.path.dirname(script_dir), "data", "cleaned_reviews.csv")

if not os.path.exists(data_path):
    # Fallback to current directory for flexibility
    data_path = "data/cleaned_reviews.csv"

print(f"Loading data from: {data_path}")
df = pd.read_csv(data_path)


# ================================
# BATCH NLP PROCESSING
# ================================
# We combine Tokenization, Stopword Removal, Lemmatization, POS Tagging, and NER 
# into a single efficient pass using nlp.pipe for much faster execution.

processed_texts = []
pos_tags_list = []
entities_list = []

print(f"Processing {len(df)} reviews using spaCy...")

# nlp.pipe is the optimized way to process large amounts of text in batches
for doc in tqdm(nlp.pipe(df["clean_text"].astype(str), batch_size=50), total=len(df)):
    
    # 1. TOKENIZATION + STOPWORD REMOVAL + LEMMATIZATION
    # spaCy splits sentence into individual words (tokens)
    # Convert word to base form (running → run) and skip common words (is, the, etc.)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    processed_texts.append(" ".join(tokens))
    
    # 2. POS TAGGING (Each token mapped to its grammatical role)
    pos_tags_list.append([(token.text, token.pos_) for token in doc])
    
    # 3. NAMED ENTITY RECOGNITION (NER) (Extract names, places, dates, etc.)
    entities_list.append([(ent.text, ent.label_) for ent in doc.ents])

# Add the results back to the dataframe
df["processed_text"] = processed_texts
df["pos_tags"] = pos_tags_list
df["entities"] = entities_list


# ================================
# DISPLAY OUTPUT
# ================================
print("\n===== SAMPLE OUTPUT =====\n")

print(df[[
    "clean_text",
    "processed_text",
    "pos_tags",
    "entities"
]].head())


# ================================
# SAVE PROCESSED DATA (FOR ML MODEL)
# ================================
output_path = os.path.join(os.path.dirname(script_dir), "data", "processed_reviews.csv")
print(f"\nSaving processed data to: {output_path}")
df.to_csv(output_path, index=False)
print("Saved successfully! ✅")