# import pandas as pd
# import spacy
# import os
# from tqdm import tqdm

# # ================================
# # LOAD SPACY MODEL
# # ================================
# try:
#     nlp = spacy.load("en_core_web_sm")
# except OSError:
#     print("SpaCy model 'en_core_web_sm' not found. Downloading...")
#     import subprocess
#     import sys
#     subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
#     nlp = spacy.load("en_core_web_sm")


# # ================================
# # LOAD CLEANED CSV DATA
# # ================================
# # Ensuring the path works whether run from root or src directory
# script_dir = os.path.dirname(os.path.abspath(__file__))
# data_path = os.path.join(os.path.dirname(script_dir), "data", "cleaned_reviews.csv")

# if not os.path.exists(data_path):
#     # Fallback to current directory for flexibility
#     data_path = "data/cleaned_reviews.csv"

# print(f"Loading data from: {data_path}")
# df = pd.read_csv(data_path)


# # ================================
# # BATCH NLP PROCESSING
# # ================================
# # We combine Tokenization, Stopword Removal, Lemmatization, POS Tagging, and NER 
# # into a single efficient pass using nlp.pipe for much faster execution.

# processed_texts = []
# pos_tags_list = []
# entities_list = []

# print(f"Processing {len(df)} reviews using spaCy...")

# # nlp.pipe is the optimized way to process large amounts of text in batches
# for doc in tqdm(nlp.pipe(df["clean_text"].astype(str), batch_size=50), total=len(df)):
    
#     # 1. TOKENIZATION + STOPWORD REMOVAL + LEMMATIZATION
#     # spaCy splits sentence into individual words (tokens)
#     # Convert word to base form (running → run) and skip common words (is, the, etc.)
#     tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
#     processed_texts.append(" ".join(tokens))
    
#     # 2. POS TAGGING (Each token mapped to its grammatical role)
#     pos_tags_list.append([(token.text, token.pos_) for token in doc])
    
#     # 3. NAMED ENTITY RECOGNITION (NER) (Extract names, places, dates, etc.)
#     entities_list.append([(ent.text, ent.label_) for ent in doc.ents])

# # Add the results back to the dataframe
# df["processed_text"] = processed_texts
# df["pos_tags"] = pos_tags_list
# df["entities"] = entities_list


# # ================================
# # DISPLAY OUTPUT
# # ================================
# print("\n===== SAMPLE OUTPUT =====\n")

# print(df[[
#     "clean_text",
#     "processed_text",
#     "pos_tags",
#     "entities"
# ]].head())


# # ================================
# # SAVE PROCESSED DATA (FOR ML MODEL)
# # ================================
# output_path = os.path.join(os.path.dirname(script_dir), "data", "processed_reviews.csv")
# print(f"\nSaving processed data to: {output_path}")
# df.to_csv(output_path, index=False)
# print("Saved successfully! ✅")




# ============================================
# CUSTOM NLP PIPELINE (WITHOUT SPACY)
# ============================================

import pandas as pd
import re

# ============================================
# LOAD DATA
# ============================================
df = pd.read_csv("../data/cleaned_reviews.csv")

# ============================================
# STEP 1: TEXT CLEANING
# ============================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ============================================
# STEP 2: TOKENIZATION
# ============================================
def tokenize(text):
    return text.split()

# ============================================
# STEP 3: STOPWORD REMOVAL
# ============================================
stopwords = set([
    'the', 'is', 'and', 'in', 'to', 'it', 'of', 'for', 'on', 'this',
    'that', 'was', 'with', 'as', 'but', 'are', 'i', 'you', 'he', 'she', 
    'they', 'we', 'have', 'has', 'had', 'do', 'does', 'did', 'a', 'an', 
    'be', 'been', 'being'
])

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stopwords]

# ============================================
# STEP 4: LEMMATIZATION (IMPROVED RULE-BASED)
# ============================================
def lemmatize(tokens):
    lemmas = []
    for word in tokens:
        if word.endswith("ies") and len(word) > 4:
            word = word[:-3] + "y"
        elif word.endswith("ing") and len(word) > 4:
            word = word[:-3]
        elif word.endswith("ed") and len(word) > 3:
            word = word[:-2]
        elif word.endswith("s") and len(word) > 3:
            word = word[:-1]
        lemmas.append(word)
    return lemmas

# ============================================
# APPLY PIPELINE
# ============================================
def process_text(text):
    # Note: text is already cleaned in cleaned_reviews.csv
    tokens = tokenize(str(text))
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return " ".join(tokens)

df["processed_text"] = df["clean_text"].apply(process_text)

# ============================================
# FILTER BINARY DATA (POSITIVE/NEGATIVE)
# ============================================
df = df[df["sentiment"] != "neutral"]
df = df.dropna(subset=["processed_text"])
df = df[df["processed_text"].str.strip() != ""] # Remove empty strings that become NaNs on reload

# ============================================
# OUTPUT
# ============================================
print(df[["clean_text", "processed_text"]].head())

# ============================================
# SAVE PROCESSED DATA
# ============================================
df.to_csv("../data/processed_reviews.csv", index=False)
print("\nSaved processed data to: ../data/processed_reviews.csv")