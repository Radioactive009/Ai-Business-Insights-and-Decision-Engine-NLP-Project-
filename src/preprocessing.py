import pandas as pd
import re
import spacy
import json

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
# LOAD DATA
# ================================
data = []

with open("Electronics.json", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 20000:  # limit for faster processing
            break
        
        review = json.loads(line)
        
        data.append({
            "review_text": review.get("reviewText"),
            "rating": review.get("overall"),
        })

df = pd.DataFrame(data)


# ================================
# TEXT CLEANING
# ================================
def clean_text(text):
    text = str(text).lower()  # convert to lowercase
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove special characters
    text = re.sub(r"\s+", " ", text)  # remove extra spaces
    return text.strip()

df["clean_text"] = df["review_text"].apply(clean_text)


# ================================
# TOKENIZATION + STOPWORD REMOVAL + LEMMATIZATION
# ================================
def process_text(text):
    doc = nlp(text)

    tokens = []
    
    for token in doc:
        # Tokenization → spaCy automatically splits words
        # Stopword Removal → skip common words
        # Lemmatization → convert to base form
        
        if not token.is_stop and not token.is_punct:
            tokens.append(token.lemma_)

    return " ".join(tokens)

df["processed_text"] = df["clean_text"].apply(process_text)


# ================================
# POS TAGGING (Morphology)
# ================================
def get_pos_tags(text):
    doc = nlp(text)
    
    # Returns (word, POS)
    return [(token.text, token.pos_) for token in doc]

df["pos_tags"] = df["clean_text"].apply(get_pos_tags)


# ================================
# NAMED ENTITY RECOGNITION (NER)
# ================================
def get_entities(text):
    doc = nlp(text)
    
    # Returns (entity, label)
    return [(ent.text, ent.label_) for ent in doc.ents]

df["entities"] = df["clean_text"].apply(get_entities)


# ================================
# DISPLAY RESULTS
# ================================
print("\n===== SAMPLE OUTPUT =====\n")

print(df[[
    "review_text",
    "processed_text",
    "pos_tags",
    "entities"
]].head())