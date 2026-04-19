import pandas as pd
import spacy

# ================================
# LOAD SPACY MODEL
# ================================
nlp = spacy.load("en_core_web_sm")


# ================================
# LOAD CLEANED CSV DATA
# ================================
df = pd.read_csv("data/cleaned_reviews.csv")


# ================================
# TOKENIZATION + STOPWORD REMOVAL + LEMMATIZATION
# ================================
def process_text(text):
    text = str(text)
    doc = nlp(text)

    tokens = []

    for token in doc:
        # TOKENIZATION:
        # spaCy splits sentence into individual words (tokens)

        # STOPWORD REMOVAL:
        # Remove common words like "the", "is", "and"

        # LEMMATIZATION:
        # Convert word to base form (running → run)

        if not token.is_stop and not token.is_punct:
            tokens.append(token.lemma_)

    return " ".join(tokens)


df["processed_text"] = df["clean_text"].apply(process_text)


# ================================
# POS TAGGING (Morphology)
# ================================
def get_pos_tags(text):
    text = str(text)
    doc = nlp(text)

    # Each token mapped to its grammatical role
    return [(token.text, token.pos_) for token in doc]


df["pos_tags"] = df["clean_text"].apply(get_pos_tags)


# ================================
# NAMED ENTITY RECOGNITION (NER)
# ================================
def get_entities(text):
    text = str(text)
    doc = nlp(text)

    # Extract entities like names, places, dates
    return [(ent.text, ent.label_) for ent in doc.ents]


df["entities"] = df["clean_text"].apply(get_entities)


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