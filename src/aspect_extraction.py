# ============================================
# ASPECT-BASED SENTIMENT ANALYSIS MODULE
# ============================================

# This script:
# 1. Loads preprocessed review data
# 2. Defines business-related aspects
# 3. Detects aspects in each review
# 4. Assigns sentiment to each detected aspect
# 5. Outputs structured insights


# ============================================
# IMPORT LIBRARIES
# ============================================
import pandas as pd


# ============================================
# LOAD DATA
# ============================================
# Make sure path is correct based on your folder structure
df = pd.read_csv("../data/cleaned_reviews.csv")


# ============================================
# DEFINE BUSINESS ASPECTS
# ============================================
# These are categories we want to analyze
# Each aspect has related keywords

aspects = {
    "delivery": ["delivery", "shipping", "late", "arrived", "delay"],
    "price": ["price", "cost", "expensive", "cheap", "value"],
    "quality": ["quality", "product", "build", "material", "durable"],
    "service": ["service", "support", "customer", "help", "response"]
}


# ============================================
# FUNCTION: DETECT ASPECTS IN TEXT
# ============================================
def detect_aspects(text):
    """
    Detects which aspects are mentioned in the review
    using keyword matching.

    Input:
        text (str): review text

    Output:
        list: detected aspects
    """
    text = str(text).lower()
    found_aspects = []

    # Loop through each aspect and its keywords
    for aspect, keywords in aspects.items():
        for word in keywords:
            if word in text:
                found_aspects.append(aspect)
                break  # stop after first match

    return list(set(found_aspects))  # remove duplicates


# ============================================
# FUNCTION: ASSIGN SENTIMENT TO EACH ASPECT
# ============================================
def aspect_sentiment(row):
    """
    Assigns sentiment to each detected aspect.

    Input:
        row: DataFrame row

    Output:
        dict: aspect → sentiment mapping
    """
    aspects_found = detect_aspects(row["clean_text"])

    result = {}

    for aspect in aspects_found:
        result[aspect] = row["sentiment"]

    return result


# ============================================
# APPLY ASPECT-BASED ANALYSIS
# ============================================
df["aspect_sentiment"] = df.apply(aspect_sentiment, axis=1)


# ============================================
# DISPLAY SAMPLE OUTPUT
# ============================================
print("\n===== SAMPLE ASPECT-BASED OUTPUT =====\n")

print(df[[
    "clean_text",
    "sentiment",
    "aspect_sentiment"
]].head())


# ============================================
# OPTIONAL: SUMMARY INSIGHTS
# ============================================
# Count how many times each aspect appears

aspect_counts = {}

for row in df["aspect_sentiment"]:
    for aspect in row:
        aspect_counts[aspect] = aspect_counts.get(aspect, 0) + 1

print("\n===== ASPECT FREQUENCY =====\n")
print(aspect_counts)