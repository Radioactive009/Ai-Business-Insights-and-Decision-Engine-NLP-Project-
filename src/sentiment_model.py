import pandas as pd

# ================================
# IMPORT ML LIBRARIES
# ================================
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# ================================
# LOAD DATA
# ================================
df = pd.read_csv("../data/processed_reviews.csv")

# Use processed_text (VERY IMPORTANT)
X = df["processed_text"]
y = df["sentiment"]


# ================================
# TEXT VECTORIZATION (TF-IDF)
# ================================
# Converts text → numerical representation
vectorizer = TfidfVectorizer(max_features=5000)

X_vectorized = vectorizer.fit_transform(X)


# ================================
# TRAIN-TEST SPLIT
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)


# ================================
# MODEL TRAINING (LOGISTIC REGRESSION)
# ================================
model = LogisticRegression(max_iter=200)

model.fit(X_train, y_train)


# ================================
# MODEL PREDICTION
# ================================
y_pred = model.predict(X_test)


# ================================
# EVALUATION
# ================================
print("\n===== MODEL PERFORMANCE =====\n")
print(classification_report(y_test, y_pred))