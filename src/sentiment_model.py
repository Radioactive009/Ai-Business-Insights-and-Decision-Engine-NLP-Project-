from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd

def train_model(df):
    X = df['clean_text']
    y = df['sentiment']

    # Convert text to numbers
    vectorizer = TfidfVectorizer(max_features=5000)
    X_vec = vectorizer.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    print("\nModel Performance:\n")
    print(classification_report(y_test, y_pred))

    return model, vectorizer


if __name__ == "__main__":
    df = pd.read_csv("data/cleaned_reviews.csv")
    train_model(df)