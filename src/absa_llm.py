# ============================================
# LLM-BASED ABSA USING OLLAMA (PHI / LLAMA3)
# ============================================

import ollama
import json

# ============================================
# FUNCTION: LLM ABSA
# ============================================
def absa_llm(review, model_name="phi:latest"):
    """
    Performs Aspect-Based Sentiment Analysis using Ollama LLM.
    Includes robust JSON extraction to handle 'chatty' models.
    """

    system_prompt = "You are a specialized NLP tool. Your ONLY job is to output raw JSON. Never explain yourself. Never write code. Only output the dictionary."
    
    user_prompt = f"""
Analyze this review and return a JSON object mapping aspects to sentiment (positive/negative).

Review: "{review}"

Return format:
{{"aspect": "sentiment"}}

Example:
{{"camera": "negative", "screen": "positive"}}
"""

    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options={"temperature": 0} # Set temperature to 0 for more deterministic output
        )

        output = response["message"]["content"].strip()

        # ROBUST JSON EXTRACTION
        # Find the first '{' and last '}' in case the model added extra text
        import re
        json_match = re.search(r'\{.*\}', output, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(0)
            try:
                return json.loads(json_str)
            except:
                pass
        
        print(f"⚠️ Failed to parse output from {model_name}: {output[:100]}...")
        return {}

    except Exception as e:
        print("❌ Error:", e)
        return {}


# ============================================
# TEST FUNCTION
# ============================================
def run_tests():
    print("\n===== LLM ABSA TESTS =====\n")

    test_reviews = [
        "The camera is blurry but the screen is amazing",
        "This book has great characters but weak plot",
        "Battery life is poor but design is excellent",
        "The story is good but writing is bad"
    ]

    for review in test_reviews:
        print("Review:", review)
        result = absa_llm(review, model_name="phi:latest")  # change to llama3:latest if needed
        print("Output:", result)
        print("-" * 50)


# ============================================
# APPLY TO DATASET (OPTIONAL)
# ============================================
def run_on_dataset():
    import pandas as pd

    print("\n===== RUNNING ON DATASET =====\n")

    df = pd.read_csv("../data/processed_reviews.csv")

    # Apply LLM ABSA to first few rows (avoid running on full dataset initially)
    df["llm_absa"] = df["clean_text"].head(10).apply(
        lambda x: absa_llm(x, model_name="phi:latest")
    )

    print(df[["clean_text", "llm_absa"]].head())


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":

    # STEP 1: Run basic tests
    run_tests()

    # STEP 2 (optional): Run on dataset
    run_on_dataset()