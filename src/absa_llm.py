# ============================================
# LLM-BASED ABSA USING OLLAMA (ROBUST)
# ============================================

import ollama
import json
import re

# ============================================
# FUNCTION: LLM ABSA
# ============================================
def absa_llm(review, model_name="llama3:latest", retry=True):
    """
    Performs Aspect-Based Sentiment Analysis using Ollama LLM.
    Uses Llama3 by default for better instruction following.
    Includes robust JSON extraction and retry logic.
    """

    system_prompt = (
        "You are a strict NLP analyzer. Your ONLY output must be a valid JSON dictionary. "
        "Do NOT include any introduction, explanation, or code blocks. "
        "Only map aspects to 'positive' or 'negative'."
    )
    
    user_prompt = f"""
Analyze the following review for aspects and sentiment.

Review: "{review}"

Format Example:
{{"camera": "negative", "screen": "positive"}}

Constraint: Output ONLY the JSON dictionary.
JSON:"""

    import time
    
    # 2-Attempt Loop for Network Stability
    for attempt in range(2):
        try:
            response = ollama.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                options={"temperature": 0.1}
            )

            # Get raw response and clean common markdown artifacts
            raw_output = response["message"]["content"].strip()
            
            # Remove markdown code blocks if present (```json ... ```)
            clean_output = re.sub(r'```json\s*|\s*```', '', raw_output)

            # ROBUST JSON EXTRACTION: Find the first { and last }
            json_match = re.search(r'\{.*\}', clean_output, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    try:
                        # Replace common invalid single quotes with double quotes
                        json_str_fixed = json_str.replace("'", '"')
                        return json.loads(json_str_fixed)
                    except:
                        pass
            
            # If parsing failed but connection was okay, break loop and handle below
            break

        except Exception as e:
            if attempt == 0:
                print(f"🔄 Network glitch (500), retrying in 2 seconds...")
                time.sleep(2)
                continue
            else:
                print(f"❌ Connection Error after retry: {e}")
                return {}

    # Final fallback if parsing failed
    print(f"⚠️ Warning: Model {model_name} returned invalid output.")
    return {}


# ============================================
# TEST FUNCTION
# ============================================
def run_tests():
    print("\n===== ROBUST LLM ABSA TESTS (LLAMA3) =====\n")

    test_reviews = [
        "The camera is blurry but the screen is amazing",
        "This book has great characters but weak plot",
        "Battery life is poor but design is excellent",
        "The story is good but writing is bad"
    ]

    for review in test_reviews:
        print("Review:", review)
        result = absa_llm(review) # Uses llama3 by default
        print("Output:", result)
        print("-" * 50)


# ============================================
# APPLY TO DATASET (OPTIONAL)
# ============================================
def run_on_dataset():
    import pandas as pd

    print("\n===== RUNNING ON DATASET (SAMPLED) =====\n")

    try:
        df = pd.read_csv("../data/processed_reviews.csv")
        
        # Apply to a small sample for verification
        df_sample = df.head(5).copy()
        print("Processing first 5 reviews...")
        
        df_sample["llm_absa"] = df_sample["clean_text"].apply(lambda x: absa_llm(x))
        
        print("\nResults:")
        print(df_sample[["clean_text", "llm_absa"]])
        
    except FileNotFoundError:
        print("Error: processed_reviews.csv not found.")


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    # STEP 1: Run basic tests
    run_tests()

    # STEP 2: Run on dataset sample
    run_on_dataset()