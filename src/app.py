import streamlit as st
import pandas as pd
import os
from absa_llm import absa_llm

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="AI Customer Intelligence Engine",
    page_icon="🧠",
    layout="wide"
)

# ============================================
# HELPER FUNCTIONS
# ============================================
def read_code(filename):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error loading file: {e}"

# ============================================
# SIDEBAR NAVIGATION
# ============================================
st.sidebar.title("🚀 Navigation")
page = st.sidebar.radio(
    "Select a Pipeline Stage:",
    ["Overview", "Preprocessing", "Sentiment Analysis", "BERT Model", "Rule-Based ABSA", "LLM-Based ABSA"]
)

st.sidebar.markdown("---")
st.sidebar.info("This engine processes customer reviews to extract deep business insights.")

# ============================================
# SECTION: OVERVIEW
# ============================================
if page == "Overview":
    st.title("🧠 AI Customer Intelligence & Decision Engine")
    st.markdown("### Turning Raw Feedback into Actionable Intelligence")
    
    st.image("https://img.freepik.com/free-vector/digital-technology-background-with-abstract-circuit-board_1017-31053.jpg", use_column_width=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("1. Preprocessing")
        st.write("Custom Tokenization, POS Tagging, and Entity Recognition.")
    with col2:
        st.subheader("2. Sentiment Analysis")
        st.write("Binary classification using Logistic Regression and DistilBERT.")
    with col3:
        st.subheader("3. ABSA")
        st.write("Extracting feelings about specific product features using LLMs.")

    st.markdown("---")
    st.markdown("#### **The Data Flow:**")
    st.success("Raw Review → Preprocessing → Sentiment Classification → Aspect Extraction → Business Insights")

# ============================================
# SECTION: PREPROCESSING
# ============================================
elif page == "Preprocessing":
    st.title("🔍 Data Preprocessing Pipeline")
    st.write("This stage handles tokenization, stopword removal, and POS tagging without using libraries like NLTK or SpaCy.")
    
    with st.expander("View Preprocessing Logic (preprocessing.py)"):
        st.code(read_code("preprocessing.py"), language="python")

# ============================================
# SECTION: SENTIMENT ANALYSIS
# ============================================
elif page == "Sentiment Analysis":
    st.title("📈 Sentiment Analysis (Logistic Regression)")
    st.write("Using a TF-IDF vectorizer and Logistic Regression to classify reviews as Positive or Negative.")
    
    with st.expander("View Model Logic (sentiment_model.py)"):
        st.code(read_code("sentiment_model.py"), language="python")

# ============================================
# SECTION: BERT MODEL
# ============================================
elif page == "BERT Model":
    st.title("🤖 Deep Learning: BERT Integration")
    st.write("Leveraging DistilBERT for state-of-the-art sequence classification.")
    
    # Note: bert_model.py might be in the directory
    with st.expander("View BERT Implementation (bert_model.py)"):
        st.code(read_code("bert_model.py"), language="python")

# ============================================
# SECTION: RULE-BASED ABSA
# ============================================
elif page == "Rule-Based ABSA":
    st.title("📏 Rule-Based Aspect Sentiment")
    st.write("Mapping aspects to sentiments using distance-based window matching and custom lexicons.")
    
    with st.expander("View Rule Logic (absa.py)"):
        st.code(read_code("absa.py"), language="python")

# ============================================
# SECTION: LLM-BASED ABSA
# ============================================
elif page == "LLM-Based ABSA":
    st.title("🔥 LLM-Based Aspect Analysis (Llama3)")
    st.write("Using a local LLM to understand context, multi-word aspects, and complex sentiment patterns.")

    # A. PRELOADED EXAMPLES
    st.subheader("📊 Dataset Samples")
    try:
        df = pd.read_csv("../data/processed_reviews.csv")
        sample_reviews = df.head(3)
        for _, row in sample_reviews.iterrows():
            with st.container():
                st.info(f"**Review:** {row['clean_text'][:150]}...")
                # Note: We won't run LLM on every page load to save performance
                st.write("*(Run interactive analysis below to see LLM output)*")
    except:
        st.warning("Processed dataset not found. Please run preprocessing first.")

    st.markdown("---")

    # B. USER INPUT
    st.subheader("🎯 Interactive Analysis")
    user_input = st.text_area("Enter a customer review to analyze:", placeholder="The camera is great but the battery life is poor.")
    
    if st.button("Analyze Review"):
        if user_input.strip() == "":
            st.warning("Please enter some text first!")
        else:
            with st.spinner("Llama3 is thinking..."):
                result = absa_llm(user_input)
            
            if result:
                st.success("Analysis Complete!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### **Raw JSON Output**")
                    st.json(result)
                
                with col2:
                    st.markdown("#### **Formatted Insights**")
                    for aspect, sentiment in result.items():
                        color = "green" if sentiment.lower() == "positive" else "red"
                        st.markdown(f"**{aspect.title()}** : <span style='color:{color}; font-weight:bold;'>{sentiment.upper()}</span>", unsafe_allow_html=True)
            else:
                st.error("Failed to get analysis. Ensure Ollama is running.")

    st.markdown("---")
    with st.expander("View LLM Module (absa_llm.py)"):
        st.code(read_code("absa_llm.py"), language="python")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.caption("AI Decision Engine Project | Built with Streamlit & Ollama")
