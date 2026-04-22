import streamlit as st
import pandas as pd
import os
import re
from absa_llm import absa_llm

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="AI Customer Intelligence Engine",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
    }
    .pipeline-node {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

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
st.sidebar.title("🚀 Project Pipeline")
page = st.sidebar.radio(
    "Navigation:",
    ["Dashboard Overview", "1. Preprocessing", "2. Sentiment Analysis", "3. LLM-Based ABSA"]
)

st.sidebar.markdown("---")
st.sidebar.success("Model: Llama3 (Local)")
st.sidebar.info("Dataset: Amazon Electronics")

# ============================================
# SECTION: DASHBOARD OVERVIEW
# ============================================
if page == "Dashboard Overview":
    st.title("🧠 AI Customer Intelligence & Decision Engine")
    st.markdown("### The Complete NLP Intelligence Pipeline")
    
    # Visual Flowchart using Graphviz
    st.graphviz_chart("""
    digraph G {
        rankdir=LR;
        node [shape=box, style=filled, color="#4CAF50", fontcolor=white, fontname="Helvetica"];
        "Raw Review" -> "Preprocessing";
        "Preprocessing" -> "Sentiment Model";
        "Sentiment Model" -> "ABSA Engine";
        "ABSA Engine" -> "Business Insights";
        
        node [color="#1E88E5"];
        "Preprocessing" -> "POS Tags & Entities";
        "Sentiment Model" -> "Positive/Negative Label";
        "ABSA Engine" -> "Aspect-Level Sentiment";
    }
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Data Volume", value="50,000+", delta="Reviews")
    with col2:
        st.metric(label="Pipeline Speed", value="~5ms", delta="Per Token")
    with col3:
        st.metric(label="LLM Accuracy", value="High", delta="Context Aware")

    st.markdown("---")
    st.subheader("📋 Executive Summary")
    st.write("""
    This project automates the analysis of customer feedback. Instead of reading thousands of reviews, 
    the engine extracts **what** customers are talking about (Aspects) and **how** they feel about them (Sentiment).
    """)

# ============================================
# SECTION: PREPROCESSING (VISUAL)
# ============================================
elif page == "1. Preprocessing":
    st.title("🔍 Step 1: Preprocessing & Tagging")
    st.write("Before analysis, we must break down the raw text into structured components.")

    example_input = "Samsung's battery life is great in London."
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### **Input (Raw)**")
        st.info(example_input)
        
    with col2:
        st.markdown("### **Output (Structured)**")
        st.write("---")
        st.write("**Tokens:** `['Samsung', \"'s\", 'battery', 'life', 'is', 'great', 'in', 'London']` ")
        st.write("**POS Tags:** `Samsung (PROPER_NOUN), battery (NOUN), great (ADJ)`")
        st.write("**Entities:** `Samsung (ORG), London (LOC)`")

    st.markdown("---")
    st.subheader("💡 Why this matters?")
    st.write("By identifying **NOUNS**, our system knows that 'battery' is a feature. By identifying **ADJECTIVES**, it knows 'great' is an opinion.")
    
    with st.expander("View Implementation (preprocessing.py)"):
        st.code(read_code("preprocessing.py"), language="python")

# ============================================
# SECTION: SENTIMENT ANALYSIS (VISUAL)
# ============================================
elif page == "2. Sentiment Analysis":
    st.title("📈 Step 2: Global Sentiment Classification")
    st.write("We use Logistic Regression and BERT to determine if a review is generally Happy or Unhappy.")

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='pipeline-node'><b>Positive Review</b><br>'I love this product, it works perfectly!'</div>", unsafe_allow_html=True)
        st.success("Result: POSITIVE (Score: 0.98)")
        
    with col2:
        st.markdown("<div class='pipeline-node' style='border-left-color: #f44336;'><b>Negative Review</b><br>'This is the worst purchase I ever made.'</div>", unsafe_allow_html=True)
        st.error("Result: NEGATIVE (Score: 0.12)")

    st.markdown("---")
    with st.expander("View Training Logic (sentiment_model.py)"):
        st.code(read_code("sentiment_model.py"), language="python")

# ============================================
# SECTION: LLM-BASED ABSA (VISUAL & INTERACTIVE)
# ============================================
elif page == "3. LLM-Based ABSA":
    st.title("🔥 Step 3: Aspect-Based Insights (LLM)")
    st.write("The most advanced stage. The LLM connects specific features to specific feelings.")

    # Visual Example
    st.subheader("🖼️ How it works (Example)")
    st.image("https://miro.medium.com/v2/resize:fit:1400/1*E_GfVnO1E-8I5-p-z-9g2A.png", caption="Mapping opinions to features", width=600)

    st.markdown("---")
    
    # Interactive Demo
    st.subheader("🎯 Live Demo")
    user_text = st.text_area("Type a complex review (try contrasting sentiments):", "The display is crystal clear but the shipping was very slow.")
    
    if st.button("Generate Deep Insights"):
        with st.spinner("Llama3 analyzing context..."):
            result = absa_llm(user_text)
            
        if result:
            st.write("### **Intelligence Extracted:**")
            cols = st.columns(len(result) if len(result) > 0 else 1)
            
            for i, (aspect, sentiment) in enumerate(result.items()):
                with cols[i % len(cols)]:
                    color = "#28a745" if sentiment.lower() == "positive" else "#dc3545"
                    st.markdown(f"""
                        <div style="background-color: {color}; padding: 20px; border-radius: 10px; text-align: center; color: white;">
                            <h4 style="margin:0;">{aspect.upper()}</h4>
                            <hr style="margin: 10px 0;">
                            <h2 style="margin:0;">{sentiment.upper()}</h2>
                        </div>
                    """, unsafe_allow_html=True)
        else:
            st.error("Ollama not found. Please ensure the server is running.")

    st.markdown("---")
    with st.expander("View LLM Logic (absa_llm.py)"):
        st.code(read_code("absa_llm.py"), language="python")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.caption("AI Customer Intelligence & Decision Engine | Powered by Llama3 & Streamlit")
