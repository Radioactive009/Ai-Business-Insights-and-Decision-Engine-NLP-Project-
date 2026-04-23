# AI Customer Intelligence & Decision Engine

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-High--End-red.svg)](https://streamlit.io/)
[![Llama3](https://img.shields.io/badge/LLM-Llama3-blueviolet.svg)](https://ollama.com/)
[![BERT](https://img.shields.io/badge/Model-BERT-orange.svg)](https://huggingface.co/docs/transformers/model_doc/bert)

> **Transforming 50,000+ Raw Reviews into Strategic Business Decisions.**

This repository contains a state-of-the-art **Natural Language Processing (NLP)** intelligence system that automates the analysis of customer feedback. By combining traditional statistics, deep learning, and Large Language Models, it provides a 360° view of brand health and customer satisfaction.

---

## Key Features

- ** Executive Intelligence:** Powered by **Llama3**, it generates 3-point action plans based on real customer pain points.
- ** Hybrid Sentiment Engine:** Uses a dual-model approach:
  - **Logistic Regression** for high-speed statistical classification (~5ms/token).
  - **BERT (Transformer)** for deep contextual understanding of sarcasm and complex emotions.
- ** Granular ABSA:** Aspect-Based Sentiment Analysis that tracks sentiment for specific features like *Battery, Camera, Price, and Screen*.
- ** Professional Preprocessing:** A complete pipeline featuring Tokenization, Stopword Removal, and custom Rule-Based Lemmatization.
- ** Global Trend Analytics:** Aggregates feedback from the entire dataset to generate **Strategic Brand Perception Reports**.

---

##  Tech Stack

- **Frontend:** Streamlit (Custom Dark-Mode UI)
- **NLP Libraries:** Scikit-Learn, Transformers (HuggingFace), Regex
- **Large Language Model:** Llama3 via Ollama
- **Data Engineering:** Pandas, NumPy
- **Visuals:** Graphviz & Custom HTML/CSS

---

##  Architecture Pipeline

1. **Preprocessing:** Clean and structure raw text.
2. **Sentiment Logic:** Classify general "vibe" (Logit vs. BERT).
3. **Feature Extraction:** Identify specific "Aspects" being discussed.
4. **Sentiment Mapping:** Link opinions to features (Rule-Based & LLM).
5. **Strategic Synthesis:** LLM generates business advice and reports.

---

##  Installation & Setup

1. **Clone the Repo:**
   ```bash
   git clone https://github.com/Radioactive009/Ai-Business-Insights-and-Decision-Engine-NLP-Project-.git
   cd Ai-Business-Insights-and-Decision-Engine-NLP-Project-
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Ollama (For LLM Insights):**
   Ensure [Ollama](https://ollama.com/) is installed and Llama3 is running:
   ```bash
   ollama run llama3
   ```

4. **Launch the Dashboard:**
   ```bash
   streamlit run src/app.py
   ```

---

##  Documentation
For a deep dive into the math, logic, and implementation details, check out the [Detailed Project Report](./Detailed_Project_Report.md).

---

##  Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request.

##  License
This project is licensed under the MIT License.

---
**Built with ❤️ for AI Business Intelligence.**
