
# 🔬 AI Research Idea Novelty Analyzer

This project analyzes the **novelty of a research idea** by expanding it into relevant search queries, retrieving related papers from **ArXiv**, and evaluating semantic similarity using **LLM (Mistral via Ollama)** and **TF-IDF scoring**.

It provides a quantitative and qualitative **novelty score**, a research summary, similar works, and actionable suggestions — all through a clean **Streamlit UI**.

---

## 🚀 Features

- 🧠 **LLM-powered Prompt Expansion**  
  Converts your research idea into precise, diverse ArXiv search queries using prompt engineering with Mistral.

- 🔍 **Semantic Paper Retrieval**  
  Retrieves relevant papers from ArXiv based on generated queries and filters them by publication year.

- 📊 **Quantitative Novelty Scoring**  
  Uses **TF-IDF** and **cosine similarity** to compute how unique your idea is compared to existing research.

- 💬 **Qualitative LLM-based Analysis**  
  Generates a final adjusted novelty score and summary with justification, similar works, and improvement tips.

- 💻 **Streamlit UI**  
  Intuitive interface to input ideas, tweak filters, and view full analysis reports in a user-friendly layout.

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) – Web app UI  
- [LangGraph](https://github.com/langchain-ai/langgraph) – Agent flow orchestration  
- [Ollama + Mistral](https://ollama.com/) – Local LLM reasoning & generation  
- [ArXiv API](https://arxiv.org/help/api/index) – Paper retrieval  
- [scikit-learn](https://scikit-learn.org/) – TF-IDF & cosine similarity for scoring  
- [LangChain](https://www.langchain.com/) – LLM tooling and prompt management  
- [dotenv](https://pypi.org/project/python-dotenv/) – Environment variable loading  

---

## 📦 Setup

1. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Ollama locally**  
   Make sure Mistral is pulled:  
   ```bash
   ollama run mistral
   ```

3. **Start the app**  
   ```bash
   streamlit run app.py
   ```

---
