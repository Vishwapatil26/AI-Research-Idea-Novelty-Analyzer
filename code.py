# app.py
import streamlit as st
import os
import re
import json
from datetime import datetime
from dotenv import load_dotenv
from typing import TypedDict, List, Dict, Optional

# --- Imports ---
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langgraph.graph import StateGraph, END
import arxiv
# --- NEW: Import for quantitative scoring ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# ==============================================================================
#  1. DEFINE THE LANGGRAPH AGENT LOGIC
# ==============================================================================

# --- MODIFIED: Added quantitative_novelty to the state ---
class ResearchGraphState(TypedDict):
    user_query: str
    filter_year: int
    search_queries: List[str]
    retrieved_papers: List[Dict]
    papers_to_process: List[Dict]
    quantitative_novelty: float
    novelty_analysis: Optional[Dict]

@tool
def search_arxiv(query: str, max_results: int = 10, filter_year: int = 2022) -> List[Dict]:
    """Searches ArXiv for a query, filtering by publication year."""
    client = arxiv.Client()
    search = arxiv.Search(query=query, max_results=max_results * 2, sort_by=arxiv.SortCriterion.Relevance)
    results = []
    for r in client.results(search):
        if r.published.year >= filter_year:
            results.append({
                "title": r.title,
                "url": r.pdf_url,
                "abstract": r.summary.replace("\n", " "),
                "source": "ArXiv"
            })
        if len(results) >= max_results:
            break
    return results

@st.cache_resource
def get_llm():
    return Ollama(model="mistral:7b-instruct", temperature=0.1)
llm = get_llm()

# --- Graph Nodes ---

def expand_queries_node(state: ResearchGraphState) -> Dict:
    """Uses a Chain-of-Thought prompt to generate high-quality search queries."""
    user_idea = state['user_query']
    prompt = PromptTemplate(
        template="""You are a research assistant. Your goal is to deconstruct a user's idea into its core concepts and then create specific search queries for ArXiv.
        **Step 1: Deconstruct the Idea** - Identify the core problem, technique, and niche keywords.
        **Step 2: Construct Queries** - Create 3 diverse queries using boolean operators and quotes.
        ---
        **EXAMPLE**
        **User's Idea:** "using transformers for time-series forecasting"
        **Your Output:**
        **Analysis:**
        - Core Problem: Time-series forecasting
        - Primary Technique: Transformer models
        - Niche Keywords: "attention mechanism", "long-range dependencies"
        **Queries:**
        "transformer models for non-stationary time-series forecasting", "(self-attention OR attention mechanism) AND financial time series", "long-range dependency forecasting with transformers"
        ---
        **YOUR TASK**
        **User's Idea:** "{idea}"
        **Your Output:**
        """,
        input_variables=["idea"],
    )
    query_generation_chain = prompt | llm | StrOutputParser()
    raw_output = query_generation_chain.invoke({"idea": user_idea})
    queries_section_match = re.search(r'Queries:\s*(.*)', raw_output, re.DOTALL | re.IGNORECASE)
    queries = []
    if queries_section_match:
        queries_text = queries_section_match.group(1)
        queries = re.findall(r'"([^"]+)"', queries_text)
        if not queries:
            queries = [q.strip() for q in queries_text.split(',') if q.strip()]
    queries.append(f'"{user_idea}"')
    valid_queries = [q for q in queries if q and len(q) > 5]
    return {"search_queries": list(set(valid_queries))}

def search_papers_node(state: ResearchGraphState) -> Dict:
    search_queries, filter_year = state['search_queries'], state['filter_year']
    all_papers = []
    for query in search_queries:
        try:
            papers = search_arxiv.invoke({"query": query, "filter_year": filter_year})
            all_papers.extend(papers)
        except Exception as e:
            st.toast(f"Query '{query[:30]}...' failed.", icon="⚠️")
    return {"retrieved_papers": all_papers}

def deduplicate_papers_node(state: ResearchGraphState) -> Dict:
    retrieved_papers = state['retrieved_papers']
    unique_papers, seen_titles = [], set()
    for paper in retrieved_papers:
        title = paper.get('title')
        if not title: continue
        clean_title = re.sub(r'\W+', '', title).lower()
        if clean_title not in seen_titles:
            seen_titles.add(clean_title)
            unique_papers.append(paper)
    return {"papers_to_process": unique_papers}

# --- NEW NODE: Calculates a quantitative novelty score ---
def calculate_quantitative_score_node(state: ResearchGraphState) -> Dict:
    """Calculates a novelty score based on semantic similarity."""
    user_query = state['user_query']
    papers = state['papers_to_process']
    
    if not papers:
        return {"quantitative_novelty": 1.0}

    abstracts = [p['abstract'] for p in papers]
    corpus = [user_query] + abstracts
    
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus)
        # Compare the user query (first doc) to all paper abstracts (the rest)
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
        
        # The highest similarity score indicates the most related paper
        max_similarity = cosine_sim.max()
        # Novelty is the inverse of similarity
        novelty_score = 1 - max_similarity
    except ValueError:
        # Handle cases where vocabulary might be empty (e.g., non-English text)
        novelty_score = 0.5 # Default to a neutral score on error

    return {"quantitative_novelty": novelty_score}

# --- MODIFIED: This node now uses the new prompt and the quantitative score ---
def generate_novelty_analysis_node(state: ResearchGraphState) -> Dict:
    user_query = state['user_query']
    top_papers = state['papers_to_process'][:10]
    quantitative_score = state['quantitative_novelty']
    
    paper_texts = '\n\n'.join([
        f"Paper {i+1}:\nTitle: {p['title']}\nAbstract: {p['abstract']}\n---"
        for i, p in enumerate(top_papers)
    ])

    # Using the new, user-provided prompt template
    prompt_template = """
You are an expert research analyst. Your task is to evaluate the novelty of a user's research idea based on a set of academic papers.

You have been provided a **quantitative novelty score of {quantitative_novelty:.2f}**, calculated using mathematical and semantic similarity measures (where 1.0 = completely novel, and 0.0 = highly similar to existing work).

Your job is to **qualitatively assess and refine this novelty score** using human judgment, taking into account subtleties such as:
- Does the idea propose a **new methodology or approach**?
- Is it applied in a **novel domain** not yet explored in literature?
- Does it **combine existing concepts in a unique or surprising way**?
- Are there **gaps in current research** that the idea tries to address?

**User's Research Idea:**
{idea}

**Retrieved Research Papers (from sources like ArXiv):**
{papers}

---

**Your Output MUST be a single valid JSON object** with the following keys:

- `is_novel`: `"Yes"` or `"No"` — Decide based on the overall qualitative and quantitative assessment.
- `novelty_score`: A float (0.0–1.0) representing your **final adjusted score**.
- `justification`: A short paragraph explaining **how you adjusted the quantitative score**, referencing novelty of methodology, domain, or synthesis.
- `research_summary`: A 2-3 sentence, human-friendly summary of **what the existing papers have already covered** about this topic.
- `similar_works`: A list of strings — **titles of up to 3 papers** that are highly similar to the user's idea.
- `suggestions`: A list of **3–5 clear, actionable ideas** to help the user **refine their idea and make it more novel**.

Format your response strictly as shown below. DO NOT include any explanations or text outside the JSON block.

{{
  "is_novel": "Yes",
  "novelty_score": 0.85,
  "justification": "...",
  "research_summary": "...",
  "similar_works": [
    "Title 1",
    "Title 2"
  ],
  "suggestions": [
    "Improve the idea by ...",
    "Another suggestion ..."
  ]
}}
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["idea", "papers", "quantitative_novelty"]
    )
    chain = prompt | llm | StrOutputParser()
    raw_output = chain.invoke({
        "idea": user_query,
        "papers": paper_texts,
        "quantitative_novelty": quantitative_score
    })
    
    analysis = {}
    try:
        start_index = raw_output.find('{')
        end_index = raw_output.rfind('}')
        if start_index != -1 and end_index != -1:
            json_str = raw_output[start_index : end_index + 1]
            analysis = json.loads(json_str)
        else:
            raise ValueError("No JSON object found in the LLM output.")
    except Exception as e:
        st.error(f"Failed to parse LLM output: {e}")
        analysis = {}
        
    return {"novelty_analysis": analysis}

def should_continue(state: ResearchGraphState) -> str:
    return "continue" if state.get("papers_to_process") else "end"

@st.cache_resource
def get_graph():
    """Compiles the research agent graph with the new scoring step."""
    workflow = StateGraph(ResearchGraphState)
    
    workflow.add_node("Expand Queries", expand_queries_node)
    workflow.add_node("Search ArXiv", search_papers_node)
    workflow.add_node("Filter Results", deduplicate_papers_node)
    workflow.add_node("Calculate Quantitative Score", calculate_quantitative_score_node)
    workflow.add_node("Analyze Novelty", generate_novelty_analysis_node)
    
    workflow.set_entry_point("Expand Queries")
    workflow.add_edge("Expand Queries", "Search ArXiv")
    workflow.add_edge("Search ArXiv", "Filter Results")
    
    # --- MODIFIED: New workflow path ---
    workflow.add_conditional_edges(
        "Filter Results",
        should_continue,
        {
            "continue": "Calculate Quantitative Score",
            "end": END
        }
    )
    workflow.add_edge("Calculate Quantitative Score", "Analyze Novelty")
    workflow.add_edge("Analyze Novelty", END)
    
    return workflow.compile()

app = get_graph()

# ==============================================================================
#  2. DEFINE THE STREAMLIT UI
# ==============================================================================
st.set_page_config(layout="wide", page_title="AI Research Agent")
st.title("🔬 AI Research Analysis Agent")
st.markdown("Enter a research idea to assess its novelty against the ArXiv database.")

with st.sidebar:
    st.header("⚙️ Configuration")
    filter_year = st.number_input("Filter papers published since:", min_value=2010, max_value=datetime.now().year, value=2022)


st.subheader("Enter Your Research Idea or Keywords")
user_idea = st.text_input("Label for text input", label_visibility="collapsed", value="Sentiment analysis using Large Language Models")
st.divider()

if st.button("Analyze Research Landscape on ArXiv", type="primary", use_container_width=True):
    if not user_idea:
        st.error("Please enter your research idea.")
    else:
        with st.spinner("🤖 Agent is running... This may take a moment."):
            inputs = {"user_query": user_idea, "filter_year": filter_year}
            st.session_state.results = app.invoke(inputs, {"recursion_limit": 6}) # Increased limit for extra step

# --- MODIFIED: Completely redesigned report section ---
if 'results' in st.session_state and st.session_state.results:
    results = st.session_state.results
    st.divider()
    st.header("Final Analysis Report")
    
    analysis = results.get('novelty_analysis')
    
    if analysis:
        # --- Main Verdict and Score ---
        is_novel = analysis.get("is_novel", "No") == "Yes"
        if is_novel:
            st.success("### Verdict: Potentially Novel Idea")
        else:
            st.warning("### Verdict: Similar to Existing Work")
        
        score = analysis.get("novelty_score", 0.0)
        st.metric(label="Final Adjusted Novelty Score", value=f"{score:.2f}",
                  help="A score adjusted by the LLM based on its qualitative analysis (1.0 is completely novel).")
        st.divider()

        # --- Detailed Breakdown ---
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Justification")
            st.write(analysis.get("justification", "No justification provided."))
            st.subheader("Suggestions for Improvement")
            suggestions = analysis.get("suggestions", [])
            if suggestions:
                for sug in suggestions:
                    st.markdown(f"- {sug}")
            else:
                st.write("No suggestions provided.")

        with col2:
            st.subheader("Summary of Existing Research")
            st.info(analysis.get("research_summary", "No summary provided."))
            st.subheader("Most Similar Works")
            similar_works = analysis.get("similar_works", [])
            if similar_works:
                for work_title in similar_works:
                    st.markdown(f"- *{work_title}*")
            else:
                st.write("No specific similar papers were highlighted.")

    retrieved_papers = results.get('papers_to_process', [])
    if retrieved_papers:
        st.divider()
        st.subheader(f"📚 Top 10 Retrieved Papers from ArXiv (Published since {filter_year})")
        for paper in retrieved_papers[:10]:
            with st.expander(f"**{paper['title']}** (Source: {paper['source']})"):
                st.markdown(f"**URL/ID:** [{paper['url']}]({paper['url']})")
                st.markdown(f"**Abstract:** {paper['abstract']}")
                
    elif not analysis:
        st.warning("No relevant papers were found on ArXiv for your query, so no analysis could be performed.")
