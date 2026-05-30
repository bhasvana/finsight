# FinSight — AI Financial Research Assistant

> Ask anything about Indian markets. Get answers grounded in real news and live stock data.

**Live app → [finsight-ai-agent.streamlit.app](https://finsight-ai-agent.streamlit.app)**

---

## What it does

FinSight is a RAG (Retrieval-Augmented Generation) application that answers questions about 5 major Indian stocks — Reliance, TCS, Infosys, HDFC Bank, and ICICI Bank.

Instead of relying on an LLM's training memory (which goes stale), it:
1. Fetches the latest news articles every day via NewsAPI
2. Converts every article into a mathematical embedding (a vector) using a local sentence-transformer model
3. When you ask a question, finds the most semantically relevant articles
4. For price-related questions, also fetches live stock data from NSE via yfinance
5. Hands everything to a Groq-hosted LLaMA model to write a clean, cited answer

The LLM never invents facts — it only summarises what was retrieved.

---

## Demo

![FinSight Demo](https://finsight-ai-agent.streamlit.app)

**Example questions to try:**
- *What is happening with TCS this week?*
- *What is Reliance trading at and how has its performance been?*
- *What is Infosys doing in AI?*
- *How has HDFC Bank stock performed?*

---

## Architecture

```
NewsAPI (7 days)          yfinance (live)
      │                        │
      ▼                        │
  finsight.db (SQLite)         │
      │                        │
      ▼                        │
  all-MiniLM-L6-v2             │
  (SentenceTransformer)        │
      │                        │
      ▼                        │
  ChromaDB (chroma_db/)        │
      │                        │
      └──────────┬─────────────┘
                 │
                 ▼
        Groq API — llama-3.1-8b-instant
                 │
                 ▼
          Streamlit UI
```

| File | Responsibility |
|---|---|
| `ingest.py` | Fetches news via NewsAPI, stores in SQLite, exposes `get_stock_data()` |
| `vectorstore.py` | Embeds articles with `all-MiniLM-L6-v2`, stores in ChromaDB, exposes `search()` |
| `rag.py` | Builds prompt from retrieved chunks + live prices, calls Groq API |
| `dashboard.py` | Streamlit UI — sidebar prices, Q&A interface, answer transparency |

---

## Tech Stack

| Layer | Technology |
|---|---|
| News data | [NewsAPI](https://newsapi.org) |
| Stock data | [yfinance](https://github.com/ranaroussi/yfinance) (Yahoo Finance) |
| Embeddings | `all-MiniLM-L6-v2` via [sentence-transformers](https://www.sbert.net) (runs locally) |
| Vector store | [ChromaDB](https://www.trychroma.com) |
| LLM inference | [Groq](https://groq.com) — `llama-3.1-8b-instant` |
| UI | [Streamlit](https://streamlit.io) |
| Database | SQLite |

---

## Running locally

**1. Clone the repo**
```bash
git clone https://github.com/bhasvana/finsight.git
cd finsight
```

**2. Create a virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your API keys**

Create a `.env` file in the project root:
```
NEWSAPI_KEY=your_newsapi_key_here
GROQ_API_KEY=your_groq_api_key_here
```

Get your keys free at:
- NewsAPI → [newsapi.org](https://newsapi.org/register)
- Groq → [console.groq.com](https://console.groq.com)

**5. Ingest news and build the vector store**
```bash
python ingest.py
python vectorstore.py
```

**6. Launch the dashboard**
```bash
streamlit run dashboard.py
```

Open [http://localhost:8501](http://localhost:8501)

---

## How RAG works

Traditional LLMs answer from their training data — which has a knowledge cutoff and can hallucinate. RAG fixes this:

```
User question
     │
     ▼
Embed question → 384-number vector
     │
     ▼
Find 3 most similar article vectors in ChromaDB  (cosine similarity)
     │
     ▼
Build prompt:  "Answer using ONLY this context: [articles] + [live price]"
     │
     ▼
LLM reads the prompt and writes a cited answer
     │
     ▼
Display answer + sources + transparency panel
```

The model only summarises what you give it. No hallucination.

---

## Project structure

```
finsight/
├── ingest.py          # Data pipeline — news + stock prices
├── vectorstore.py     # Embedding + semantic search
├── rag.py             # RAG pipeline + Groq integration
├── dashboard.py       # Streamlit UI
├── requirements.txt   # Python dependencies
├── .python-version    # Pins Python 3.11
├── .gitignore
└── .streamlit/
    └── secrets.toml.example   # Secret keys format reference
```

---

## Tracked companies

| Company | NSE Ticker |
|---|---|
| Reliance Industries | RELIANCE.NS |
| Tata Consultancy Services | TCS.NS |
| Infosys | INFY.NS |
| HDFC Bank | HDFCBANK.NS |
| ICICI Bank | ICICIBANK.NS |
