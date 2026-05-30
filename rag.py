import os
import logging
import sqlite3
from datetime import datetime, timedelta

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import yfinance as yf
from dotenv import load_dotenv
from groq import Groq

from vectorstore import search

load_dotenv()

_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant"
DB_PATH = "finsight.db"


# ---------------------------------------------------------------------------
# Company / ticker helpers
# ---------------------------------------------------------------------------

_TICKER_MAP = {
    "reliance":   "RELIANCE.NS",
    "tcs":        "TCS.NS",
    "infosys":    "INFY.NS",
    "hdfc bank":  "HDFCBANK.NS",
    "hdfc":       "HDFCBANK.NS",
    "icici bank": "ICICIBANK.NS",
    "icici":      "ICICIBANK.NS",
}

_PRICE_KEYWORDS = {"price", "stock", "performance", "return", "trading at"}


def _detect_ticker(query: str) -> str | None:
    q = query.lower()
    # longest match first so "hdfc bank" beats "hdfc"
    for name in sorted(_TICKER_MAP, key=len, reverse=True):
        if name in q:
            return _TICKER_MAP[name]
    return None


def _is_price_query(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in _PRICE_KEYWORDS)


# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------

def answer_question(query: str) -> dict:
    """Return {"answer": str, "meta": dict} where meta describes how the answer was built."""
    chunks = search(query, n=3)

    context_blocks = []
    sources = []
    for i, chunk in enumerate(chunks, 1):
        m = chunk["metadata"]
        context_blocks.append(
            f"[{i}] Company: {m['company']}\n"
            f"    Published: {m['published_at']}\n"
            f"    URL: {m['url']}\n"
            f"    Text: {chunk['text']}"
        )
        if m.get("url"):
            sources.append(m["url"])
    context = "\n\n".join(context_blocks)

    ticker_detected = None
    realtime_fetched = False
    realtime_block = ""

    if _is_price_query(query):
        ticker_detected = _detect_ticker(query)
        if ticker_detected:
            price_data = get_price(ticker_detected)
            returns_data = get_returns(ticker_detected, days=7)
            price_str = (
                f"₹{price_data['price']:,.2f}"
                if price_data.get("price") is not None
                else "unavailable"
            )
            ret_str = (
                f"{returns_data['return_pct']:+.2f}%"
                if returns_data.get("return_pct") is not None
                else "unavailable"
            )
            realtime_block = (
                f"\nReal-time data ({ticker_detected}):\n"
                f"  Current price : {price_str}\n"
                f"  7-day return  : {ret_str}\n"
            )
            realtime_fetched = True

    prompt = f"""You are a financial news analyst. Answer the user's question using ONLY the context provided below.
If the context does not contain enough information, say so honestly.
At the end of your answer, cite every source you used in the format: Source: <URL>

Context:
{context}
{realtime_block}
Question: {query}

Answer:"""

    response = _groq.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=512,
    )
    answer = response.choices[0].message.content.strip()

    return {
        "answer": answer,
        "meta": {
            "realtime_fetched": realtime_fetched,
            "ticker_detected": ticker_detected,
            "chunks_retrieved": len(chunks),
            "sources": sources,
        },
    }


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

def get_price(symbol: str) -> dict:
    """Return the current (latest closing) price for a ticker symbol."""
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="2d")
    if hist.empty:
        return {"symbol": symbol, "price": None, "error": "No data returned"}
    price = round(float(hist["Close"].iloc[-1]), 2)
    return {"symbol": symbol, "price": price, "currency": "INR"}


def get_news(company: str) -> list[dict]:
    """Return the 3 most recent headlines for a company from finsight.db."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT title, url, published_at
        FROM articles
        WHERE company = ?
        ORDER BY published_at DESC
        LIMIT 3
        """,
        (company,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_returns(symbol: str, days: int) -> dict:
    """Return the percentage return for a ticker over the last N calendar days."""
    end = datetime.utcnow()
    start = end - timedelta(days=days + 1)
    hist = yf.Ticker(symbol).history(
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
    )
    if len(hist) < 2:
        return {"symbol": symbol, "days": days, "return_pct": None, "error": "Insufficient data"}
    old = float(hist["Close"].iloc[0])
    new = float(hist["Close"].iloc[-1])
    return_pct = round((new - old) / old * 100, 2)
    return {"symbol": symbol, "days": days, "return_pct": return_pct}


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    query = "What is happening with TCS this week?"
    print(f"Question: {query}\n")
    print("=" * 60)
    result = answer_question(query)
    print(result["answer"])
    print("\nMeta:", result["meta"])
