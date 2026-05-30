import os
import sqlite3
import requests
import yfinance as yf
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
DB_PATH = "finsight.db"

COMPANIES = {
    "Reliance": "Reliance Industries",
    "TCS": "Tata Consultancy Services",
    "Infosys": "Infosys",
    "HDFC Bank": "HDFC Bank",
    "ICICI Bank": "ICICI Bank",
}

TICKERS = {
    "Reliance": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
}


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            company      TEXT NOT NULL,
            title        TEXT,
            description  TEXT,
            url          TEXT UNIQUE,
            published_at TEXT
        )
    """)
    conn.commit()


def fetch_news(company_label: str, query: str) -> list[dict]:
    from_date = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")
    resp = requests.get(
        "https://newsapi.org/v2/everything",
        params={
            "q": query,
            "from": from_date,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 100,
            "apiKey": NEWSAPI_KEY,
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "ok":
        print(f"  [WARN] NewsAPI error for {company_label}: {data.get('message')}")
        return []
    return data.get("articles", [])


def store_articles(conn: sqlite3.Connection, company_label: str, articles: list[dict]) -> int:
    inserted = 0
    for art in articles:
        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO articles (company, title, description, url, published_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    company_label,
                    art.get("title"),
                    art.get("description"),
                    art.get("url"),
                    art.get("publishedAt"),
                ),
            )
            if conn.execute("SELECT changes()").fetchone()[0]:
                inserted += 1
        except sqlite3.Error as e:
            print(f"  [WARN] DB insert failed: {e}")
    conn.commit()
    return inserted


def get_stock_data() -> dict[str, dict]:
    results = {}
    end = datetime.utcnow()
    start = end - timedelta(days=8)

    for label, ticker_sym in TICKERS.items():
        ticker = yf.Ticker(ticker_sym)
        hist = ticker.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))

        if hist.empty:
            results[label] = {"ticker": ticker_sym, "price": None, "7d_return_pct": None}
            continue

        current_price = round(float(hist["Close"].iloc[-1]), 2)
        oldest_price = float(hist["Close"].iloc[0])
        seven_day_return = round((current_price - oldest_price) / oldest_price * 100, 2)

        results[label] = {
            "ticker": ticker_sym,
            "price": current_price,
            "7d_return_pct": seven_day_return,
        }

    return results


def main() -> None:
    if not NEWSAPI_KEY:
        raise EnvironmentError("NEWSAPI_KEY not found in environment / .env file")

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    print("=== Fetching news articles ===")
    total = 0
    for label, query in COMPANIES.items():
        articles = fetch_news(label, query)
        count = store_articles(conn, label, articles)
        print(f"  {label:12s}: {count:3d} new articles stored  (API returned {len(articles)})")
        total += count
    print(f"\n  Total new articles stored: {total}")

    conn.close()

    print("\n=== Stock data ===")
    stock_data = get_stock_data()
    print(f"  {'Company':<12}  {'Ticker':<14}  {'Price (INR)':>12}  {'7d Return':>10}")
    print("  " + "-" * 55)
    for label, info in stock_data.items():
        price = f"{info['price']:,.2f}" if info["price"] is not None else "N/A"
        ret = f"{info['7d_return_pct']:+.2f}%" if info["7d_return_pct"] is not None else "N/A"
        print(f"  {label:<12}  {info['ticker']:<14}  {price:>12}  {ret:>10}")


if __name__ == "__main__":
    main()
