import sqlite3
import chromadb
from sentence_transformers import SentenceTransformer

DB_PATH = "finsight.db"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "financial_news"
EMBED_MODEL = "all-MiniLM-L6-v2"


def _load_articles(db_path: str) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, company, title, description, url, published_at FROM articles"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _build_chunk(article: dict) -> str:
    title = article.get("title") or ""
    description = article.get("description") or ""
    return f"{title}. {description}".strip(". ")


def build_vectorstore() -> tuple[chromadb.Collection, SentenceTransformer]:
    model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    articles = _load_articles(DB_PATH)

    existing_ids = set(collection.get(include=[])["ids"])
    new_articles = [a for a in articles if str(a["id"]) not in existing_ids]

    if not new_articles:
        print("Vectorstore is up to date — no new articles to embed.")
        return collection, model

    texts = [_build_chunk(a) for a in new_articles]
    ids = [str(a["id"]) for a in new_articles]
    metadatas = [
        {
            "company": a["company"],
            "published_at": a["published_at"] or "",
            "url": a["url"] or "",
            "title": a["title"] or "",
        }
        for a in new_articles
    ]

    print(f"Embedding {len(new_articles)} articles with {EMBED_MODEL}...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64).tolist()

    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    print(f"Stored {len(new_articles)} chunks in collection '{COLLECTION_NAME}'.")

    return collection, model


def search(query: str, n: int = 3) -> list[dict]:
    model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)

    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({"text": doc, "metadata": meta, "distance": round(dist, 4)})
    return hits


if __name__ == "__main__":
    build_vectorstore()

    print("\n--- Search: 'Reliance quarterly results' ---\n")
    hits = search("Reliance quarterly results", n=3)
    for i, hit in enumerate(hits, 1):
        m = hit["metadata"]
        print(f"[{i}] distance={hit['distance']}")
        print(f"    company    : {m['company']}")
        print(f"    title      : {m['title']}")
        print(f"    published  : {m['published_at']}")
        print(f"    url        : {m['url']}")
        print(f"    text       : {hit['text'][:200]}")
        print()
