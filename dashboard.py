import os
import subprocess
import sys
import re

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import streamlit as st

# Inject Streamlit Cloud secrets into env so all modules read them via os.getenv
try:
    for _key in ("NEWSAPI_KEY", "GROQ_API_KEY"):
        if _key in st.secrets and not os.environ.get(_key):
            os.environ[_key] = st.secrets[_key]
except Exception:
    pass  # running locally — .env handled by load_dotenv() inside each module

from ingest import get_stock_data, main as run_ingest
from vectorstore import build_vectorstore
from rag import answer_question


# ---------------------------------------------------------------------------
# Auto-initialise on first deploy (finsight.db and chroma_db won't exist)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def initialize():
    db_missing = not os.path.exists("finsight.db")
    chroma_missing = not os.path.exists("chroma_db") or not os.listdir("chroma_db")

    if db_missing:
        run_ingest()

    if db_missing or chroma_missing:
        build_vectorstore()


with st.spinner("Loading FinSight…"):
    initialize()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FinSight",
    page_icon="📈",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar — company prices
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("FinSight")
    st.caption("AI Financial Research")
    st.divider()

    st.subheader("Tracked Companies")

    with st.spinner("Loading prices..."):
        stock_data = get_stock_data()

    for company, info in stock_data.items():
        price = info.get("price")
        ret = info.get("7d_return_pct")

        price_str = f"₹{price:,.2f}" if price is not None else "N/A"

        if ret is None:
            ret_str = "N/A"
            color = "gray"
        elif ret >= 0:
            ret_str = f"+{ret:.2f}%"
            color = "green"
        else:
            ret_str = f"{ret:.2f}%"
            color = "red"

        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown(f"**{company}**  \n{price_str}")
        with col2:
            st.markdown(
                f"<span style='color:{color}; font-weight:600; font-size:0.95rem'>{ret_str}</span>"
                f"<br><span style='color:gray; font-size:0.75rem'>7d</span>",
                unsafe_allow_html=True,
            )

        st.write("")

    st.divider()

    if st.button("🔄 Refresh Data", use_container_width=True):
        with st.spinner("Running ingest…"):
            result = subprocess.run(
                [sys.executable, "ingest.py"],
                capture_output=True,
                text=True,
            )
        if result.returncode == 0:
            st.success("Data refreshed!")
            st.rerun()
        else:
            st.error("Ingest failed.")
            st.code(result.stderr[-500:])

# ---------------------------------------------------------------------------
# Main area — Q&A
# ---------------------------------------------------------------------------

st.header("Ask anything about Indian markets")
st.write("")

question = st.text_input(
    label="Your question",
    placeholder="e.g. What is happening with Infosys this week?",
    label_visibility="collapsed",
)

ask = st.button("Ask", type="primary")

if ask and question.strip():
    with st.spinner("Thinking…"):
        result = answer_question(question.strip())

    raw_answer = result["answer"]
    meta = result["meta"]

    # Split answer body from sources block emitted by the LLM
    source_pattern = re.compile(
        r"(source[s]?\s*:?\s*\n)(.*)", re.IGNORECASE | re.DOTALL
    )
    match = source_pattern.search(raw_answer)

    if match:
        body = raw_answer[: match.start()].strip()
        sources_block = match.group(2).strip()
    else:
        body = raw_answer.strip()
        sources_block = ""

    st.markdown("### Answer")
    st.markdown(body)

    if sources_block:
        st.markdown("---")
        st.markdown("**Sources**")
        for line in sources_block.splitlines():
            line = line.strip()
            if not line:
                continue
            url_match = re.search(r"(https?://\S+)", line)
            if url_match:
                url = url_match.group(1).rstrip(")")
                label = re.sub(r"^[\d\.\[\]\s]+", "", line).strip()
                label = label if label and not label.startswith("http") else url
                st.markdown(f"- [{label}]({url})")
            else:
                st.markdown(f"- {line}")

    # --- Transparency expander ---
    with st.expander("How this answer was generated"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Real-time price data fetched**")
            if meta["realtime_fetched"]:
                st.success("Yes")
            else:
                st.info("No")

        with col2:
            st.markdown("**Ticker detected**")
            st.markdown(meta["ticker_detected"] if meta["ticker_detected"] else "—")

        st.markdown(f"**News chunks retrieved from vector store:** {meta['chunks_retrieved']}")

        if meta["sources"]:
            st.markdown("**Sources used as context**")
            for i, url in enumerate(meta["sources"], 1):
                st.markdown(f"{i}. [{url}]({url})")

elif ask and not question.strip():
    st.warning("Please enter a question.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption("Knowledge base updated daily via NewsAPI.")
