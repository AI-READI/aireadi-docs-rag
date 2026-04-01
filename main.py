from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

import re
from collections import deque
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

from flask import Flask, request, jsonify
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from flask_cors import CORS
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
import requests


# --- Models pulled via `ollama pull ...`
LLM_MODEL = "llama3.2"            # generation model
EMBED_MODEL = "nomic-embed-text"


## CRAWL AIREADI DOCS
START_URL = "https://docs.aireadi.org/"
MAX_PAGES = 100
MAX_DEPTH = 5

OUT_DIR = Path("data/aireadi_texts")
import shutil
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0 (aireadi-rag)"})
SKIP_EXT = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".zip", ".mp4", ".pdf")

def normalize_url(u: str) -> str:
    return u.split("#", 1)[0].rstrip("/")

def is_aireadi_url(u: str) -> bool:
    if not urlparse(u).netloc.endswith("docs.aireadi.org"):
        return False
    path = urlparse(u).path
    # search only /v3 pages
    if path.startswith("/docs/1/") or path.startswith("/docs/2/"):
        return False
    return True

def should_skip(u: str) -> bool:
    return u.lower().endswith(SKIP_EXT)

def slugify(u: str) -> str:
    p = urlparse(u).path.strip("/") or "home"
    p = re.sub(r"[^a-zA-Z0-9]+", "-", p).strip("-").lower()
    return p[:120] or "page"

def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "iframe", "form"]):
        tag.decompose()
    for sel in ["header", "footer", "nav", "aside"]:
        for t in soup.find_all(sel):
            t.decompose()
    text = soup.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()

# Collect internal URLs (BFS)
visited = set()
queue = deque([(START_URL, 0)])
urls = []

while queue and len(urls) < MAX_PAGES:
    url, depth = queue.popleft()
    url = normalize_url(url)

    if url in visited or depth > MAX_DEPTH:
        continue
    if not is_aireadi_url(url) or should_skip(url):
        continue
    visited.add(url)
    urls.append(url)

    try:
        r = session.get(url, timeout=20)
        if r.status_code != 200:
            continue
        ctype = r.headers.get("content-type", "").lower()
        if "text/html" not in ctype:
            continue

        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.find_all("a", href=True):
            nxt = normalize_url(urljoin(url, a["href"]))
            if is_aireadi_url(nxt) and not should_skip(nxt):
                queue.append((nxt, depth + 1))
    except Exception:
        pass

print(f"Collected {len(urls)} URLs (v3 only)")

# Save each page as .txt
for i, url in enumerate(urls, start=1):
    try:
        r = session.get(url, timeout=20)
        if r.status_code != 200:
            continue
        txt = html_to_text(r.text)
        out = OUT_DIR / f"{i:02d}_{slugify(url)}.txt"
        out.write_text(f"URL: {url}\n\n{txt}\n", encoding="utf-8")
        print(f"[{i:02d}] Saved -> {out.name}")
    except Exception as e:
        print(f" Failed {url}: {e}")

print(f"\n Texts saved in: {OUT_DIR.resolve()}")

print("\n Clinical data pages found:")
for url in urls:
    if "clinical" in url.lower() or "vision" in url.lower() or "monofilament" in url.lower() or "moca" in url.lower() or "questionnaire" in url.lower() or "physical" in url.lower():
        print(f"{url}")

## step 3
TXT_DIR = Path("data/aireadi_texts")
paths = sorted(TXT_DIR.glob("*.txt"))
if not paths:
    raise FileNotFoundError(f"No .txt files found in {TXT_DIR}. Run Step 3 first.")
docs = []
for p in paths:
    d = TextLoader(str(p), encoding="utf-8").load()[0]
    d.metadata["source_file"] = p.name
    docs.append(d)

print(f"Loaded {len(docs)} documents")

# Small chunks
splitter_small = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=150, separators=["\n\n", "\n", " ", ""]
)
chunks_small = splitter_small.split_documents(docs)
print(f"Small chunks: {len(chunks_small)}")
# Large chunks
splitter_large = RecursiveCharacterTextSplitter(
    chunk_size=2000, chunk_overlap=300, separators=["\n\n", "\n", " ", ""]
)
chunks_large = splitter_large.split_documents(docs)
print(f"Large chunks: {len(chunks_large)}")


def make_retriever(vs):
    return vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8}
    )

def answer_with_rag(question: str, app) -> str:
    def invoke(retriever):
        retrieved = retriever.invoke(question)

        context = "\n\n---\n\n".join(
            f"[Source: {d.metadata.get('source_file', '?')}]\n{d.page_content}"
            for d in retrieved
        )
        prompt = f"""You are answering questions about the AI-READI dataset using documentation.
            Read the context carefully and answer the question. When you find something, only answer the direct answer, do not say "According to the documentation,"
            If you are not confident the context contains the correct answer, say: "Not found in the provided pages."

            Documentation context:
            {context}

            Question:
            {question}

            Answer:
        """
        return app.llm.invoke(prompt).content

    answer = invoke(app.vectorstore_small)
    if "not found" in answer.lower():
        answer = invoke(app.vectorstore_large)
    return answer


def register_routes(app):

    @app.route("/chat", methods=["POST"])
    def chat():
        question = request.json["question"]
        answer = answer_with_rag(question, app)
        return jsonify({"answer": answer})



def create_app():
    app = Flask(__name__)

    CORS(app)

    # --- Models (app context içine koy)
    app.llm = ChatOllama(model=LLM_MODEL, temperature=0.2)
    app.embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    #  Build two FAISS indexes
    app.vectorstore_small = FAISS.from_documents(chunks_small, app.embeddings)
    app.vectorstore_small.save_local("faiss_index_small")
    print("Small index saved")

    vectorstore_large = FAISS.from_documents(chunks_large, app.embeddings)
    vectorstore_large.save_local("faiss_index_large")
    print("Large index saved")

    app.retriever_small = make_retriever(app.vectorstore_small)
    app.retriever_large = make_retriever(app.vectorstore_large)

    print("Ollama models configured:", LLM_MODEL, "|", EMBED_MODEL)

    # --- routes
    register_routes(app)

    return app