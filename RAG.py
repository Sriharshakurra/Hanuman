# rag.py — Simple Local RAG (Ollama + Chroma + PyPDF)
# Sri Harsha's Hanuman RAG 🚩

import os, uuid, json, time, sys
from typing import List, Dict, Tuple
import urllib.request

import chromadb
from chromadb.config import Settings
from pypdf import PdfReader

# ----------------------- Config -----------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")  # 127.0.0.1 avoids some localhost quirks on Windows
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
LLM_MODEL   = os.getenv("LLM_MODEL", "llama3.1")
DB_DIR      = os.getenv("DB_DIR", "./chroma_db")
DATA_DIR    = os.getenv("DATA_DIR", "./data")
COLLECTION  = os.getenv("COLLECTION", "docs")

HTTP_TIMEOUT = 120  # seconds

# ----------------------- HTTP helper ------------------
def http_post_json(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type":"application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
        return json.loads(r.read().decode())

def ollama_is_alive() -> bool:
    # Simple GET on / for "Ollama is running." without raising
    try:
        with urllib.request.urlopen(OLLAMA_URL + "/", timeout=5) as r:
            txt = r.read().decode(errors="ignore")
            return "Ollama" in txt or "running" in txt
    except Exception:
        return False

# ----------------------- Ollama calls -----------------
def ollama_embeddings(texts: List[str]) -> List[List[float]]:
    # Call embeddings one-by-one (simpler; reliable). You can batch later.
    vectors = []
    for t in texts:
        out = http_post_json(f"{OLLAMA_URL}/api/embeddings", {
            "model": EMBED_MODEL,
            "prompt": t
        })
        vectors.append(out["embedding"])
    return vectors

def ollama_generate(prompt: str, temperature: float = 0.2) -> str:
    data = http_post_json(f"{OLLAMA_URL}/api/generate", {
        "model": LLM_MODEL,
        "prompt": prompt,
        "options": {
            "temperature": temperature,
            "num_predict": 512,
            "num_ctx": 4096
        },
        "stream": False
    })
    resp = (data.get("response") or "").strip()
    return resp if resp else "[No response from model]"



# ----------------------- Loading & chunking -----------
def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_pdf_file(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages)

def load_docs(folder: str) -> List[Tuple[str, str]]:
    docs = []
    for root, _, files in os.walk(folder):
        for fn in files:
            p = os.path.join(root, fn)
            ext = os.path.splitext(fn)[1].lower()
            if ext in (".txt", ".md"):
                text = read_text_file(p)
            elif ext == ".pdf":
                text = read_pdf_file(p)
            else:
                continue
            if text.strip():
                docs.append((p, text))
    return docs

def chunk_text(text: str, size: int = 800, overlap: int = 120) -> List[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = words[i:i+size]
        chunks.append(" ".join(chunk))
        i += max(1, size - overlap)  # guard against non-advance
    return chunks

# ----------------------- Vector store (Chroma) --------
def get_collection():
    client = chromadb.PersistentClient(path=DB_DIR, settings=Settings(allow_reset=False))
    return client.get_or_create_collection(name=COLLECTION, metadata={"hnsw:space":"cosine"})

def ingest():
    if not ollama_is_alive():
        print(f"[!] Ollama not reachable at {OLLAMA_URL}. Start it with `ollama serve` (or Windows service), then retry.")
        return

    col = get_collection()
    docs = load_docs(DATA_DIR)
    if not docs:
        print("No docs found in ./data. Add PDFs/TXTs/MDs and rerun.")
        return

    add_count = 0
    for path, text in docs:
        chunks = chunk_text(text)
        if not chunks:
            continue
        ids    = [str(uuid.uuid4()) for _ in chunks]
        metas  = [{"source": path, "chunk_idx": i} for i in range(len(chunks))]
        embeds = ollama_embeddings(chunks)
        col.add(ids=ids, embeddings=embeds, documents=chunks, metadatas=metas)
        add_count += len(chunks)
        print(f"Ingested {len(chunks)} chunks from: {path}")
    print(f"✅ Total chunks added: {add_count}")

def retrieve(query: str, k: int = 4) -> List[Dict]:
    col = get_collection()
    qv = ollama_embeddings([query])[0]
    res = col.query(query_embeddings=[qv], n_results=max(1, k),
                    include=["documents","metadatas","distances"])
    items = []
    if not res["ids"]:
        return items
    for i in range(len(res["ids"][0])):
        items.append({
            "doc":  res["documents"][0][i],
            "meta": res["metadatas"][0][i],
            "score": 1 - res["distances"][0][i]  # cosine sim approx
        })
    return items

# ----------------------- RAG prompt -------------------
def build_prompt(question: str, contexts: List[Dict]) -> str:
    header = (
        "You are a precise, grounded assistant. Answer using ONLY the context.\n"
        "If the answer is not in the context, say you don’t know.\n"
        "Cite sources with (Source: <filename>, chunk <idx>).\n\n"
    )
    ctx_blocks = []
    for c in contexts:
        src = os.path.basename(c["meta"]["source"])
        idx = c["meta"]["chunk_idx"]
        snippet = c["doc"][:1200]
        ctx_blocks.append(f"[{src} :: chunk {idx}]\n{snippet}\n")
    context_block = "\n---\n".join(ctx_blocks)
    return f"{header}Context:\n{context_block}\n\nQuestion: {question}\nAnswer:"



def ask(question: str, k: int = 4, show_debug: bool = True) -> str:
    if not ollama_is_alive():
        return f"[!] Ollama not reachable at {OLLAMA_URL}. Start it with `ollama serve`."

    ctx = retrieve(question, k=k)
    if not ctx:
        return "[No context retrieved. Try increasing -k or check ingestion.]"

    if show_debug:
        picks = "; ".join(f"{os.path.basename(c['meta']['source'])}#chunk-{c['meta']['chunk_idx']}" for c in ctx)
        print("Context →", picks)

    prompt = build_prompt(question, ctx)
    ans = ollama_generate(prompt)

    cites = [f"{os.path.basename(c['meta']['source'])}#chunk-{c['meta']['chunk_idx']}" for c in ctx]
    return (ans or "[No response from model]").strip() + "\n\nSources: " + ", ".join(sorted(set(cites)))

# ----------------------- CLI -------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Simple RAG (Ollama + Chroma)")
    ap.add_argument("--ingest", action="store_true", help="Ingest files from ./data")
    ap.add_argument("--ask", type=str, help="Ask a question")
    ap.add_argument("-k", type=int, default=4, help="Top-K chunks")
    ap.add_argument("--no-debug", action="store_true", help="Hide picked context lines")
    args = ap.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(DB_DIR, exist_ok=True)

    if args.ingest:
        ingest()
    if args.ask:
        print("\n🕉  Hanuman RAG says:\n")
        print(f"Using model: {LLM_MODEL}")
        print(ask(args.ask, k=args.k, show_debug=not args.no_debug))
