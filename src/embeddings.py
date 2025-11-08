# src/embeddings.py
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import FAISS, Chroma
from sentence_transformers import SentenceTransformer
from src.ingest import ingest_all
from pathlib import Path
import pickle

PERSIST_DIR = os.getenv("FAISS_DIR", "faiss_index")
VECTORSTORE_TYPE = os.getenv("VECTORSTORE_TYPE", "faiss").lower()
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

def build_vectorstore(persist: bool = True):
    print(f"🔹 Using embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)
    docs = ingest_all()

    texts = [d.page_content for d in docs]
    metas = [d.metadata for d in docs]
    embeddings = model.encode(texts, show_progress_bar=True)

    if VECTORSTORE_TYPE == "faiss":
        # ✅ fix: zip texts and embeddings into pairs
        text_embedding_pairs = list(zip(texts, embeddings))
        vs = FAISS.from_embeddings(text_embedding_pairs, model, metadatas=metas)

        if persist:
            Path(PERSIST_DIR).mkdir(parents=True, exist_ok=True)
            vs.save_local(PERSIST_DIR)
            with open(Path(PERSIST_DIR) / "meta.pkl", "wb") as f:
                pickle.dump({"embed_model": EMBED_MODEL}, f)
        print("✅ FAISS vectorstore built and persisted.")
        return vs

    else:
        # Optional: use Chroma instead
        vs = Chroma.from_texts(texts, embedding_function=model, metadatas=metas, persist_directory=PERSIST_DIR)
        if persist:
            vs.persist()
        print("✅ Chroma vectorstore built and persisted.")
        return vs

if __name__ == "__main__":
    build_vectorstore(persist=True)
