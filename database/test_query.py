"""Query the ChromaDB with a user-supplied problem; print top-k matches.

Runs fully offline: the model is loaded from the local ModelScope cache
(~/.cache/modelscope/models/Qwen--Qwen3-Embedding-0.6B). No HuggingFace access.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from sentence_transformers import SentenceTransformer
import chromadb

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
DB_DIR = Path(r"E:/test/AI-MO/database/v2")
COLLECTION = "olympiad_problems"
TOP_K = 5
DEVICE = "cpu"  # set "cuda" if no build is running concurrently


def resolve_model_dir() -> Path:
    try:
        from modelscope import snapshot_download
        return Path(snapshot_download(MODEL_ID))
    except Exception:
        for cand in (
            Path.home() / ".cache/modelscope/models/Qwen--Qwen3-Embedding-0.6B/snapshots/master",
            Path.home() / ".cache/modelscope/hub/Qwen/Qwen3-Embedding-0.6B",
        ):
            if (cand / "config.json").exists():
                return cand
        raise


def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = (
            "Prove that for any triangle ABC with incenter I, the angle "
            "bisector of angle BIC passes through the midpoint of arc BC "
            "of the circumcircle not containing A."
        )
    print(f"\n[query]\n{query}\n")

    client = chromadb.PersistentClient(path=str(DB_DIR))
    col = client.get_collection(COLLECTION)
    print(f"[db] collection size = {col.count()}\n")

    model_dir = resolve_model_dir()
    print(f"[model] loading from {model_dir} on {DEVICE}...")
    model = SentenceTransformer(str(model_dir), device=DEVICE, trust_remote_code=True)
    if DEVICE == "cuda":
        model = model.half()

    vec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0].tolist()

    res = col.query(
        query_embeddings=[vec],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    for i, (doc, meta, dist) in enumerate(zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    )):
        sim = 1 - dist
        print(f"--- Top {i+1} | similarity={sim:.4f} | source={meta['source']} ---")
        print(doc[:1500])
        if len(doc) > 1500:
            print(f"... [{len(doc) - 1500} more chars]")
        print()


if __name__ == "__main__":
    main()
