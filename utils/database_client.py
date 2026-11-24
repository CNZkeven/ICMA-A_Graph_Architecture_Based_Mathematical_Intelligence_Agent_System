"""ChromaDB client for olympiad problem retrieval.

Loads Qwen3-Embedding-0.6B from ModelScope cache (CPU) and queries the
pre-built ChromaDB index. Returns top-k similar problems+solutions.
"""

import os
import re
import threading
from pathlib import Path
from typing import List, Dict, Optional

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# 进程级共享的模型/集合单例与加载锁。ChromaDB 的 PersistentClient 在并发
# 首次初始化时会触发 Rust bindings 竞态（2026-08-31 评委报告第 6 节：
# 'RustBindingsAPI' object has no attribute 'bindings'，启动期两 worker 并发
# 查询全部失败、无参考注入；开发冒烟测试亦稳定复现）。每次 new 一个
# DatabaseClient 但共享同一份底层资源，首次加载在进程内互斥。
_SHARED_LOCK = threading.Lock()
_SHARED_MODEL = None
_SHARED_COLLECTION = None
_SHARED_KEY = None


class DatabaseClient:
    """Client for querying olympiad problem database via ChromaDB."""

    def __init__(
        self,
        db_dir: str = "database",
        collection_name: str = "olympiad_problems",
        model_id: str = "Qwen/Qwen3-Embedding-0.6B",
        device: str = "cpu",
    ):
        """Initialize the database client.

        Args:
            db_dir: path to ChromaDB directory (relative to project root)
            collection_name: ChromaDB collection name
            model_id: embedding model ID (ModelScope)
            device: "cpu" or "cuda"
        """
        self.db_dir = Path(db_dir)
        self.collection_name = collection_name
        self.model_id = model_id
        self.device = device
        self._model = None
        self._collection = None

    def _load_model(self):
        """Lazy-load embedding model from ModelScope cache (进程内单例 + 互斥)."""
        self._ensure_shared()

    def _ensure_shared(self):
        """模型与集合一次性互斥加载，进程内所有 DatabaseClient 实例共享。

        并发首次初始化 ChromaDB PersistentClient 会触发 Rust bindings 竞态
        （'RustBindingsAPI' object has no attribute 'bindings'），加载必须在
        锁内完成；加载后的 query/encode 只读，可安全并发。
        """
        global _SHARED_MODEL, _SHARED_COLLECTION, _SHARED_KEY
        if self._model is not None and self._collection is not None:
            return
        key = (str(self.db_dir.resolve()), self.collection_name, self.model_id)
        with _SHARED_LOCK:
            if _SHARED_KEY == key and _SHARED_MODEL is not None \
                    and _SHARED_COLLECTION is not None:
                self._model = _SHARED_MODEL
                self._collection = _SHARED_COLLECTION
                return
            if _SHARED_MODEL is None:
                from sentence_transformers import SentenceTransformer

                _SHARED_MODEL = SentenceTransformer(
                    str(self._resolve_model_dir()), device=self.device,
                    trust_remote_code=True)
            if _SHARED_COLLECTION is None:
                import chromadb

                if not (self.db_dir / "chroma.sqlite3").exists():
                    raise FileNotFoundError(
                        f"ChromaDB not found at {self.db_dir}. "
                        "Copy the database from E:/test/AI-MO/database/v2."
                    )
                client = chromadb.PersistentClient(path=str(self.db_dir))
                _SHARED_COLLECTION = client.get_collection(self.collection_name)
            _SHARED_KEY = key
            self._model = _SHARED_MODEL
            self._collection = _SHARED_COLLECTION

    def _resolve_model_dir(self) -> Path:
        """Find model in project directory or ModelScope cache."""
        # Priority 1: Project-local model directory
        project_model = Path(__file__).parent.parent / "models" / "Qwen3-Embedding-0.6B" / "snapshots" / "master"
        if (project_model / "config.json").exists():
            return project_model

        # Priority 2: Try ModelScope download/cache
        try:
            from modelscope import snapshot_download
            return Path(snapshot_download(self.model_id))
        except Exception:
            pass

        # Priority 3: Known cache paths
        for cand in (
            Path.home() / ".cache/modelscope/models/Qwen--Qwen3-Embedding-0.6B/snapshots/master",
            Path.home() / ".cache/modelscope/hub/Qwen/Qwen3-Embedding-0.6B",
        ):
            if (cand / "config.json").exists():
                return cand

        raise FileNotFoundError(
            f"Model {self.model_id} not found. Expected locations:\n"
            f"  1. Project-local: {project_model}\n"
            f"  2. ModelScope cache: ~/.cache/modelscope/\n"
            "To download: pip install modelscope && python -c "
            "\"from modelscope import snapshot_download; "
            "snapshot_download('Qwen/Qwen3-Embedding-0.6B')\""
        )

    def _load_collection(self):
        """Lazy-load ChromaDB collection (进程内单例 + 互斥，防 Rust bindings 竞态)."""
        self._ensure_shared()

    def query(self, problem: str, top_k: int = 3) -> List[Dict[str, any]]:
        """Query database for similar problems+solutions.

        Args:
            problem: problem text (query)
            top_k: number of results to return

        Returns:
            List of dicts with keys:
                - problem: problem text
                - solution: solution text
                - similarity: cosine similarity (0-1)
                - source: source file path
                - contest: contest name
                - year: year (if available)
        """
        self._load_model()
        self._load_collection()

        # Encode query
        vec = self._model.encode(
            [problem], normalize_embeddings=True, convert_to_numpy=True
        )[0].tolist()

        # Query ChromaDB
        res = self._collection.query(
            query_embeddings=[vec],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        # Parse results
        results = []
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            problem_text, solution_text = self._parse_document(doc)
            results.append({
                "problem": problem_text,
                "solution": solution_text,
                "similarity": 1 - dist,  # cosine distance -> similarity
                "source": meta.get("source", ""),
                "contest": meta.get("contest", ""),
                "year": meta.get("year", ""),
            })

        return results

    @staticmethod
    def _parse_document(doc: str) -> tuple[str, str]:
        """Parse a document into (problem, solution) pair.

        Documents are stored as:
            ## Problem
            <problem text>

            ## Solution
            <solution text>
        """
        prob_match = re.search(
            r'## Problem\s*(.*?)(?=## Solution|\Z)',
            doc, re.DOTALL | re.IGNORECASE
        )
        sol_match = re.search(
            r'## Solution\s*(.*?)(?=## Problem|\Z)',
            doc, re.DOTALL | re.IGNORECASE
        )

        problem = prob_match.group(1).strip() if prob_match else ""
        solution = sol_match.group(1).strip() if sol_match else ""

        return problem, solution

