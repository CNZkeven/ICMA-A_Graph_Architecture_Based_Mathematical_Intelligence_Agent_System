"""Build a ChromaDB over olympiad problems using Qwen3-Embedding-0.6B on GPU.

v2.2 (2026-08-19)
  * Dynamic batch packing instead of fixed BATCH: docs are token-counted up
    front, then greedily packed so that  batch x padded_len^2 <= MASK_BUDGET.
    Short docs get big batches, long docs small ones.
  * VRAM tuning (RTX 4060 Laptop 8GB, WDDM): measured fixed cost ~1.6 GB
    (weights fp16 + CUDA context); mask family ~ 1.5*k*batch*seq^2 with
    k ~= 0.2 KB/(doc*token^2), calibrated from the v2.1 run (peak 2081 MB at
    batch 8 / seq 512) and the v1 OOM crash. With MASK_BUDGET=10M and
    BATCH_MAX=40 the predicted peak is ~4.9 GB, worst case ~6.9 GB.
    On a bigger GPU raise MASK_BUDGET (x2 budget ~= +3 GB peak at seq 512)
    and BATCH_MAX; watch the gpu_peak field in [progress] lines.
  * OOM safety net: a batch that still OOMs is emptied, halved and retried
    recursively (down to single docs) instead of crashing the run.

v2.1 changes (kept): ModelScope-only model loading (offline), DB-id based
resume, per-file manifest audit log, logging with per-record flush,
MAX_SEQ=512, MAX_TEXT_CHARS=5000.

Ids: "<relative_path>::<problem_index>"; upsert is idempotent.
Usage:
  python build_db.py                 # normal (resume) run
  python build_db.py --limit=16      # only the first 16 pending items
  python build_db.py --limit=16 --force   # smoke test: re-embed first 16
  python build_db.py --force         # full in-place re-embed (no DB reset)
"""

import json
import logging
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
DB_DIR = Path(r"E:/test/AI-MO/database/v2")
COLLECTION = "olympiad_problems"

MAX_SEQ = 512          # token cap per doc (truncated by the model tokenizer)
MAX_TEXT_CHARS = 5000  # per-field char cap (user-specified)
BATCH_MAX = 40         # hard cap: sequences per forward pass
MASK_BUDGET = 10_000_000  # cap on batch x padded_len^2 per forward pass.
                          # peak VRAM ~= 1.6GB + 1.5*k*budget (k~=0.2KB) + act
PROGRESS_EVERY = 200   # log a [progress] line every N embedded items

ROOT = Path(r"E:/test/AI-MO")
TARGET_DIRS = [
    ROOT / "olympiads",
    ROOT / "parquet" / "olympiads-ref-base",
    ROOT / "parquet" / "aimo-validation-aime",
]
MANIFEST = DB_DIR / "manifest.json"
LOG_FILE = DB_DIR / "build.log"
DEVICE = None  # resolved after torch import


# --- logging: file + stderr, flushed per record -----------------------------

class _FlushFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


class _FlushStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


log = logging.getLogger("build_db")
log.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
_fh = _FlushFileHandler(LOG_FILE, mode="a", encoding="utf-8")
_fh.setFormatter(_fmt)
_sh = _FlushStreamHandler(sys.stderr)
_sh.setFormatter(_fmt)
log.addHandler(_fh)
log.addHandler(_sh)


# --- model (ModelScope only) -------------------------------------------------

def resolve_model_dir() -> Path:
    try:
        from modelscope import snapshot_download
        return Path(snapshot_download(MODEL_ID))
    except Exception as e:
        log.warning(f"[model] modelscope snapshot_download failed ({e!r}); using local cache")
        for cand in (
            Path.home() / ".cache/modelscope/models/Qwen--Qwen3-Embedding-0.6B/snapshots/master",
            Path.home() / ".cache/modelscope/hub/Qwen/Qwen3-Embedding-0.6B",
        ):
            if (cand / "config.json").exists():
                return cand
        raise


# --- dataset parsing ---------------------------------------------------------

def split_blocks(content: str):
    chunks = re.split(r"(?=^##\s*Problem\s)", content, flags=re.MULTILINE)
    for chunk in chunks:
        if not chunk.lstrip().startswith("## Problem"):
            continue
        prob = re.search(
            r"^##\s*Problem[^\n]*\n(.*?)(?=^##\s*Solution\b)",
            chunk, re.MULTILINE | re.DOTALL | re.IGNORECASE,
        )
        sol = re.search(
            r"^##\s*Solution\b(.*?)(?=^##\s*Problem\b|\Z)",
            chunk, re.MULTILINE | re.DOTALL | re.IGNORECASE,
        )
        if not prob:
            continue
        yield prob.group(1).strip(), (sol.group(1).strip() if sol else "")


def collect_files():
    for base in TARGET_DIRS:
        if not base.exists():
            log.warning(f"[skip missing] {base}")
            continue
        for md_path in base.rglob("*.md"):
            rel = md_path.relative_to(base)
            contest = rel.parts[0] if len(rel.parts) > 1 else base.name
            year_match = re.search(r"(19|20)\d{2}", md_path.stem)
            year = year_match.group(0) if year_match else ""
            yield md_path, contest, year


def write_manifest(manifest: dict):
    tmp = MANIFEST.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, MANIFEST)


def load_existing_ids(col) -> set:
    ids = set()
    offset = 0
    while True:
        try:
            got = col.get(limit=10000, offset=offset, include=[])
        except TypeError:
            got = col.get(include=[])
            ids.update(got["ids"])
            break
        batch = got["ids"]
        ids.update(batch)
        if len(batch) < 10000:
            break
        offset += 10000
    return ids


# --- batching ----------------------------------------------------------------

def pack_batches(lengths, budget=MASK_BUDGET, max_count=BATCH_MAX):
    """Greedy pack of item indices, longest first.

    Cost model: a forward pass over N docs padded to length L allocates mask
    intermediates proportional to N*L^2, so a batch is closed when adding the
    next doc would push (N+1)*L_max^2 past `budget`.
    """
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    batches, cur = [], []
    for i in order:
        if cur:
            lmax = lengths[cur[0]]  # sorted desc: first item is the longest
            if (len(cur) + 1) * lmax * lmax > budget or len(cur) >= max_count:
                batches.append(cur)
                cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    return batches


def encode_safe(model, torch, docs):
    """Encode with OOM fallback: halve the batch and retry recursively."""
    try:
        return model.encode(
            docs,
            batch_size=len(docs),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    except torch.cuda.OutOfMemoryError:
        log.warning(f"[oom] batch of {len(docs)} docs hit CUDA OOM; "
                    f"emptying cache and splitting")
        torch.cuda.empty_cache()
        if len(docs) == 1:
            raise
        import numpy as np
        half = len(docs) // 2
        return np.vstack([
            encode_safe(model, torch, docs[:half]),
            encode_safe(model, torch, docs[half:]),
        ])


# --- main --------------------------------------------------------------------

def main():
    limit = None
    force = "--force" in sys.argv
    for a in sys.argv[1:]:
        if a.startswith("--limit="):
            limit = int(a.split("=", 1)[1])

    import chromadb
    import torch
    from sentence_transformers import SentenceTransformer

    global DEVICE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    log.info("=" * 70)
    log.info(f"[start] build_db v2.2 pid={os.getpid()} device={DEVICE} "
             f"max_seq={MAX_SEQ} max_chars={MAX_TEXT_CHARS} "
             f"batch_max={BATCH_MAX} mask_budget={MASK_BUDGET}"
             + (f" LIMIT={limit}" if limit else "")
             + (" FORCE(re-embed)" if force else ""))

    DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_DIR))
    col = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    db_before = col.count()
    log.info(f"[db] records before this run: {db_before}")

    existing = set()
    if not force:
        log.info("[resume] reading existing ids from DB ...")
        existing = load_existing_ids(col)
        log.info(f"[resume] {len(existing)} existing ids in DB")

    model_dir = resolve_model_dir()
    log.info(f"[model] loading {MODEL_ID} from {model_dir} on {DEVICE} ...")
    model = SentenceTransformer(str(model_dir), device=DEVICE, trust_remote_code=True)
    model.max_seq_length = MAX_SEQ
    if DEVICE == "cuda":
        model = model.half()
    log.info(f"[model] loaded, dim={model.get_embedding_dimension()} "
             f"max_seq={model.max_seq_length}")

    # ---- phase 1: scan + parse, build the exact work list -------------------
    manifest = {}
    if MANIFEST.exists():
        try:
            manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning(f"[manifest] unreadable ({e!r}); starting fresh audit log")
            manifest = {}

    work = []
    skipped_existing = 0
    files_total = files_ok = files_readfail = files_parsefail = 0
    t_parse = time.time()
    for md_path, contest, year in collect_files():
        files_total += 1
        rel_key = str(md_path.relative_to(ROOT))
        try:
            content = md_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            files_readfail += 1
            log.warning(f"[read fail] {md_path}: {e}")
            manifest[rel_key] = "read_error"
            continue

        n_new = 0
        n_found = 0
        try:
            for idx, (prob, sol) in enumerate(split_blocks(content)):
                prob_trunc = prob[:MAX_TEXT_CHARS]
                sol_trunc = sol[:MAX_TEXT_CHARS]
                doc = f"## Problem\n{prob_trunc}\n\n## Solution\n{sol_trunc}".strip()
                if len(prob_trunc) < 20:
                    continue
                n_found += 1
                rid = f"{rel_key}::{idx}"
                if rid in existing:
                    skipped_existing += 1
                    continue
                work.append({
                    "id": rid,
                    "doc": doc,
                    "meta": {"source": rel_key, "contest": contest,
                             "year": year, "idx": idx},
                })
                n_new += 1
        except Exception as e:
            files_parsefail += 1
            log.warning(f"[parse fail] {md_path}: {e}")
            manifest[rel_key] = "parse_error"
            continue

        files_ok += 1
        manifest[rel_key] = n_found  # audit log only; resume uses DB ids
        if files_total % 2000 == 0:
            log.info(f"[scan] {files_total} files scanned, "
                     f"{len(work)} new items queued ...")

    write_manifest(manifest)
    log.info(f"[scan] done in {time.time() - t_parse:.1f}s: files={files_total} "
             f"ok={files_ok} read_fail={files_readfail} parse_fail={files_parsefail} "
             f"already_in_db={skipped_existing} to_embed={len(work)}")
    if limit is not None:
        work = work[:limit]
        log.info(f"[limit] work list truncated to first {len(work)} items")
    if not work:
        log.info(f"[done] nothing to embed; db_total={col.count()}")
        return

    # ---- phase 2: token-count + pack batches --------------------------------
    t_tok = time.time()
    enc = model.tokenizer([it["doc"] for it in work], add_special_tokens=False)
    lengths = [min(len(ids), MAX_SEQ) + 2 for ids in enc["input_ids"]]
    del enc
    batches = pack_batches(lengths)
    log.info(f"[pack] {len(work)} docs -> {len(batches)} batches in "
             f"{time.time() - t_tok:.1f}s "
             f"(avg {len(work) / len(batches):.1f} docs/batch; "
             f"tokens p50={sorted(lengths)[len(lengths)//2]} "
             f"max={max(lengths)})")

    # ---- phase 3: embed + upsert --------------------------------------------
    inserted = 0
    last_log_at = 0
    t_total = time.time()
    total = len(work)
    for bnum, idxs in enumerate(batches):
        chunk = [work[i] for i in idxs]
        docs = [it["doc"] for it in chunk]
        embeddings = encode_safe(model, torch, docs)
        col.upsert(
            ids=[it["id"] for it in chunk],
            documents=docs,
            embeddings=embeddings.tolist(),
            metadatas=[it["meta"] for it in chunk],
        )
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        inserted += len(chunk)
        if inserted - last_log_at >= PROGRESS_EVERY or bnum == len(batches) - 1:
            last_log_at = inserted
            elapsed = time.time() - t_total
            rate = inserted / elapsed if elapsed > 0 else 0
            eta = (total - inserted) / rate if rate > 0 else 0
            if DEVICE == "cuda":
                gpu_peak = torch.cuda.max_memory_allocated() // (1024 * 1024)
                gpu_cur = torch.cuda.memory_allocated() // (1024 * 1024)
            else:
                gpu_peak = gpu_cur = 0
            log.info(f"[progress] embedded={inserted}/{total} rate={rate:.1f}/s "
                     f"eta={eta / 60:.1f}min db_total={db_before + inserted} "
                     f"gpu={gpu_cur}MB peak={gpu_peak}MB batch={len(chunk)}")

    elapsed = time.time() - t_total
    db_total = col.count()
    gpu_peak = (torch.cuda.max_memory_allocated() // (1024 * 1024)
                if DEVICE == "cuda" else 0)
    log.info(f"[done] inserted={inserted} skipped={skipped_existing} "
             f"elapsed={elapsed:.1f}s db_total={db_total} gpu_peak={gpu_peak}MB")
    if db_total != db_before + inserted:
        log.warning(f"[done] db_total={db_total} != db_before+inserted="
                    f"{db_before + inserted} (concurrent writer/deleter?)")


if __name__ == "__main__":
    main()
