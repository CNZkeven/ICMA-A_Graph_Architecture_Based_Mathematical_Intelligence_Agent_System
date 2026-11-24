"""Recover the 68 hu-archive task files whose problem text was truncated to
empty by fix_hu_archive.py's header-parsing bug (section() stopped at the
nested '## Problems' wrapper line instead of at '## Solution').

The original problem text survives in the DB records (these items were never
re-embedded because the merge made them fail the >=20-char filter), so:
  1. extract original problem from the DB document,
  2. keep the current file's (real) Solution section,
  3. rewrite the file as problem + solution,
  4. re-embed and upsert.
"""

import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).parent))
import chromadb
from build_db import (ROOT, COLLECTION, DB_DIR, MAX_SEQ, MAX_TEXT_CHARS,
                      collect_files, split_blocks, resolve_model_dir, log)

PROB_RE = re.compile(r"^##\s*Problem[^\n]*\n(.*?)(?=^##\s*Solution\b)",
                     re.MULTILINE | re.DOTALL | re.IGNORECASE)
SOL_RE = re.compile(r"^##\s*Solution\b(.*?)(?=^##\s*Problem\b|\Z)",
                    re.MULTILINE | re.DOTALL | re.IGNORECASE)


def main():
    client = chromadb.PersistentClient(path=str(DB_DIR))
    col = client.get_collection(COLLECTION)

    db_ids = set()
    offset = 0
    while True:
        got = col.get(limit=10000, offset=offset, include=[])
        db_ids.update(got["ids"])
        if len(got["ids"]) < 10000:
            break
        offset += 10000

    scan_ids = set()
    for md_path, contest, year in collect_files():
        rel_key = str(md_path.relative_to(ROOT))
        content = md_path.read_text(encoding="utf-8", errors="replace")
        for idx, (prob, sol) in enumerate(split_blocks(content)):
            if len(prob[:MAX_TEXT_CHARS]) >= 20:
                scan_ids.add(f"{rel_key}::{idx}")

    missing = sorted(db_ids - scan_ids)
    log.info(f"[recover] DB ids missing from scan: {len(missing)}")
    if not missing:
        log.info("[recover] nothing to do")
        return

    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = resolve_model_dir()
    log.info(f"[recover] loading model from {model_dir} on {device}")
    model = SentenceTransformer(str(model_dir), device=device, trust_remote_code=True)
    model.max_seq_length = MAX_SEQ
    if device == "cuda":
        model = model.half()

    fixed = 0
    for rid in missing:
        rel_key, _, idx_s = rid.rpartition("::")
        md_path = ROOT / rel_key
        if not md_path.exists():
            log.warning(f"[recover] file gone, skipping: {md_path}")
            continue
        doc = col.get(ids=[rid], include=["documents"])["documents"][0]
        m = PROB_RE.search(doc)
        if not m:
            log.warning(f"[recover] no problem section in DB doc: {rid}")
            continue
        orig_prob = m.group(1).strip()
        cur = md_path.read_text(encoding="utf-8", errors="replace")
        sm = SOL_RE.search(cur)
        cur_sol = sm.group(1).strip() if sm else ""
        md_path.write_text(f"## Problem\n{orig_prob}\n\n## Solution\n{cur_sol}\n",
                           encoding="utf-8")

        prob_trunc = orig_prob[:MAX_TEXT_CHARS]
        sol_trunc = cur_sol[:MAX_TEXT_CHARS]
        new_doc = f"## Problem\n{prob_trunc}\n\n## Solution\n{sol_trunc}".strip()
        year_match = re.search(r"(19|20)\d{2}", md_path.stem)
        meta = {"source": rel_key, "contest": "Hungary",
                "year": year_match.group(0) if year_match else "",
                "idx": int(idx_s)}
        emb = model.encode([new_doc], normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False)
        col.upsert(ids=[rid], documents=[new_doc],
                   embeddings=emb.tolist(), metadatas=[meta])
        fixed += 1
        if fixed % 20 == 0:
            log.info(f"[recover] {fixed}/{len(missing)} ...")

    if device == "cuda":
        torch.cuda.empty_cache()
    log.info(f"[recover] done: fixed={fixed} db_total={col.count()}")


if __name__ == "__main__":
    main()
