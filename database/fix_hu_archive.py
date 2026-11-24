"""Fix hu-archive _task/_solution pairing (files + DB).

The hu-archive subset was split into <base>_task.md (real problem + placeholder
solution) and <base>_solution.md (solution text misplaced under '## Problem').
This script:
  1. Merges each pair: rewrite <base>_task.md as problem + real solution.
  2. Deletes the now-redundant <base>_solution.md files.
  3. Re-embeds merged records (same params as build_db.py) and upserts them
     over the placeholder versions (id '<task relpath>::0').
  4. Deletes the broken '<solution relpath>::0' records from the DB.

Idempotent: skips task files that no longer carry the placeholder; DB ops are
upserts/deletes. fix_report.json (written in phase 1) is the authoritative
pair list so phase 2 + reruns work after the solution files are gone.

Usage:
  python fix_hu_archive.py --dry-run    # report only, no changes
  python fix_hu_archive.py              # full fix (files + DB)
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

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
DB_DIR = Path(r"E:/test/AI-MO/database/v2")
COLLECTION = "olympiad_problems"
ROOT = Path(r"E:/test/AI-MO")
HU_ROOT = ROOT / "olympiads/Hungary/md/hu-archive"
BATCH = 8
MAX_SEQ = 512
MAX_TEXT_CHARS = 5000
PLACEHOLDER = "无解答内容"
REPORT = DB_DIR / "fix_report.json"
LOG_FILE = DB_DIR / "fix_hu_archive.log"


class _FlushFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


class _FlushStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


log = logging.getLogger("fix_hu")
log.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
_fh = _FlushFileHandler(LOG_FILE, mode="a", encoding="utf-8")
_fh.setFormatter(_fmt)
_sh = _FlushStreamHandler(sys.stderr)
_sh.setFormatter(_fmt)
log.addHandler(_fh)
log.addHandler(_sh)


def section(content: str, header: str) -> str:
    m = re.search(
        rf"^##\s*{header}[^\n]*\n(.*?)(?=^##\s|\Z)",
        content, re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    return m.group(1).strip() if m else ""


def problem_section(content: str) -> str:
    """Problem text up to '## Solution' ONLY — the problem body may contain
    nested '## ...' wrapper lines (e.g. '## Problems'), which must be kept.
    (The 2026-08-19 run lost 68 files to this; see recover_truncated_tasks.py.)
    """
    m = re.search(
        r"^##\s*Problem[^\n]*\n(.*?)(?=^##\s*Solution\b)",
        content, re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    return m.group(1).strip() if m else ""


def resolve_model_dir() -> Path:
    try:
        from modelscope import snapshot_download
        return Path(snapshot_download(MODEL_ID))
    except Exception:
        cand = Path.home() / ".cache/modelscope/models/Qwen--Qwen3-Embedding-0.6B/snapshots/master"
        if (cand / "config.json").exists():
            return cand
        raise


def scan_pairs():
    tasks, sols = {}, {}
    for p in HU_ROOT.rglob("*.md"):
        if p.name.endswith("_task.md"):
            tasks[p.name[:-len("_task.md")]] = p
        elif p.name.endswith("_solution.md"):
            sols[p.name[:-len("_solution.md")]] = p
    return tasks, sols


def main():
    dry = "--dry-run" in sys.argv
    log.info("=" * 70)
    log.info(f"[start] fix_hu_archive dry_run={dry} pid={os.getpid()}")

    tasks, sols = scan_pairs()
    paired = sorted(set(tasks) & set(sols))
    log.info(f"[scan] task files={len(tasks)} solution files={len(sols)} paired={len(paired)}")

    # If a report exists from an earlier phase-1 run, trust it for DB work.
    report = None
    if REPORT.exists():
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        log.info(f"[scan] existing fix_report.json: {len(report['pairs'])} pairs "
                 f"({sum(1 for p in report['pairs'] if p['has_real_solution'])} with real solution)")

    # ---------------- phase 1: merge files --------------------------------
    pairs_info = []
    n_merged = n_kept_placeholder = n_skipped_done = 0
    for base in paired:
        tp, sp = tasks[base], sols[base]
        rel_task = str(tp.relative_to(ROOT))
        rel_sol = str(sp.relative_to(ROOT))
        tcontent = tp.read_text(encoding="utf-8", errors="replace")
        scontent = sp.read_text(encoding="utf-8", errors="replace")
        tsol = section(tcontent, "Solution")
        if PLACEHOLDER not in tsol and len(tsol) >= 20:
            n_skipped_done += 1  # already merged in a previous run
            pairs_info.append({"base": base, "rel_task": rel_task,
                               "rel_sol": rel_sol, "has_real_solution": True})
            continue
        prob = problem_section(tcontent)
        real_sol = section(scontent, "Problem")
        has_real = len(real_sol) > 20 and PLACEHOLDER not in real_sol
        if has_real:
            merged = f"## Problem\n{prob}\n\n## Solution\n{real_sol}\n"
        else:
            merged = f"## Problem\n{prob}\n\n## Solution\n{tsol}\n"
        pairs_info.append({"base": base, "rel_task": rel_task,
                           "rel_sol": rel_sol, "has_real_solution": has_real})
        if not dry:
            tp.write_text(merged, encoding="utf-8")
            sp.unlink()
        if has_real:
            n_merged += 1
        else:
            n_kept_placeholder += 1
        done = n_merged + n_kept_placeholder
        if done % 2000 == 0:
            log.info(f"[files] {done}/{len(paired)} processed ...")

    log.info(f"[files] merged_real_solution={n_merged} kept_placeholder={n_kept_placeholder} "
             f"already_done={n_skipped_done} dry_run={dry}")

    if dry:
        log.info("[dry-run] no files changed, stopping before DB phase")
        return

    REPORT.write_text(json.dumps({"pairs": pairs_info}, ensure_ascii=False, indent=1),
                      encoding="utf-8")
    log.info(f"[files] report written: {REPORT}")

    # ---------------- phase 2: DB updates ---------------------------------
    import chromadb
    import torch
    from sentence_transformers import SentenceTransformer

    if report is None:
        report = {"pairs": pairs_info}
    pairs_info = report["pairs"]

    client = chromadb.PersistentClient(path=str(DB_DIR))
    col = client.get_or_create_collection(name=COLLECTION,
                                          metadata={"hnsw:space": "cosine"})
    db_before = col.count()
    log.info(f"[db] records before fix: {db_before}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = resolve_model_dir()
    log.info(f"[model] loading from {model_dir} on {device} ...")
    model = SentenceTransformer(str(model_dir), device=device, trust_remote_code=True)
    model.max_seq_length = MAX_SEQ
    if device == "cuda":
        model = model.half()

    # re-embed merged records that gained a real solution
    to_embed = []
    for info in pairs_info:
        if not info["has_real_solution"]:
            continue
        tp = ROOT / info["rel_task"]
        content = tp.read_text(encoding="utf-8", errors="replace")
        prob = section(content, "Problem")
        sol = section(content, "Solution")
        doc = f"## Problem\n{prob[:MAX_TEXT_CHARS]}\n\n## Solution\n{sol[:MAX_TEXT_CHARS]}".strip()
        if len(prob[:MAX_TEXT_CHARS]) < 20:
            continue
        rel_key = info["rel_task"]
        year_match = re.search(r"(19|20)\d{2}", tp.stem)
        to_embed.append({
            "id": f"{rel_key}::0",
            "doc": doc,
            "meta": {"source": rel_key, "contest": "Hungary",
                     "year": year_match.group(0) if year_match else "", "idx": 0},
        })
    log.info(f"[db] merged records to re-embed: {len(to_embed)}")

    inserted = 0
    t0 = time.time()
    for start in range(0, len(to_embed), BATCH):
        chunk = to_embed[start:start + BATCH]
        docs = [it["doc"] for it in chunk]
        emb = model.encode(docs, batch_size=len(docs), normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False)
        col.upsert(ids=[it["id"] for it in chunk], documents=docs,
                   embeddings=emb.tolist(), metadatas=[it["meta"] for it in chunk])
        if device == "cuda":
            torch.cuda.empty_cache()
        inserted += len(chunk)
        if inserted % 800 < BATCH or inserted == len(to_embed):
            rate = inserted / max(time.time() - t0, 1e-6)
            eta = (len(to_embed) - inserted) / rate if rate > 0 else 0
            log.info(f"[progress] re-embedded={inserted}/{len(to_embed)} "
                     f"rate={rate:.1f}/s eta={eta/60:.1f}min")

    # delete the broken solution-file records
    sol_ids = [f"{info['rel_sol']}::0" for info in pairs_info]
    # also the orphaned solution files (no matching task): remove their DB
    # records too (files are kept on disk), they are solutions without problems
    orphan_sols = sorted(set(sols) - set(tasks))
    for base in orphan_sols:
        sol_ids.append(f"{str(sols[base].relative_to(ROOT))}::0")
        log.info(f"[orphan] solution without task (file kept, DB record removed): {sols[base]}")
    deleted = 0
    for start in range(0, len(sol_ids), 1000):
        batch = sol_ids[start:start + 1000]
        present = col.get(ids=batch, include=[])["ids"]
        if present:
            col.delete(ids=present)
            deleted += len(present)
    log.info(f"[db] broken solution-file records deleted: {deleted} "
             f"(of {len(sol_ids)} candidate ids)")

    log.info(f"[done] re_embedded={inserted} deleted={deleted} "
             f"db_total={col.count()} (was {db_before})")


if __name__ == "__main__":
    main()
