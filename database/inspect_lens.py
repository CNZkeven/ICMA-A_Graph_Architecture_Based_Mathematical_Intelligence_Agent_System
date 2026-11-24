"""Inspect real text length distribution across the olympiad dataset."""

import re
from pathlib import Path

ROOT = Path(r"E:/test/AI-MO")
DIRS = [
    ROOT / "olympiads",
    ROOT / "parquet" / "olympiads-ref-base",
    ROOT / "parquet" / "aimo-validation-aime",
]

prob_lens = []  # in characters
sol_lens = []
combined_lens = []

file_count = 0
for base in DIRS:
    for md_path in base.rglob("*.md"):
        try:
            content = md_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        file_count += 1
        chunks = re.split(r"(?=^##\s*Problem\s)", content, flags=re.MULTILINE)
        for chunk in chunks:
            if not chunk.lstrip().startswith("## Problem"):
                continue
            prob = re.search(
                r"^##\s*Problem[^\n]*\n(.*?)(?=^##\s*Solution\b)",
                chunk,
                re.MULTILINE | re.DOTALL | re.IGNORECASE,
            )
            sol = re.search(
                r"^##\s*Solution\b(.*?)(?=^##\s*Problem\b|\Z)",
                chunk,
                re.MULTILINE | re.DOTALL | re.IGNORECASE,
            )
            if not prob:
                continue
            p = prob.group(1).strip()
            s = sol.group(1).strip() if sol else ""
            prob_lens.append(len(p))
            sol_lens.append(len(s))
            combined_lens.append(len(p) + len(s))

print(f"scanned {file_count} files, {len(prob_lens)} problem-solution pairs")

import statistics as st

def summary(name, arr):
    if not arr:
        return
    arr_sorted = sorted(arr)
    n = len(arr_sorted)
    p50 = arr_sorted[n // 2]
    p90 = arr_sorted[int(n * 0.90)]
    p95 = arr_sorted[int(n * 0.95)]
    p99 = arr_sorted[int(n * 0.99)]
    print(f"{name:>10}  n={n}  mean={st.mean(arr):.0f}  p50={p50}  p90={p90}  p95={p95}  p99={p99}  max={max(arr)}")

summary("problem", prob_lens)
summary("solution", sol_lens)
summary("combined", combined_lens)
