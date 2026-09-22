"""Copy just what the HF job needs (code, splits, spec) into hf_job/bundle/.

It excludes .venv, models/, runs/ and the raw/relabel drafts, so the upload stays at a few MB.

    python hf_job/make_bundle.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "hf_job/bundle"
ITEMS = ["src/jevlite", "scripts", "data/splits", "data/review/merge_report.txt", "docs/jev_spec.md",
         "pyproject.toml", "README.md"]


def main() -> None:
    splits = ROOT / "data/splits"
    missing = [s for s in ("train", "val", "test") if not (splits / f"{s}.jsonl").exists()]
    if missing:
        raise SystemExit(f"missing data/splits/{missing}: run merge_labels.py merge and split.py first")
    for item in ITEMS:
        src, dst = ROOT / item, DEST / item
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(src, dst)
    size = sum(f.stat().st_size for f in DEST.rglob("*") if f.is_file())
    print(f"bundle ready at {DEST} ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
