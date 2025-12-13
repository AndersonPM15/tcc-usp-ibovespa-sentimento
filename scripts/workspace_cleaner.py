from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORT = ROOT / "reports" / "cleanup_audit.md"
DEFAULT_QUARANTINE_ROOT = ROOT / "reports"

SAFE_DIR_NAMES = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".mypy_cache", ".ruff_cache"}
SAFE_FILE_GLOBS = [
    "*.log",
    "*.tmp",
    "tmp_*.txt",
    "tmp_*.log",
    ".tmp_*.txt",
    "*.bak",
    "*.swp",
    ".DS_Store",
    "Thumbs.db",
]
ARCHIVE_PREFIXES = ["mlruns", "notebooks/_runs", "reports/tmp"]
KEEP_TOP = {"data_raw", "data_processed", "configs", "src", "scripts", "venv"}
KEEP_EXTS = {".py", ".md", ".yaml", ".yml", ".txt", ".csv"}
KEEP_FILES = {"requirements.txt"}


@dataclass
class Item:
    path: Path
    rel: Path
    size: int
    mtime: float
    kind: str
    category: str


def human_size(num: float) -> str:
    step = 1024.0
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if num < step:
            return f"{num:.1f} {unit}"
        num /= step
    return f"{num:.1f} PB"


def is_within(rel: Path, prefix: str) -> bool:
    parts = Path(prefix).parts
    return rel.parts[: len(parts)] == parts


def matches_any(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in patterns)


def classify_path(rel: Path, path: Path) -> str:
    # Hard keep for raw/processed data and venv
    if rel.parts and rel.parts[0] in {"data_raw", "data_processed", "venv"}:
        return "KEEP"

    # Notebooks special cases
    if rel.parts and rel.parts[0] == "notebooks":
        if ".ipynb_checkpoints" in rel.parts:
            return "SAFE_DELETE"
        if len(rel.parts) > 1 and rel.parts[1] == "_runs":
            return "ARCHIVE"
        if rel.suffix == ".ipynb":
            return "KEEP"

    # Safe-delete patterns anywhere (except raw/processed handled above)
    if any(name in SAFE_DIR_NAMES for name in rel.parts):
        return "SAFE_DELETE"
    if matches_any(rel.name, SAFE_FILE_GLOBS):
        return "SAFE_DELETE"

    # Archive prefixes
    rel_posix = rel.as_posix()
    if any(rel_posix == p or rel_posix.startswith(f"{p}/") for p in ARCHIVE_PREFIXES):
        return "ARCHIVE"
    # Archive for generic *_run* or *_cache* outside protected tops
    if any(fnmatch.fnmatch(rel.name, pat) for pat in ["*_run*", "*_cache*"]):
        if not (rel.parts and rel.parts[0] in KEEP_TOP):
            return "ARCHIVE"

    # Keep rules
    if rel.parts and rel.parts[0] in {"configs", "src", "scripts"}:
        return "KEEP"
    if rel.suffix in KEEP_EXTS or rel.name in KEEP_FILES:
        return "KEEP"

    return "KEEP"


def size_of_tree(path: Path) -> int:
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for sub in path.rglob("*"):
        if sub.is_file():
            try:
                total += sub.stat().st_size
            except OSError:
                continue
    return total


def collect_inventory() -> Tuple[List[Item], Dict[str, Dict[str, float]], Dict[str, int]]:
    items: List[Item] = []
    top_stats: Dict[str, Dict[str, float]] = {}
    dir_size: Dict[str, int] = {}

    for path in ROOT.rglob("*"):
        try:
            rel = path.relative_to(ROOT)
        except ValueError:
            continue
        if rel == Path(".git") or ".git" in rel.parts:
            continue
        kind = "dir" if path.is_dir() else "file"
        size = 0
        if path.is_file():
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
        mtime = path.stat().st_mtime if path.exists() else 0
        category = classify_path(rel, path)
        items.append(Item(path=path, rel=rel, size=size, mtime=mtime, kind=kind, category=category))

        if path.is_file():
            top = rel.parts[0] if rel.parts else "."
            ts = top_stats.setdefault(top, {"files": 0, "size": 0})
            ts["files"] += 1
            ts["size"] += size
            parent_key = rel.parent.as_posix() or "."
            dir_size[parent_key] = dir_size.get(parent_key, 0) + size
            dir_size["."] = dir_size.get(".", 0) + size
    return items, top_stats, dir_size


def summarize_top_entries(items: List[Item], dir_size: Dict[str, int], limit: int = 30) -> List[Tuple[str, str, int]]:
    file_entries = [(it.rel.as_posix(), "file", it.size) for it in items if it.kind == "file"]
    dir_entries = [(path, "dir", sz) for path, sz in dir_size.items()]
    combined = file_entries + dir_entries
    combined.sort(key=lambda x: x[2], reverse=True)
    return combined[:limit]


def render_table(rows: List[Tuple[str, str, int]]) -> str:
    lines = ["| # | Path | Type | Size |", "|---|------|------|------|"]
    for idx, (path, typ, size) in enumerate(rows, start=1):
        lines.append(f"| {idx} | {path} | {typ} | {human_size(size)} |")
    return "\n".join(lines)


def write_report(report_path: Path, before: Dict[str, str], after: Dict[str, str] | None, plan: Dict[str, str]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Workspace Cleanup Audit")
    lines.append(f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Before")
    lines.append(before["summary"])
    lines.append("")
    lines.append("### Top 30 largest files/dirs")
    lines.append(before["top_table"])
    lines.append("")
    lines.append("## Plan (this run)")
    lines.append(plan["summary"])
    lines.append("")
    if plan.get("safe_list"):
        lines.append("### SAFE_DELETE candidates")
        lines.append(plan["safe_list"])
        lines.append("")
    if plan.get("archive_list"):
        lines.append("### ARCHIVE candidates")
        lines.append(plan["archive_list"])
        lines.append("")
    if after:
        lines.append("## After (post-apply)")
        lines.append(after["summary"])
        lines.append("")
        lines.append("### Top 30 largest files/dirs (after)")
        lines.append(after["top_table"])
        lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def format_summary(total_size: int, total_files: int, top_stats: Dict[str, Dict[str, float]]) -> str:
    lines = [f"- Total size: {human_size(total_size)}", f"- Total files: {total_files}", "- Top-level breakdown:"]
    for k in sorted(top_stats.keys()):
        size = human_size(top_stats[k]["size"])
        files = int(top_stats[k]["files"])
        lines.append(f"  - {k}/ : {files} files, {size}")
    return "\n".join(lines)


def format_list(paths: List[Item], limit: int = 100) -> str:
    paths_sorted = sorted(paths, key=lambda x: x.size, reverse=True)
    lines = []
    for idx, item in enumerate(paths_sorted[:limit], start=1):
        lines.append(f"- {item.rel.as_posix()} ({item.kind}, {human_size(item.size)})")
    if len(paths_sorted) > limit:
        lines.append(f"- ... and {len(paths_sorted) - limit} more")
    return "\n".join(lines) if lines else "- none"


def delete_target(path: Path) -> int:
    size = size_of_tree(path)
    if not path.exists():
        return 0
    if path.is_file():
        try:
            path.unlink()
        except OSError:
            return 0
    else:
        try:
            shutil.rmtree(path, ignore_errors=True)
        except OSError:
            return 0
    return size


def move_to_quarantine(path: Path, quarantine_root: Path) -> int:
    if not path.exists():
        return 0
    size = size_of_tree(path)
    target = quarantine_root / path.relative_to(ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(path), str(target))
    except shutil.Error:
        return 0
    return size


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Audit and clean workspace")
    parser.add_argument("--apply", action="store_true", help="apply changes (default is dry-run)")
    parser.add_argument("--dry-run", action="store_true", help="explicit dry-run (default behavior)")
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT), help="path to write markdown report")
    args = parser.parse_args(argv)

    mode_apply = args.apply and not args.dry_run

    report_path = Path(args.report_path)
    quarantine_dir = DEFAULT_QUARANTINE_ROOT / f"_quarantine_{dt.datetime.now():%Y%m%d}"

    items, top_stats, dir_size = collect_inventory()
    total_size = sum(it.size for it in items if it.kind == "file")
    total_files = sum(1 for it in items if it.kind == "file")

    safe_items = [it for it in items if it.category == "SAFE_DELETE"]
    archive_items = [it for it in items if it.category == "ARCHIVE"]
    keep_items = [it for it in items if it.category == "KEEP"]

    before = {
        "summary": format_summary(total_size, total_files, top_stats),
        "top_table": render_table(summarize_top_entries(items, dir_size)),
    }

    plan = {
        "summary": "\n".join(
            [
                f"- SAFE_DELETE: {len(safe_items)} entries",
                f"- ARCHIVE: {len(archive_items)} entries",
                f"- KEEP: {len(keep_items)} entries",
                f"- Mode: {'apply' if mode_apply else 'dry-run'}",
            ]
        ),
        "safe_list": format_list(safe_items, limit=150),
        "archive_list": format_list(archive_items, limit=150),
    }

    after = None
    deleted_size = 0
    moved_size = 0
    removed_log: List[Tuple[str, int]] = []
    moved_log: List[Tuple[str, int]] = []

    if mode_apply:
        # Delete safe entries
        for entry in sorted(safe_items, key=lambda x: len(x.rel.parts), reverse=True):
            if entry.path.exists():
                sz = delete_target(entry.path)
                deleted_size += sz
                if sz > 0:
                    removed_log.append((entry.rel.as_posix(), sz))
        # Move archive entries
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        for entry in sorted(archive_items, key=lambda x: len(x.rel.parts)):
            if entry.path.exists():
                sz = move_to_quarantine(entry.path, quarantine_dir)
                moved_size += sz
                if sz > 0:
                    moved_log.append((entry.rel.as_posix(), sz))

        # Rescan for after-state
        items_after, top_after, dir_after = collect_inventory()
        total_after_size = sum(it.size for it in items_after if it.kind == "file")
        total_after_files = sum(1 for it in items_after if it.kind == "file")
        after = {
            "summary": format_summary(total_after_size, total_after_files, top_after),
            "top_table": render_table(summarize_top_entries(items_after, dir_after)),
        }

    write_report(report_path, before, after, plan)

    print("=== CLEANUP SUMMARY ===")
    print(f"Mode: {'apply' if mode_apply else 'dry-run'}")
    print(f"Report: {report_path}")
    print(f"SAFE_DELETE entries: {len(safe_items)}")
    print(f"ARCHIVE entries: {len(archive_items)}")
    if mode_apply:
        print(f"Deleted: {human_size(deleted_size)}")
        print(f"Moved: {human_size(moved_size)} to {quarantine_dir}")
        if removed_log:
            top_removed = sorted(removed_log, key=lambda x: x[1], reverse=True)[:10]
            print("Top removed:")
            for rel, sz in top_removed:
                print(f"- {rel} ({human_size(sz)})")
        if moved_log:
            top_moved = sorted(moved_log, key=lambda x: x[1], reverse=True)[:10]
            print("Top moved:")
            for rel, sz in top_moved:
                print(f"- {rel} ({human_size(sz)})")
    else:
        print("Dry-run only; no changes applied.")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
