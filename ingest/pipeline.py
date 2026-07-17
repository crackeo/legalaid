"""Phase 1 CLI: crawl sources, then build the JSONL corpus from raw PDFs.

Usage:
    python -m ingest.pipeline crawl [--source oag-acts] [--raw-dir corpus/raw]
    python -m ingest.pipeline build [--raw-dir corpus/raw] [--out corpus/chunks.jsonl]

`crawl` downloads every PDF linked from the registered index pages into
corpus/raw/ and records them in corpus/raw/manifest.json. `build` extracts,
cleans and chunks every manifest entry into one JSONL file — the canonical
corpus that Phase 2 indexes into the vector database.
"""

import argparse
import json
import sys
from pathlib import Path

from .chunk import chunk_document
from .crawler import crawl_source
from .extract import pdf_to_clean_pages
from .sources import SOURCES, get_source


def cmd_crawl(args: argparse.Namespace) -> int:
    raw_dir = Path(args.raw_dir)
    sources = [get_source(args.source)] if args.source else SOURCES
    for source in sources:
        print(f"== {source.name}")
        summary = crawl_source(source, raw_dir)
        print(f"   {summary}")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    raw_dir = Path(args.raw_dir)
    manifest_path = raw_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"no manifest at {manifest_path} — run `crawl` first", file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_docs = n_chunks = n_empty = 0
    with out_path.open("w", encoding="utf-8") as out:
        for url, entry in sorted(manifest.items()):
            pdf_path = raw_dir / entry["file"]
            if not pdf_path.exists():
                print(f"  MISSING {pdf_path}", file=sys.stderr)
                continue
            pages = pdf_to_clean_pages(pdf_path)
            if not any(pages):
                # No text layer: scanned PDF, needs OCR (later phase).
                print(f"  NO-TEXT (needs OCR): {entry['title'][:60]!r}")
                n_empty += 1
                continue
            chunks = chunk_document(
                pages, doc_title=entry["title"], source_url=url,
                doc_type=entry.get("doc_type", ""),
                language=entry.get("language", "en"),
            )
            for c in chunks:
                out.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
            n_docs += 1
            n_chunks += len(chunks)

    print(f"built {out_path}: {n_chunks} chunks from {n_docs} documents "
          f"({n_empty} scanned PDFs skipped, need OCR)")
    return 0


def cmd_update(args: argparse.Namespace) -> int:
    """Re-crawl -> rebuild corpus -> re-index. Run weekly (cron) so new Acts
    and amendments flow into the chatbot automatically. The crawler's
    SHA-256 manifest means unchanged PDFs are never re-downloaded."""
    raw_dir = Path(args.raw_dir)
    changed = 0
    for source in SOURCES:
        print(f"== {source.name}")
        summary = crawl_source(source, raw_dir)
        print(f"   {summary}")
        changed += summary["downloaded"]
    if not changed and not args.force:
        print("no new or amended documents; index left untouched")
        return 0
    rc = cmd_build(argparse.Namespace(raw_dir=args.raw_dir, out=args.out))
    if rc != 0:
        return rc
    from rag.cli import cmd_index  # late import: rag is optional for pure crawling
    return cmd_index(argparse.Namespace(corpus=args.out, db=args.db))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ingest.pipeline", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_crawl = sub.add_parser("crawl", help="download PDFs from official sources")
    p_crawl.add_argument("--source", choices=[s.key for s in SOURCES],
                         help="crawl one source only (default: all)")
    p_crawl.add_argument("--raw-dir", default="corpus/raw")
    p_crawl.set_defaults(func=cmd_crawl)

    p_build = sub.add_parser("build", help="extract+chunk raw PDFs into JSONL corpus")
    p_build.add_argument("--raw-dir", default="corpus/raw")
    p_build.add_argument("--out", default="corpus/chunks.jsonl")
    p_build.set_defaults(func=cmd_build)

    p_update = sub.add_parser(
        "update", help="re-crawl, rebuild corpus and re-index if anything changed")
    p_update.add_argument("--raw-dir", default="corpus/raw")
    p_update.add_argument("--out", default="corpus/chunks.jsonl")
    p_update.add_argument("--db", default="corpus/index.db")
    p_update.add_argument("--force", action="store_true",
                          help="rebuild even when the crawl found nothing new")
    p_update.set_defaults(func=cmd_update)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
