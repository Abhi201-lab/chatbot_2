"""One-off normalization script for existing pgvector embeddings.

Usage (PowerShell example):
  $env:DATABASE_URL="postgresql://postgres:123@localhost:5543/chatdb"
  python scripts/normalize_vectors.py --batch-size 500 --dry-run
  python scripts/normalize_vectors.py --batch-size 500

Features:
- Streams rows with a server-side cursor (low memory footprint)
- Computes L2 norm and rescales to unit length when norm>0
- Supports dry-run mode (reports how many would be updated)
- Safe commit batching

Prereqs: psycopg2-binary installed (already in requirements for services)

IMPORTANT: Stop heavy write traffic to the table during normalization to avoid
unnecessary contention. Reads (retrieval) can continue; partial normalization
won't break queries but similarity scores gradually stabilize as more rows
become unit vectors.
"""
from __future__ import annotations
import os, math, argparse, ast, time, sys
import psycopg2

DEFAULT_BATCH = 500


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH, help="Update batch commit size")
    p.add_argument("--dry-run", action="store_true", help="Scan only; do not write updates")
    p.add_argument("--limit", type=int, default=None, help="Optional max rows to process (debug)")
    return p.parse_args()


def main():
    args = parse_args()
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not set", file=sys.stderr)
        sys.exit(1)

    conn = psycopg2.connect(db_url)
    # server-side cursor
    cur = conn.cursor(name="vecnorm")
    select_sql = "SELECT id, embedding::text FROM vector_chunks"
    if args.limit:
        select_sql += f" LIMIT {int(args.limit)}"
    cur.execute(select_sql)

    update_sql = "UPDATE vector_chunks SET embedding = %s WHERE id = %s"

    batch = []
    processed = 0
    updated = 0
    t0 = time.time()

    def flush():
        nonlocal batch, updated
        if not batch or args.dry_run:
            batch.clear(); return
        with conn.cursor() as wcur:
            for emb_text, vid in batch:
                wcur.execute(update_sql, (emb_text, vid))
        conn.commit()
        updated += len(batch)
        batch.clear()

    try:
        for vid, emb_text in cur:
            processed += 1
            try:
                vec = ast.literal_eval(emb_text)
                if not isinstance(vec, list):
                    continue
                norm_sq = sum(v*v for v in vec)
                if norm_sq <= 0:
                    continue
                length = math.sqrt(norm_sq)
                # already normalized? skip small drift (<1e-6)
                if abs(1.0 - length) < 1e-6:
                    continue
                normed = [v/length for v in vec]
                batch.append(("[" + ",".join(f"{v:.8f}" for v in normed) + "]", vid))
            except Exception:
                # Skip malformed row
                continue
            if len(batch) >= args.batch_size:
                flush()
                if processed % 5000 == 0:
                    rate = processed / max(time.time() - t0, 1e-6)
                    print(f"Processed={processed} Updated={updated} Rate={rate:.1f} rows/s")
        # final flush
        flush()
    finally:
        cur.close(); conn.close()

    elapsed = time.time() - t0
    print(f"DONE processed={processed} updated={updated} elapsed={elapsed:.2f}s dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
