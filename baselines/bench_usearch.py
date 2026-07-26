#!/usr/bin/env python3
"""usearch baseline — in-memory HNSW upper bound, used on Windows where
hnswlib has no prebuilt wheel (they implement the same algorithm; on Linux
run bench_hnswlib.py as well and keep both rows).

Usage:
    python baselines/bench_usearch.py --data-dir data/sift

Appends rows to the shared harness CSV schema:
    method,params,recall_at_K,qps,p50_us,p99_us
"""

import argparse
import csv
import os
import time

import numpy as np


def read_fvecs(path: str) -> np.ndarray:
    a = np.fromfile(path, dtype=np.int32)
    d = int(a[0])
    return a.reshape(-1, d + 1)[:, 1:].copy().view(np.float32)


def read_ivecs(path: str) -> np.ndarray:
    a = np.fromfile(path, dtype=np.int32)
    d = int(a[0])
    return a.reshape(-1, d + 1)[:, 1:].copy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--prefix", default="sift")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--ef-construction", type=int, default=200)
    ap.add_argument("--efs", default="10,20,40,80,120,200,400")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--latency-queries", type=int, default=1000)
    ap.add_argument("--csv", default="results/all.csv")
    args = ap.parse_args()

    from usearch.index import Index  # imported late so --help works without it

    base = read_fvecs(os.path.join(args.data_dir, f"{args.prefix}_base.fvecs"))
    queries = read_fvecs(os.path.join(args.data_dir, f"{args.prefix}_query.fvecs"))
    gt = read_ivecs(os.path.join(args.data_dir, f"{args.prefix}_groundtruth.ivecs"))
    n, dim = base.shape
    print(f"base {n}x{dim}, {len(queries)} queries, gt@{gt.shape[1]}")

    index = Index(
        ndim=dim,
        metric="l2sq",
        dtype="f32",
        connectivity=args.M,
        expansion_add=args.ef_construction,
    )
    t0 = time.perf_counter()
    index.add(np.arange(n), base, threads=args.threads)
    build_s = time.perf_counter() - t0
    print(f"build: {build_s:.1f}s (M={args.M}, efC={args.ef_construction}, {args.threads} threads)")

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    is_new = not os.path.exists(args.csv)
    with open(args.csv, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["method", "params", f"recall_at_{args.k}", "qps", "p50_us", "p99_us"])
        for ef in (int(x) for x in args.efs.split(",")):
            if ef < args.k:
                continue
            index.expansion_search = ef

            # throughput: all queries, all threads
            t0 = time.perf_counter()
            matches = index.search(queries, args.k, threads=args.threads)
            qps = len(queries) / (time.perf_counter() - t0)

            keys = np.asarray(matches.keys).reshape(len(queries), -1)
            hits = sum(
                len(set(keys[i].tolist()) & set(gt[i, : args.k].tolist()))
                for i in range(len(queries))
            )
            recall = hits / (len(queries) * args.k)

            # latency: single-threaded, one query at a time
            lat_us = []
            for q in queries[: args.latency_queries]:
                t0 = time.perf_counter()
                index.search(q.reshape(1, -1), args.k, threads=1)
                lat_us.append((time.perf_counter() - t0) * 1e6)
            p50, p99 = np.percentile(lat_us, [50, 99])

            w.writerow(
                ["usearch", f"M{args.M}-ef{ef}", f"{recall:.4f}", f"{qps:.0f}", f"{p50:.0f}", f"{p99:.0f}"]
            )
            print(
                f"ef={ef:4d}  recall@{args.k}={recall:.4f}  qps={qps:8.0f}  "
                f"p50={p50:6.0f}us  p99={p99:6.0f}us"
            )
    print(f"rows appended to {args.csv}")


if __name__ == "__main__":
    main()
