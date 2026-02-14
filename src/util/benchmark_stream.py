#!/usr/bin/env python3
"""
Stream and Concurrency Test for SISF CDN

Tests:
1. Sequential chunk fetching (baseline)
2. Concurrent chunk fetching (parallel requests)
3. Large region streaming with Range headers
4. Performance degradation over time
"""

import requests
import time
import argparse
import concurrent.futures
from dataclasses import dataclass
from typing import List, Tuple
import statistics
import sys


@dataclass
class ChunkRequest:
    x_start: int
    x_end: int
    y_start: int
    y_end: int
    z_start: int
    z_end: int
    scale: int = 1


@dataclass
class RequestResult:
    chunk: ChunkRequest
    status_code: int
    response_time: float  # seconds
    bytes_received: int
    error: str = None


def make_tile_key(chunk: ChunkRequest) -> str:
    return f"{chunk.x_start}-{chunk.x_end}_{chunk.y_start}-{chunk.y_end}_{chunk.z_start}-{chunk.z_end}"


def fetch_chunk(base_url: str, dataset: str, chunk: ChunkRequest, use_stream: bool = True) -> RequestResult:
    """Fetch a single chunk and measure performance."""
    endpoint = "stream" if use_stream else ""
    tile_key = make_tile_key(chunk)

    if use_stream:
        url = f"{base_url}/{dataset}/stream/{chunk.scale}/{tile_key}"
    else:
        url = f"{base_url}/{dataset}/{chunk.scale}/{tile_key}"

    start = time.perf_counter()
    try:
        resp = requests.get(url, timeout=30)
        elapsed = time.perf_counter() - start
        return RequestResult(
            chunk=chunk,
            status_code=resp.status_code,
            response_time=elapsed,
            bytes_received=len(resp.content)
        )
    except Exception as e:
        elapsed = time.perf_counter() - start
        return RequestResult(
            chunk=chunk,
            status_code=-1,
            response_time=elapsed,
            bytes_received=0,
            error=str(e)
        )


def fetch_chunk_with_range(base_url: str, dataset: str, chunk: ChunkRequest,
                            range_start: int, range_end: int) -> RequestResult:
    """Fetch a chunk with HTTP Range header."""
    tile_key = make_tile_key(chunk)
    url = f"{base_url}/{dataset}/stream/{chunk.scale}/{tile_key}"

    headers = {"Range": f"bytes={range_start}-{range_end}"}

    start = time.perf_counter()
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        elapsed = time.perf_counter() - start
        return RequestResult(
            chunk=chunk,
            status_code=resp.status_code,
            response_time=elapsed,
            bytes_received=len(resp.content)
        )
    except Exception as e:
        elapsed = time.perf_counter() - start
        return RequestResult(
            chunk=chunk,
            status_code=-1,
            response_time=elapsed,
            bytes_received=0,
            error=str(e)
        )


def divide_region_into_chunks(
    x_start: int, x_end: int,
    y_start: int, y_end: int,
    z_start: int, z_end: int,
    chunk_x: int = 256,
    chunk_y: int = 256,
    chunk_z: int = 32,
    scale: int = 1
) -> List[ChunkRequest]:
    """Divide a large region into smaller chunks (default 256x256x32)."""
    chunks = []

    for x in range(x_start, x_end, chunk_x):
        for y in range(y_start, y_end, chunk_y):
            for z in range(z_start, z_end, chunk_z):
                chunks.append(ChunkRequest(
                    x_start=x,
                    x_end=min(x + chunk_x, x_end),
                    y_start=y,
                    y_end=min(y + chunk_y, y_end),
                    z_start=z,
                    z_end=min(z + chunk_z, z_end),
                    scale=scale
                ))

    return chunks


def print_stats(results: List[RequestResult], label: str):
    """Print statistics for a batch of results."""
    if not results:
        print(f"  {label}: No results")
        return

    times = [r.response_time for r in results]
    success = sum(1 for r in results if r.status_code == 200 or r.status_code == 206)
    total_bytes = sum(r.bytes_received for r in results)

    print(f"\n  {label}:")
    print(f"    Requests: {len(results)} ({success} success, {len(results) - success} failed)")
    print(f"    Total bytes: {total_bytes:,}")
    print(f"    Response times:")
    print(f"      Min:    {min(times)*1000:.1f} ms")
    print(f"      Max:    {max(times)*1000:.1f} ms")
    print(f"      Mean:   {statistics.mean(times)*1000:.1f} ms")
    print(f"      Median: {statistics.median(times)*1000:.1f} ms")
    if len(times) > 1:
        print(f"      Stdev:  {statistics.stdev(times)*1000:.1f} ms")
    print(f"    Total time: {sum(times):.2f} s")
    if sum(times) > 0:
        print(f"    Throughput: {total_bytes / sum(times) / 1024 / 1024:.2f} MB/s")


def test_sequential(base_url: str, dataset: str, chunks: List[ChunkRequest], use_stream: bool = True) -> List[RequestResult]:
    """Test sequential chunk fetching (baseline)."""
    print(f"\n[TEST] Sequential fetching ({len(chunks)} chunks)...")

    results = []
    for i, chunk in enumerate(chunks):
        result = fetch_chunk(base_url, dataset, chunk, use_stream)
        results.append(result)

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(chunks)} chunks, last={result.response_time*1000:.1f}ms")

    return results


def test_concurrent(base_url: str, dataset: str, chunks: List[ChunkRequest],
                   max_workers: int = 8, use_stream: bool = True) -> List[RequestResult]:
    """Test concurrent chunk fetching."""
    print(f"\n[TEST] Concurrent fetching ({len(chunks)} chunks, {max_workers} workers)...")

    start_time = time.perf_counter()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(fetch_chunk, base_url, dataset, chunk, use_stream)
            for chunk in chunks
        ]

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            results.append(result)

            if (i + 1) % 10 == 0:
                elapsed = time.perf_counter() - start_time
                print(f"  Progress: {i+1}/{len(chunks)} chunks in {elapsed:.1f}s")

    return results


def test_degradation(base_url: str, dataset: str, chunk: ChunkRequest,
                     iterations: int = 100, use_stream: bool = True) -> List[RequestResult]:
    """Test for performance degradation over time."""
    print(f"\n[TEST] Degradation test ({iterations} iterations of same chunk)...")

    results = []
    batch_size = 10

    for i in range(iterations):
        result = fetch_chunk(base_url, dataset, chunk, use_stream)
        results.append(result)

        # Print batch stats every batch_size requests
        if (i + 1) % batch_size == 0:
            batch_results = results[-batch_size:]
            avg_time = statistics.mean(r.response_time for r in batch_results)
            print(f"  Iteration {i+1}: avg response time = {avg_time*1000:.1f} ms")

    return results


def test_range_requests(base_url: str, dataset: str, chunk: ChunkRequest,
                        num_ranges: int = 10) -> List[RequestResult]:
    """Test HTTP Range header support."""
    print(f"\n[TEST] Range requests ({num_ranges} range fetches)...")

    # First, get total size
    result = fetch_chunk(base_url, dataset, chunk, use_stream=True)
    if result.status_code != 200:
        print(f"  Error: Failed to fetch full chunk (status {result.status_code})")
        return []

    total_size = result.bytes_received
    range_size = total_size // num_ranges

    print(f"  Total size: {total_size:,} bytes, range size: {range_size:,} bytes")

    results = []
    for i in range(num_ranges):
        range_start = i * range_size
        range_end = min((i + 1) * range_size - 1, total_size - 1)

        result = fetch_chunk_with_range(base_url, dataset, chunk, range_start, range_end)
        results.append(result)

        expected_size = range_end - range_start + 1
        if result.bytes_received != expected_size:
            print(f"  Warning: Range {i} expected {expected_size} bytes, got {result.bytes_received}")

    return results


def test_concurrent_degradation(base_url: str, dataset: str, chunks: List[ChunkRequest],
                                 rounds: int = 5, max_workers: int = 8) -> None:
    """Test if concurrent requests cause degradation over multiple rounds."""
    print(f"\n[TEST] Concurrent degradation ({rounds} rounds, {max_workers} workers)...")

    for round_num in range(rounds):
        start = time.perf_counter()

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(fetch_chunk, base_url, dataset, chunk, True)
                for chunk in chunks
            ]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        elapsed = time.perf_counter() - start
        success = sum(1 for r in results if r.status_code in (200, 206))
        avg_time = statistics.mean(r.response_time for r in results)

        print(f"  Round {round_num + 1}: {len(chunks)} requests in {elapsed:.2f}s, "
              f"avg response {avg_time*1000:.1f}ms, {success}/{len(chunks)} success")


def main():
    parser = argparse.ArgumentParser(description="SISF CDN Stream Test")
    parser.add_argument("--url", default="http://localhost:7000", help="CDN base URL")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--scale", type=int, default=1, help="Scale factor")
    parser.add_argument("--region", default="0-512_0-512_0-64",
                       help="Region to test (x0-x1_y0-y1_z0-z1)")
    parser.add_argument("--chunk-x", type=int, default=256, help="Chunk size X (default 256)")
    parser.add_argument("--chunk-y", type=int, default=256, help="Chunk size Y (default 256)")
    parser.add_argument("--chunk-z", type=int, default=32, help="Chunk size Z (default 32)")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent workers")
    parser.add_argument("--test", choices=["all", "seq", "concurrent", "degradation", "range", "concurrent-degrade"],
                       default="all", help="Test to run")

    args = parser.parse_args()

    # Parse region
    parts = args.region.split("_")
    x_parts = parts[0].split("-")
    y_parts = parts[1].split("-")
    z_parts = parts[2].split("-")

    x_start, x_end = int(x_parts[0]), int(x_parts[1])
    y_start, y_end = int(y_parts[0]), int(y_parts[1])
    z_start, z_end = int(z_parts[0]), int(z_parts[1])

    print(f"SISF CDN Stream Test")
    print(f"=" * 50)
    print(f"URL: {args.url}")
    print(f"Dataset: {args.dataset}")
    print(f"Region: ({x_start}-{x_end}, {y_start}-{y_end}, {z_start}-{z_end})")
    print(f"Scale: {args.scale}")
    print(f"Chunk size: {args.chunk_x}x{args.chunk_y}x{args.chunk_z}")
    print(f"Workers: {args.workers}")

    # Divide region into chunks
    chunks = divide_region_into_chunks(
        x_start, x_end, y_start, y_end, z_start, z_end,
        chunk_x=args.chunk_x, chunk_y=args.chunk_y, chunk_z=args.chunk_z,
        scale=args.scale
    )
    print(f"Total chunks: {len(chunks)}")

    # Single chunk for some tests
    single_chunk = ChunkRequest(
        x_start=x_start, x_end=min(x_start + args.chunk_x, x_end),
        y_start=y_start, y_end=min(y_start + args.chunk_y, y_end),
        z_start=z_start, z_end=min(z_start + args.chunk_z, z_end),
        scale=args.scale
    )

    # Run tests
    if args.test in ("all", "seq"):
        results = test_sequential(args.url, args.dataset, chunks[:min(20, len(chunks))])
        print_stats(results, "Sequential")

    if args.test in ("all", "concurrent"):
        results = test_concurrent(args.url, args.dataset, chunks, max_workers=args.workers)
        print_stats(results, f"Concurrent ({args.workers} workers)")

    if args.test in ("all", "degradation"):
        results = test_degradation(args.url, args.dataset, single_chunk, iterations=50)
        print_stats(results, "Degradation (single chunk repeated)")

        # Check for degradation trend
        first_10 = [r.response_time for r in results[:10]]
        last_10 = [r.response_time for r in results[-10:]]
        print(f"    First 10 avg: {statistics.mean(first_10)*1000:.1f} ms")
        print(f"    Last 10 avg:  {statistics.mean(last_10)*1000:.1f} ms")
        degradation = (statistics.mean(last_10) - statistics.mean(first_10)) / statistics.mean(first_10) * 100
        print(f"    Degradation:  {degradation:+.1f}%")

    if args.test in ("all", "range"):
        results = test_range_requests(args.url, args.dataset, single_chunk)
        print_stats(results, "Range requests")

    if args.test in ("all", "concurrent-degrade"):
        test_concurrent_degradation(args.url, args.dataset, chunks[:min(50, len(chunks))],
                                    rounds=5, max_workers=args.workers)

    print("\n" + "=" * 50)
    print("Tests complete!")


if __name__ == "__main__":
    main()
