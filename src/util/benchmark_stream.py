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
    bytes_expected: int = 0
    error: str = None
    data_valid: bool = True  # True if size matches and data looks valid


def make_tile_key(chunk: ChunkRequest) -> str:
    return f"{chunk.x_start}-{chunk.x_end}_{chunk.y_start}-{chunk.y_end}_{chunk.z_start}-{chunk.z_end}"


def calculate_expected_size(chunk: ChunkRequest, num_channels: int = 1, bytes_per_voxel: int = 2) -> int:
    """Calculate expected response size for a chunk."""
    width = chunk.x_end - chunk.x_start
    height = chunk.y_end - chunk.y_start
    depth = chunk.z_end - chunk.z_start
    return width * height * depth * num_channels * bytes_per_voxel


def verify_data(data: bytes, expected_size: int) -> tuple[bool, str]:
    """Verify that received data is valid.

    Returns (is_valid, error_message)
    """
    if len(data) != expected_size:
        return False, f"size mismatch: got {len(data)}, expected {expected_size}"

    # Check if data is not all zeros (likely indicates error)
    if len(data) > 0:
        # Sample a few positions to check for non-zero values
        import struct
        sample_positions = [0, len(data)//4, len(data)//2, 3*len(data)//4, len(data)-2]
        all_zeros = True
        for pos in sample_positions:
            if pos + 2 <= len(data):
                val = struct.unpack('<H', data[pos:pos+2])[0]
                if val != 0:
                    all_zeros = False
                    break
        # Note: all zeros could be valid for some regions, so just warn
        # if all_zeros:
        #     return True, "warning: data appears to be all zeros"

    return True, ""


def fetch_chunk(base_url: str, dataset: str, chunk: ChunkRequest,
                use_stream: bool = True, verify: bool = True,
                num_channels: int = 1) -> RequestResult:
    """Fetch a single chunk and measure performance."""
    tile_key = make_tile_key(chunk)
    expected_size = calculate_expected_size(chunk, num_channels)

    if use_stream:
        url = f"{base_url}/{dataset}/stream/{chunk.scale}/{tile_key}"
    else:
        url = f"{base_url}/{dataset}/{chunk.scale}/{tile_key}"

    start = time.perf_counter()
    try:
        resp = requests.get(url, timeout=30)
        elapsed = time.perf_counter() - start

        data_valid = True
        error_msg = None

        if verify and resp.status_code == 200:
            data_valid, error_msg = verify_data(resp.content, expected_size)

        return RequestResult(
            chunk=chunk,
            status_code=resp.status_code,
            response_time=elapsed,
            bytes_received=len(resp.content),
            bytes_expected=expected_size,
            data_valid=data_valid,
            error=error_msg
        )
    except Exception as e:
        elapsed = time.perf_counter() - start
        return RequestResult(
            chunk=chunk,
            status_code=-1,
            response_time=elapsed,
            bytes_received=0,
            bytes_expected=expected_size,
            data_valid=False,
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


def print_stats(results: List[RequestResult], label: str, wall_time: float = None):
    """Print statistics for a batch of results."""
    if not results:
        print(f"  {label}: No results")
        return

    times = [r.response_time for r in results]
    success = sum(1 for r in results if r.status_code == 200 or r.status_code == 206)
    valid = sum(1 for r in results if r.data_valid)
    total_bytes = sum(r.bytes_received for r in results)
    total_expected = sum(r.bytes_expected for r in results)

    # Size mismatches
    size_mismatches = [(r, r.bytes_expected - r.bytes_received)
                       for r in results if r.bytes_received != r.bytes_expected and r.status_code == 200]

    print(f"\n  {label}:")
    print(f"    Requests: {len(results)} ({success} success, {len(results) - success} failed)")
    print(f"    Data validation: {valid}/{len(results)} valid")
    print(f"    Total bytes: {total_bytes:,} (expected: {total_expected:,})")

    if size_mismatches:
        print(f"    WARNING: {len(size_mismatches)} size mismatches!")
        for r, diff in size_mismatches[:3]:  # Show first 3
            print(f"      - {make_tile_key(r.chunk)}: got {r.bytes_received}, expected {r.bytes_expected} (diff: {diff})")

    print(f"    Response times:")
    print(f"      Min:    {min(times)*1000:.1f} ms")
    print(f"      Max:    {max(times)*1000:.1f} ms")
    print(f"      Mean:   {statistics.mean(times)*1000:.1f} ms")
    print(f"      Median: {statistics.median(times)*1000:.1f} ms")
    if len(times) > 1:
        print(f"      Stdev:  {statistics.stdev(times)*1000:.1f} ms")

    # Calculate throughput
    if wall_time is not None and wall_time > 0:
        # Use wall time for concurrent tests (more accurate)
        print(f"    Wall time: {wall_time:.2f} s")
        print(f"    Throughput (wall): {total_bytes / wall_time / 1024 / 1024:.2f} MB/s")
    else:
        # Sum of individual times (sequential)
        print(f"    Total time: {sum(times):.2f} s")
        if sum(times) > 0:
            print(f"    Throughput (sum): {total_bytes / sum(times) / 1024 / 1024:.2f} MB/s")


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
                   max_workers: int = 8, use_stream: bool = True) -> tuple[List[RequestResult], float]:
    """Test concurrent chunk fetching. Returns (results, wall_time)."""
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

    wall_time = time.perf_counter() - start_time
    return results, wall_time


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


def test_verify_data(base_url: str, dataset: str, chunks: List[ChunkRequest],
                     num_samples: int = 10) -> bool:
    """Verify data integrity by checking response sizes and content."""
    print(f"\n[TEST] Data verification ({num_samples} sample chunks)...")

    sample_chunks = chunks[:min(num_samples, len(chunks))]
    all_valid = True

    for i, chunk in enumerate(sample_chunks):
        result = fetch_chunk(base_url, dataset, chunk, use_stream=True, verify=True)

        status = "✓" if result.data_valid else "✗"
        size_match = result.bytes_received == result.bytes_expected

        print(f"  [{status}] Chunk {i+1}: {make_tile_key(chunk)}")
        print(f"      Status: {result.status_code}, "
              f"Size: {result.bytes_received:,} / {result.bytes_expected:,} bytes "
              f"({'match' if size_match else 'MISMATCH'})")

        if result.error:
            print(f"      Error: {result.error}")

        if not result.data_valid:
            all_valid = False

    print(f"\n  Result: {'ALL VALID' if all_valid else 'SOME INVALID'}")
    return all_valid


def test_scalability(base_url: str, dataset: str, chunks: List[ChunkRequest],
                     worker_counts: List[int] = None) -> dict:
    """Test throughput scalability with different concurrency levels."""
    if worker_counts is None:
        worker_counts = [1, 2, 4, 8, 16, 32, 64]

    print(f"\n[TEST] Scalability test ({len(chunks)} chunks, varying concurrency)...")
    print(f"  Testing worker counts: {worker_counts}")
    print()

    results_by_workers = {}

    for num_workers in worker_counts:
        print(f"  Testing {num_workers} workers...", end=" ", flush=True)

        start = time.perf_counter()
        results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(fetch_chunk, base_url, dataset, chunk, True)
                for chunk in chunks
            ]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        wall_time = time.perf_counter() - start
        success = sum(1 for r in results if r.status_code in (200, 206))
        valid = sum(1 for r in results if r.data_valid and r.status_code in (200, 206))
        failed = len(results) - success
        rejected = sum(1 for r in results if r.status_code == 503)
        total_bytes = sum(r.bytes_received for r in results)
        total_expected = sum(r.bytes_expected for r in results)
        times = [r.response_time for r in results]

        # Only count validated bytes for throughput
        validated_bytes = sum(r.bytes_received for r in results if r.data_valid)
        throughput_mbps = (validated_bytes / wall_time) / (1024 * 1024) if wall_time > 0 else 0
        requests_per_sec = len(results) / wall_time if wall_time > 0 else 0
        avg_latency = statistics.mean(times) * 1000
        p99_latency = sorted(times)[int(len(times) * 0.99)] * 1000 if len(times) > 1 else times[0] * 1000

        results_by_workers[num_workers] = {
            "wall_time": wall_time,
            "success": success,
            "valid": valid,
            "failed": failed,
            "rejected": rejected,
            "total_bytes": total_bytes,
            "total_expected": total_expected,
            "validated_bytes": validated_bytes,
            "throughput_mbps": throughput_mbps,
            "requests_per_sec": requests_per_sec,
            "avg_latency_ms": avg_latency,
            "p99_latency_ms": p99_latency,
        }

        size_ok = "✓" if total_bytes == total_expected else f"✗ ({total_bytes}/{total_expected})"
        print(f"done in {wall_time:.1f}s | "
              f"{throughput_mbps:.1f} MB/s | "
              f"{requests_per_sec:.1f} req/s | "
              f"latency: {avg_latency:.0f}ms avg, {p99_latency:.0f}ms p99 | "
              f"{valid}/{len(results)} valid | size: {size_ok}" +
              (f" ({rejected} rejected)" if rejected > 0 else ""))

    # Print summary table
    print("\n  " + "=" * 100)
    print(f"  {'Workers':>8} | {'Throughput':>12} | {'Requests/s':>10} | {'Avg Lat':>10} | {'P99 Lat':>10} | {'Valid':>10} | {'Size Match':>10}")
    print("  " + "-" * 100)

    best_throughput = max(r["throughput_mbps"] for r in results_by_workers.values())

    for workers, stats in sorted(results_by_workers.items()):
        marker = " *" if stats["throughput_mbps"] == best_throughput else ""
        size_match = "✓" if stats["total_bytes"] == stats["total_expected"] else "✗"
        print(f"  {workers:>8} | {stats['throughput_mbps']:>9.1f} MB/s | {stats['requests_per_sec']:>10.1f} | "
              f"{stats['avg_latency_ms']:>8.0f}ms | {stats['p99_latency_ms']:>8.0f}ms | "
              f"{stats['valid']:>4}/{len(chunks):<4} | {size_match:>10}{marker}")

    print("  " + "=" * 90)

    # Find optimal
    optimal_workers = max(results_by_workers.keys(), key=lambda w: results_by_workers[w]["throughput_mbps"])
    print(f"\n  Optimal concurrency: {optimal_workers} workers "
          f"({results_by_workers[optimal_workers]['throughput_mbps']:.1f} MB/s)")

    # Check for saturation/bottleneck
    if len(worker_counts) >= 2:
        last_two = sorted(results_by_workers.keys())[-2:]
        if len(last_two) == 2:
            t1 = results_by_workers[last_two[0]]["throughput_mbps"]
            t2 = results_by_workers[last_two[1]]["throughput_mbps"]
            if t2 <= t1 * 1.05:  # Less than 5% improvement
                print(f"  Note: Throughput saturated at ~{last_two[0]} workers")

    return results_by_workers


def test_sustained_load(base_url: str, dataset: str, chunks: List[ChunkRequest],
                        duration_sec: int = 60, max_workers: int = 16) -> dict:
    """Test sustained load over time to detect memory leaks or degradation."""
    print(f"\n[TEST] Sustained load test ({duration_sec}s, {max_workers} workers)...")

    start_time = time.perf_counter()
    interval = 10  # Report every 10 seconds
    next_report = interval

    all_results = []
    interval_results = []
    chunk_idx = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        pending_futures = set()

        while True:
            elapsed = time.perf_counter() - start_time

            # Check if we should stop
            if elapsed >= duration_sec and len(pending_futures) == 0:
                break

            # Submit new work if under duration and have capacity
            while elapsed < duration_sec and len(pending_futures) < max_workers * 2:
                chunk = chunks[chunk_idx % len(chunks)]
                chunk_idx += 1
                future = executor.submit(fetch_chunk, base_url, dataset, chunk, True)
                pending_futures.add(future)

            # Collect completed futures
            done, pending_futures = concurrent.futures.wait(
                pending_futures,
                timeout=0.1,
                return_when=concurrent.futures.FIRST_COMPLETED
            )

            for future in done:
                result = future.result()
                all_results.append(result)
                interval_results.append(result)

            # Report progress
            elapsed = time.perf_counter() - start_time
            if elapsed >= next_report:
                if interval_results:
                    success = sum(1 for r in interval_results if r.status_code in (200, 206))
                    rejected = sum(1 for r in interval_results if r.status_code == 503)
                    total_bytes = sum(r.bytes_received for r in interval_results)
                    avg_time = statistics.mean(r.response_time for r in interval_results)
                    throughput = total_bytes / interval / (1024 * 1024)

                    print(f"  t={int(elapsed):>3}s | "
                          f"{len(interval_results):>4} reqs | "
                          f"{throughput:>6.1f} MB/s | "
                          f"lat: {avg_time*1000:>5.0f}ms | "
                          f"{success}/{len(interval_results)} ok" +
                          (f" ({rejected} rejected)" if rejected > 0 else ""))

                interval_results = []
                next_report = elapsed + interval

    # Final stats
    total_time = time.perf_counter() - start_time
    success = sum(1 for r in all_results if r.status_code in (200, 206))
    rejected = sum(1 for r in all_results if r.status_code == 503)
    total_bytes = sum(r.bytes_received for r in all_results)

    print(f"\n  Summary:")
    print(f"    Total requests: {len(all_results)}")
    print(f"    Success: {success} ({100*success/len(all_results):.1f}%)")
    print(f"    Rejected (503): {rejected}")
    print(f"    Total bytes: {total_bytes:,}")
    print(f"    Duration: {total_time:.1f}s")
    print(f"    Avg throughput: {total_bytes/total_time/1024/1024:.1f} MB/s")
    print(f"    Avg requests/s: {len(all_results)/total_time:.1f}")

    # Check for degradation
    if len(all_results) >= 20:
        first_10_pct = all_results[:len(all_results)//10]
        last_10_pct = all_results[-len(all_results)//10:]
        first_avg = statistics.mean(r.response_time for r in first_10_pct)
        last_avg = statistics.mean(r.response_time for r in last_10_pct)
        change = (last_avg - first_avg) / first_avg * 100
        print(f"    Latency trend: {change:+.1f}% (first 10%: {first_avg*1000:.0f}ms, last 10%: {last_avg*1000:.0f}ms)")

    return {
        "total_requests": len(all_results),
        "success": success,
        "rejected": rejected,
        "total_bytes": total_bytes,
        "duration": total_time,
        "throughput_mbps": total_bytes / total_time / 1024 / 1024,
        "requests_per_sec": len(all_results) / total_time,
    }


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
    parser.add_argument("--test", choices=["all", "seq", "concurrent", "degradation", "range",
                                           "concurrent-degrade", "scalability", "sustained", "verify"],
                       default="all", help="Test to run")
    parser.add_argument("--duration", type=int, default=60, help="Duration for sustained load test (seconds)")

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
        results, wall_time = test_concurrent(args.url, args.dataset, chunks, max_workers=args.workers)
        print_stats(results, f"Concurrent ({args.workers} workers)", wall_time=wall_time)

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

    if args.test == "scalability":
        # Use more chunks for scalability test
        test_scalability(args.url, args.dataset, chunks[:min(200, len(chunks))])

    if args.test == "sustained":
        test_sustained_load(args.url, args.dataset, chunks,
                           duration_sec=args.duration, max_workers=args.workers)

    if args.test == "verify":
        test_verify_data(args.url, args.dataset, chunks, num_samples=20)

    print("\n" + "=" * 50)
    print("Tests complete!")


if __name__ == "__main__":
    main()
