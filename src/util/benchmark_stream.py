#!/usr/bin/env python3
"""
Benchmark script for the SISF CDN stream endpoint.

Usage:
    python benchmark_stream.py --host localhost:7000 --dataset my_dataset --scale 1

Options:
    --host          Server host:port (default: localhost:7000)
    --dataset       Dataset ID to benchmark
    --scale         Resolution scale (default: 1)
    --sizes         Comma-separated region sizes to test (default: 64,128,256,512)
    --iterations    Number of iterations per size (default: 5)
    --concurrent    Number of concurrent requests (default: 1)
    --range-test    Test HTTP Range requests (partial downloads)
"""

import argparse
import time
import statistics
import concurrent.futures
import urllib.request
import urllib.error
import json
import sys


def format_bytes(size):
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"


def format_time(seconds):
    """Format time in appropriate units."""
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.2f} us"
    elif seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    else:
        return f"{seconds:.2f} s"


def fetch_stream(url, range_header=None):
    """Fetch data from stream endpoint and return timing info."""
    req = urllib.request.Request(url)
    if range_header:
        req.add_header('Range', range_header)

    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            first_byte_time = time.perf_counter()
            data = response.read()
            end_time = time.perf_counter()

            return {
                'success': True,
                'total_time': end_time - start,
                'ttfb': first_byte_time - start,  # Time to first byte
                'download_time': end_time - first_byte_time,
                'size': len(data),
                'status': response.status,
            }
    except urllib.error.HTTPError as e:
        return {
            'success': False,
            'error': f"HTTP {e.code}: {e.reason}",
            'total_time': time.perf_counter() - start,
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'total_time': time.perf_counter() - start,
        }


def get_dataset_info(host, dataset):
    """Get dataset info to determine valid ranges."""
    url = f"http://{host}/{dataset}/info"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return json.loads(response.read())
    except:
        return None


def run_benchmark(host, dataset, scale, size, z_size, iterations, concurrent_requests):
    """Run benchmark for a specific region size."""
    # Build URL for stream endpoint
    # Format: /<data_id>/stream/<scale>/<x_begin>-<x_end>_<y_begin>-<y_end>_<z_begin>-<z_end>
    url = f"http://{host}/{dataset}/stream/{scale}/0-{size}_0-{size}_0-{z_size}"

    results = []

    if concurrent_requests == 1:
        # Sequential requests
        for i in range(iterations):
            result = fetch_stream(url)
            results.append(result)
            if not result['success']:
                print(f"  Warning: Request failed - {result.get('error', 'Unknown error')}")
    else:
        # Concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            for i in range(iterations):
                futures = [executor.submit(fetch_stream, url) for _ in range(concurrent_requests)]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    results.append(result)
                    if not result['success']:
                        print(f"  Warning: Request failed - {result.get('error', 'Unknown error')}")

    return results


def run_range_benchmark(host, dataset, scale, size, z_size, iterations):
    """Benchmark HTTP Range requests (partial downloads)."""
    # First, get total size
    url = f"http://{host}/{dataset}/stream/{scale}/0-{size}_0-{size}_0-{z_size}"

    # Fetch with range for first 10%, middle 10%, last 10%
    # Assuming uint16 data: size * size * z_size * 2 bytes
    total_bytes = size * size * z_size * 2
    chunk_size = total_bytes // 10

    ranges = [
        ('first_10%', f'bytes=0-{chunk_size-1}'),
        ('middle_10%', f'bytes={total_bytes//2}-{total_bytes//2 + chunk_size-1}'),
        ('last_10%', f'bytes={total_bytes - chunk_size}-{total_bytes-1}'),
    ]

    results = {}
    for name, range_header in ranges:
        times = []
        for _ in range(iterations):
            result = fetch_stream(url, range_header)
            if result['success']:
                times.append(result['total_time'])

        if times:
            results[name] = {
                'mean': statistics.mean(times),
                'std': statistics.stdev(times) if len(times) > 1 else 0,
                'min': min(times),
                'max': max(times),
            }

    return results


def print_stats(results, size, z_size):
    """Print statistics for benchmark results."""
    successful = [r for r in results if r['success']]

    if not successful:
        print(f"  {size}x{size}x{z_size}: All requests failed!")
        return

    times = [r['total_time'] for r in successful]
    ttfbs = [r['ttfb'] for r in successful]
    sizes = [r['size'] for r in successful]

    avg_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0
    avg_ttfb = statistics.mean(ttfbs)
    avg_size = statistics.mean(sizes)
    throughput = avg_size / avg_time if avg_time > 0 else 0

    print(f"  {size}x{size}x{z_size}:")
    print(f"    Total time:    {format_time(avg_time)} ± {format_time(std_time)}")
    print(f"    TTFB:          {format_time(avg_ttfb)}")
    print(f"    Data size:     {format_bytes(avg_size)}")
    print(f"    Throughput:    {format_bytes(throughput)}/s")
    print(f"    Success rate:  {len(successful)}/{len(results)}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark SISF CDN stream endpoint')
    parser.add_argument('--host', default='localhost:7000', help='Server host:port')
    parser.add_argument('--dataset', required=True, help='Dataset ID to benchmark')
    parser.add_argument('--scale', type=int, default=1, help='Resolution scale')
    parser.add_argument('--sizes', default='64,128,256,512',
                        help='Comma-separated XY region sizes to test')
    parser.add_argument('--z-size', type=int, default=32, help='Z dimension size')
    parser.add_argument('--iterations', type=int, default=5, help='Iterations per size')
    parser.add_argument('--concurrent', type=int, default=1, help='Concurrent requests')
    parser.add_argument('--range-test', action='store_true', help='Test HTTP Range requests')
    parser.add_argument('--warmup', type=int, default=1, help='Warmup iterations (not counted)')

    args = parser.parse_args()

    sizes = [int(s.strip()) for s in args.sizes.split(',')]

    print("=" * 60)
    print("SISF CDN Stream Endpoint Benchmark")
    print("=" * 60)
    print(f"Host:        {args.host}")
    print(f"Dataset:     {args.dataset}")
    print(f"Scale:       {args.scale}")
    print(f"Z-size:      {args.z_size}")
    print(f"Sizes:       {sizes}")
    print(f"Iterations:  {args.iterations}")
    print(f"Concurrent:  {args.concurrent}")
    print("=" * 60)

    # Try to get dataset info
    info = get_dataset_info(args.host, args.dataset)
    if info:
        print(f"Dataset info: {json.dumps(info, indent=2)[:200]}...")
    else:
        print("Could not fetch dataset info (continuing anyway)")

    print()

    # Warmup
    if args.warmup > 0:
        print(f"Warming up ({args.warmup} iterations)...")
        run_benchmark(args.host, args.dataset, args.scale, sizes[0], args.z_size,
                     args.warmup, args.concurrent)
        print()

    # Main benchmark
    print("Running benchmark...")
    all_results = {}

    for size in sizes:
        results = run_benchmark(args.host, args.dataset, args.scale, size, args.z_size,
                               args.iterations, args.concurrent)
        all_results[size] = results
        print_stats(results, size, args.z_size)
        print()

    # Range test
    if args.range_test:
        print("=" * 60)
        print("HTTP Range Request Benchmark")
        print("=" * 60)

        for size in sizes:
            print(f"\n  {size}x{size}x{args.z_size}:")
            range_results = run_range_benchmark(args.host, args.dataset, args.scale,
                                                size, args.z_size, args.iterations)
            for name, stats in range_results.items():
                print(f"    {name}: {format_time(stats['mean'])} ± {format_time(stats['std'])}")

    # Summary table
    print("=" * 60)
    print("Summary (avg total time)")
    print("=" * 60)
    print(f"{'Size':<15} {'Time':<15} {'Throughput':<15}")
    print("-" * 45)

    for size in sizes:
        successful = [r for r in all_results[size] if r['success']]
        if successful:
            avg_time = statistics.mean([r['total_time'] for r in successful])
            avg_size = statistics.mean([r['size'] for r in successful])
            throughput = avg_size / avg_time if avg_time > 0 else 0
            print(f"{size}x{size}x{args.z_size:<6} {format_time(avg_time):<15} {format_bytes(throughput)}/s")
        else:
            print(f"{size}x{size}x{args.z_size:<6} FAILED")

    print("=" * 60)


if __name__ == '__main__':
    main()
