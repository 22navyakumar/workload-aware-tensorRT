"""Shared latency benchmark: warmed-up median / p95 latency, throughput, peak memory."""
import statistics
import torch


def benchmark(fn, inputs, warmup=20, iters=100):
    """Time `fn(*inputs)` with CUDA events. Returns a dict of metrics in ms."""
    with torch.inference_mode():
        for _ in range(warmup):
            fn(*inputs)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn(*inputs)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    times.sort()
    batch = inputs[0].shape[0]
    median = statistics.median(times)
    return {
        "median_ms": median,
        "p95_ms": times[int(0.95 * (len(times) - 1))],
        "throughput_seq_per_s": batch / (median / 1000),
        "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20,
    }
