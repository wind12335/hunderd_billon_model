"""通信微基准：标定 comm_model 的机内/机间带宽、时延与每卡算力常数。

在集群 worker 上由 run_dual_node_test.sh（BENCH_COMM=1）经 torchrun 拉起：
  机内：单节点 8 卡    → torchrun --nproc_per_node 8 .../bench_comm_hygon.py
  跨机：双机 16 卡     → torchrun --nnodes 2 --nproc_per_node 8 ...（自动）
输出 CSV（rank0 写 --out，所有 rank 打印同表）：
  op, world, size_mb, p50_us, algbw_GBs, busbw_GBs[, tflops]
口径：all-reduce busbw = 2(n-1)/n × M / t；a2a 按每卡实发 M(n-1)/n / t；
时延项由最小尺寸的 p50 近似。回填方式见 megatron_mvp/types.py 拓扑注释。

本地逻辑冒烟（不碰 DCU）：
  torchrun --nproc_per_node 2 --master_port 29777 tools/bench_comm_hygon.py \
      --backend gloo --device cpu --sizes 1,4 --skip-matmul
"""

from __future__ import annotations

import argparse
import csv
import datetime
import os
import statistics
import time

import torch
import torch.distributed as dist


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BW1000 通信微基准")
    parser.add_argument("--backend", default="nccl", help="nccl(集群) 或 gloo(本地冒烟)")
    parser.add_argument("--device", default=None, help="cuda(集群) 或 cpu(本地冒烟)")
    parser.add_argument("--sizes", default="1,4,16,33.6,64,256", help="每卡消息 MB 阶梯(逗号分隔)")
    parser.add_argument("--iters", type=int, default=20, help="每个尺寸的计时迭代数")
    parser.add_argument("--warmup", type=int, default=5, help="预热迭代数")
    parser.add_argument("--skip-matmul", action="store_true", help="跳过 matmul TFLOPS(冒烟用)")
    parser.add_argument("--out", default="", help="CSV 输出路径(仅 rank0)")
    return parser.parse_args()


def _timed(fn, iters: int, warmup: int, sync) -> list[float]:
    """返回每次调用的毫秒耗时（事件计时，逐次同步）。"""
    durations_ms: list[float] = []
    for i in range(warmup + iters):
        sync()
        start = time.perf_counter()
        fn()
        sync()
        if i >= warmup:
            durations_ms.append((time.perf_counter() - start) * 1e3)
    return durations_ms


def bench_allreduce(x: torch.Tensor, world: int, args) -> dict:
    """all-reduce：busbw = 2(n-1)/n × M / t。"""
    times = _timed(lambda: dist.all_reduce(x, op=dist.ReduceOp.SUM), args.iters, args.warmup, torch.cuda.synchronize)
    p50 = statistics.median(times) * 1e3  # µs
    mbytes = x.numel() * x.element_size() / 1024**2
    algbw = mbytes / 1024 / (p50 * 1e-6)
    busbw = algbw * 2 * (world - 1) / world
    return {"op": "allreduce", "size_mb": round(mbytes, 1), "p50_us": round(p50, 1),
            "algbw_GBs": round(algbw, 1), "busbw_GBs": round(busbw, 1)}


def bench_alltoall(x: torch.Tensor, world: int, args) -> dict:
    """a2a（等分 split）：每卡实发 ≈ M(n-1)/n。"""
    times = _timed(lambda: dist.all_to_all_single(x, x), args.iters, args.warmup, torch.cuda.synchronize)
    p50 = statistics.median(times) * 1e3
    mbytes = x.numel() * x.element_size() / 1024**2
    sent = mbytes * (world - 1) / world / 1024
    return {"op": "a2a", "size_mb": round(mbytes, 1), "p50_us": round(p50, 1),
            "algbw_GBs": round(mbytes / 1024 / (p50 * 1e-6), 1), "busbw_GBs": round(sent / (p50 * 1e-6), 1)}


def bench_matmul(device: str, args) -> dict:
    """bf16 方阵乘法 → 每卡 TFLOPS（标定 comm_model 的计算常数）。"""
    n = 8192
    a = torch.randn(n, n, dtype=torch.bfloat16, device=device)
    b = torch.randn(n, n, dtype=torch.bfloat16, device=device)
    flops_per_iter = 2.0 * n**3
    times = _timed(lambda: torch.matmul(a, b), min(args.iters, 10), 3, torch.cuda.synchronize)
    tflops = flops_per_iter / (statistics.median(times) * 1e-3) / 1e12
    return {"op": "matmul", "size_mb": n, "p50_us": round(statistics.median(times) * 1e3, 0),
            "algbw_GBs": "", "busbw_GBs": "", "tflops": round(tflops, 1)}


def main() -> None:
    args = _parse_args()
    # 显式进程组超时：一侧死亡时，本侧挂住的集合通信默认要等 30 分钟 watchdog
    # ——会吃掉训练 rendezvous 窗口。300s 自毙（入口脚本可经 BENCH_TIMEOUT_SEC 注入）
    dist.init_process_group(
        args.backend,
        timeout=datetime.timedelta(seconds=int(os.environ.get("BENCH_TIMEOUT_SEC", "300"))),
    )
    rank = dist.get_rank()
    world = dist.get_world_size()
    device = args.device or ("cuda" if args.backend == "nccl" else "cpu")
    torch.cuda.set_device(rank % max(torch.cuda.device_count(), 1)) if device == "cuda" else None

    rows = []
    for size_mb in [float(s) for s in args.sizes.split(",")]:
        numel = int(size_mb * 1024**2 / 2)  # bf16
        x = torch.randn(max(numel // world, 1), dtype=torch.bfloat16, device=device)
        rows.append(bench_allreduce(x, world, args))
        # a2a 沿 dim0 等分：张量总长须为 world 的倍数
        n2 = max(numel // world * world, world)
        y = torch.randn(n2, dtype=torch.bfloat16, device=device)
        rows.append(bench_alltoall(y, world, args))
        if rank == 0:
            last = rows[-2:]
            print(f"[bench] world={world} dev={device} "
                  + " | ".join(f"{r['op']} {r['size_mb']}MB: {r['p50_us']}µs busbw={r['busbw_GBs']}GB/s" for r in last))
    if not args.skip_matmul and device == "cuda":
        rows.append(bench_matmul(device, args))
        if rank == 0:
            print(f"[bench] matmul 8192³ bf16: {rows[-1]['tflops']} TFLOPS")

    if rank == 0 and args.out:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["op", "size_mb", "p50_us", "algbw_GBs", "busbw_GBs", "tflops"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"[bench] CSV → {args.out}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
