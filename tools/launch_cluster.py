#!/usr/bin/env python3
"""基于 SSH + hostfile 的多机一键拉起器。

定位
----
autoparallel 生成的启动脚本（launch_megatron_plan.sh / launch_pytorch_plan.sh）
已经具备多机语义：读取 NNODES / NODE_RANK / MASTER_ADDR / MASTER_PORT /
GPUS_PER_NODE 环境变量。本工具补上"集群编排"这最后一环：

    hostfile（每行一个节点） -> 逐节点 SSH 注入 NODE_RANK 等环境变量
                              -> 并发拉起同一份启动脚本
                              -> 聚合各节点日志（带节点前缀）

与 Megatron-LM-das 的 examples/GLM/run_glm5_300B.sh（mpirun + hostfile slots=8）
等价的流程，但只依赖标准 SSH 免密，不依赖 MPI。

hostfile 格式（与 das 仓库一致，向后兼容 slots 字段）::

    # 注释行
    node-01 slots=8        # 第一行 = 主节点 (NODE_RANK=0, MASTER_ADDR)
    node-02:36000 slots=8  # 支持 host:ssh_port 指定非默认 SSH 端口
    node-03 slots=8

用法::

    python3 tools/launch_cluster.py \
        --hostfile generated/autoparallel/hostfile \
        --script generated/autoparallel/launch_megatron_plan.sh \
        --gpus-per-node 8

    # 只打印命令不执行:
    python3 tools/launch_cluster.py --hostfile ... --script ... --dry-run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HostEntry:
    """描述 hostfile 中的一个节点。"""

    host: str
    ssh_port: int
    slots: int
    line_number: int


def parse_hostfile(path: Path) -> list[HostEntry]:
    """解析 hostfile，返回按行序排列的节点列表。"""
    if not path.is_file():
        raise FileNotFoundError(
            f"未找到 hostfile: {path}\n"
            "提示: autoparallel 在 num_nodes>1 时会自动生成 hostfile 模板，"
            "请先把其中的 localhost 替换成真实节点名。"
        )

    entries: list[HostEntry] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue

        fields = line.split()
        host_field = fields[0]
        if ":" in host_field:
            host, port_text = host_field.rsplit(":", 1)
            ssh_port = int(port_text)
        else:
            host, ssh_port = host_field, 22

        slots = 8
        for extra in fields[1:]:
            if extra.startswith("slots="):
                slots = int(extra.split("=", 1)[1])

        entries.append(
            HostEntry(host=host, ssh_port=ssh_port, slots=slots, line_number=line_number)
        )

    if not entries:
        raise ValueError(f"hostfile {path} 中没有有效节点条目。")
    return entries


def build_remote_command(
    entry: HostEntry,
    node_rank: int,
    num_nodes: int,
    script: Path,
    master_addr: str,
    master_port: int,
    gpus_per_node: int,
    remote_workdir: str | None,
) -> list[str]:
    """构造单节点的 SSH 拉起命令。"""
    remote_script = str(script) if script.is_absolute() else os.path.abspath(str(script))
    # 默认以本仓库根目录为远程工作目录（要求所有节点在相同路径部署本仓库；
    # 生成的启动脚本内部也会 cd 到自己的 ROOT_DIR）。
    workdir = remote_workdir or str(Path(__file__).resolve().parent.parent)

    env_prefix = (
        f"NODE_RANK={node_rank} "
        f"NNODES={num_nodes} "
        f"MASTER_ADDR={master_addr} "
        f"MASTER_PORT={master_port} "
        f"GPUS_PER_NODE={gpus_per_node} "
    )
    remote_shell = (
        f"cd '{workdir}' && {env_prefix}bash '{remote_script}'"
    )

    return [
        "ssh",
        "-p",
        str(entry.ssh_port),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "BatchMode=yes",
        entry.host,
        remote_shell,
    ]


def stream_process(
    prefix: str,
    process: subprocess.Popen,
    sink,
    results: dict[str, int],
    lock: threading.Lock,
) -> None:
    """把单个节点进程的输出按行加前缀后写入 sink，并记录退出码。"""
    assert process.stdout is not None
    for raw_line in iter(process.stdout.readline, ""):
        with lock:
            sink.write(f"[{prefix}] {raw_line}")
            sink.flush()
    process.stdout.close()
    return_code = process.wait()
    with lock:
        results[prefix] = return_code


def run_cluster(
    entries: list[HostEntry],
    script: Path,
    master_port: int,
    gpus_per_node: int,
    master_addr_override: str | None,
    ssh_timeout: int,
    dry_run: bool,
    log_dir: Path | None,
) -> int:
    """并发拉起所有节点并聚合日志，返回失败节点数。"""
    num_nodes = len(entries)
    master_addr = master_addr_override or entries[0].host

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)

    commands: list[tuple[HostEntry, int, list[str]]] = []
    for node_rank, entry in enumerate(entries):
        command = build_remote_command(
            entry=entry,
            node_rank=node_rank,
            num_nodes=num_nodes,
            script=script,
            master_addr=master_addr,
            master_port=master_port,
            gpus_per_node=gpus_per_node,
            remote_workdir=None,
        )
        commands.append((entry, node_rank, command))

    print(
        f"[launch-cluster] 节点数={num_nodes} 主节点={master_addr}:{master_port} "
        f"每节点卡数={gpus_per_node} 脚本={script}"
    )

    if dry_run:
        for entry, node_rank, command in commands:
            joined = " ".join(command)
            print(f"[dry-run] node{node_rank} ({entry.host}):\n  {joined}")
        return 0

    lock = threading.Lock()
    results: dict[str, int] = {}
    threads: list[threading.Thread] = []
    log_files: list = []

    import time

    stamp = time.strftime("%Y%m%d-%H%M%S")
    for entry, node_rank, command in commands:
        prefix = f"node{node_rank}:{entry.host}"
        sink = sys.stdout
        if log_dir is not None:
            log_file = open(log_dir / f"{stamp}-node{node_rank}-{entry.host}.log", "w", encoding="utf-8")
            log_files.append(log_file)
            sink = log_file

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        thread = threading.Thread(
            target=stream_process,
            args=(prefix, process, sink, results, lock),
            daemon=True,
        )
        thread.start()
        threads.append(thread)
        print(f"[launch-cluster] 已拉起 node{node_rank} -> {entry.host} (ssh -p {entry.ssh_port})")

    for thread in threads:
        thread.join()
    for log_file in log_files:
        log_file.close()

    failed = {prefix: code for prefix, code in results.items() if code != 0}
    for prefix, code in sorted(results.items()):
        status = "OK" if code == 0 else f"FAILED({code})"
        print(f"[launch-cluster] {prefix} 退出码: {status}")

    if failed:
        print(
            f"[launch-cluster] 有 {len(failed)} 个节点失败: "
            + ", ".join(sorted(failed))
        )
        return 1
    print("[launch-cluster] 全部节点正常退出。")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""
    default_hostfile = Path(__file__).resolve().parent.parent / "generated" / "autoparallel" / "hostfile"
    default_script = Path(__file__).resolve().parent.parent / "generated" / "autoparallel" / "launch_megatron_plan.sh"
    parser = argparse.ArgumentParser(description="SSH + hostfile 多机一键拉起器")
    parser.add_argument("--hostfile", type=Path, default=default_hostfile, help="hostfile 路径")
    parser.add_argument("--script", type=Path, default=default_script, help="每节点执行的启动脚本")
    parser.add_argument("--gpus-per-node", type=int, default=8, help="每节点卡数（默认 8）")
    parser.add_argument("--master-addr", default=None, help="覆盖 MASTER_ADDR（默认取 hostfile 第一行）")
    parser.add_argument("--master-port", type=int, default=29500, help="rendezvous 端口（默认 29500）")
    parser.add_argument("--ssh-timeout", type=int, default=30, help="SSH 连接超时秒数")
    parser.add_argument("--log-dir", type=Path, default=None, help="按节点分文件保存日志的目录")
    parser.add_argument("--dry-run", action="store_true", help="只打印将执行的 SSH 命令")
    return parser


def main() -> int:
    """命令行入口。"""
    args = build_arg_parser().parse_args()
    entries = parse_hostfile(args.hostfile)

    total_slots = sum(entry.slots for entry in entries)
    print(
        f"[launch-cluster] hostfile={args.hostfile} 节点数={len(entries)} "
        f"slots 合计={total_slots}"
    )

    return run_cluster(
        entries=entries,
        script=args.script,
        master_port=args.master_port,
        gpus_per_node=args.gpus_per_node,
        master_addr_override=args.master_addr,
        ssh_timeout=args.ssh_timeout,
        dry_run=args.dry_run,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
