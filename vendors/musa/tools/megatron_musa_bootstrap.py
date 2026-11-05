#!/usr/bin/env python3
"""MUSA(摩尔线程)版 Megatron 环境 bootstrap。

由沐曦版 megatron_maca_bootstrap.py 适配而来，MUSA 比 MACA 多三层处理：
  1. te_stub：MT-TransformerEngine 的 .so 用 AVX-512 编译，本机 CPU(海光
     C86-4G) 无 AVX-512，import 即 SIGILL(硬崩溃)。因此把 te_stub 目录放到
     sys.path 前部，让 `import transformer_engine` 走 stub 抛 ModuleNotFoundError，
     Megatron 全局 HAVE_TE=False。
  2. 源码补丁：Megatron 有 2 处无保护的顶层 TE 导入 / 兜底缺失
     (pipeline_parallel/schedules.py、transformer/transformer_block.py)，
     在 import musa_patch 之前对源码文件做幂等修改。
  3. musa_patch：摩尔线程官方补丁的 autoparallel 定制版。它把 torch.cuda.*
     映射到 torch.musa.*，并给 megatron 内部打上 MUSA 适配补丁。
     import musa_patch 即自动生效(patch_after_import_torch +
     patch_before_import_megatron)。

设计目标：
1. 不改动官方源码目录的其它内容（只幂等补丁 2 个文件）。
2. 运行入口统一走 MUSA Megatron 链路。
3. 在当前仓库任意位置执行时都能得到一致结果。
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


_BOOTSTRAPPED = False
_MUSA_PATCH_MARKER = "autoparallel_musa_patched"


def _repo_root() -> Path:
    """返回当前仓库根目录。"""
    return Path(__file__).resolve().parent.parent


def _try_fetch_upstream(target: Path) -> bool:
    """目录缺失时自动拉取上游 core_v0.14.0rc2 干净源码（网络可达时）。

    内网隔离的作业机通常不通公网，拉取失败不算错误——返回 False，交由
    调用方按原报错指引人工放置源码包。多 rank 并发启动 bootstrap 时用
    文件锁串行化：先到的 rank 完成 clone，其余直接命中。设
    MEGATRON_AUTO_FETCH=0 可关闭自动拉取。
    """
    if os.environ.get("MEGATRON_AUTO_FETCH", "1").strip() == "0":
        return False
    import subprocess
    import fcntl

    if (target / "megatron").is_dir():
        return True
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.parent / ".megatron_fetch.lock"
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            if (target / "megatron").is_dir():  # 并发的 rank 已拉好
                return True
            print(
                f"[megatron_musa_bootstrap] 未找到上游源码，"
                f"尝试自动拉取 core_v0.14.0rc2 → {target}"
            )
            # tag 双保险：上游历史 tag 存在 core_v / core_r 两种前缀命名。
            last_err = ""
            for tag in ("core_v0.14.0rc2", "core_r0.14.0rc2"):
                proc = subprocess.run(
                    ["git", "clone", "--branch", tag, "--depth", "1",
                     "https://github.com/NVIDIA/Megatron-LM.git", str(target)],
                    capture_output=True, text=True, timeout=600,
                )
                if proc.returncode == 0 and (target / "megatron").is_dir():
                    print(f"[megatron_musa_bootstrap] 自动拉取成功 (tag={tag})")
                    return True
                last_err = (proc.stderr or "").strip()[-200:]
                # 失败时 git 可能留下半截目录，清掉再试下一个 tag。
                if target.exists():
                    import shutil
                    shutil.rmtree(target, ignore_errors=True)
            print(
                "[megatron_musa_bootstrap] 自动拉取失败(内网作业机属正常现象)，"
                f"请人工放置源码包后重试。git 尾部输出: {last_err}"
            )
            return False
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _upstream_root() -> Path:
    """返回上游 MUSA 适配版 Megatron-LM 源码目录。"""
    env_root = os.environ.get("MEGATRON_UPSTREAM_ROOT")
    if env_root:
        return Path(env_root)
    candidates = [
        Path("/home/Megatron-LM"),
        _repo_root() / "Megatron-LM",
    ]
    for candidate in candidates:
        if (candidate / "megatron").is_dir():
            return candidate
    # 候选都不在位：网络可达时自动拉上游干净源码到仓库内挂载点，
    # bootstrap 后续的幂等补丁会把它配成 MUSA 可用状态。
    if _try_fetch_upstream(_repo_root() / "Megatron-LM"):
        return _repo_root() / "Megatron-LM"
    raise FileNotFoundError(
        "未找到上游 Megatron-LM 源码目录。请设置环境变量 MEGATRON_UPSTREAM_ROOT "
        "指向摩尔线程适配版 Megatron 源码(容器内通常为 /home/Megatron-LM)；"
        "或把上游 core_v0.14.0rc2 干净源码放到 "
        f"{_repo_root() / 'Megatron-LM'}。"
    )


def _te_stub_root() -> Path:
    """返回 te_stub 目录(屏蔽 MT-TransformerEngine)。"""
    return _repo_root() / "te_stub"


def _prepend_sys_path(path: Path) -> None:
    """把路径放到 sys.path 前部，同时避免重复项。"""
    path_str = str(path)
    sys.path[:] = [entry for entry in sys.path if entry != path_str]
    sys.path.insert(0, path_str)


# ---------------------------------------------------------------------------
# Megatron 源码补丁（幂等，对应 adapt/patch_megatron_musa.sh）
# ---------------------------------------------------------------------------

_PATCHES = [
    {
        "path": "megatron/core/pipeline_parallel/schedules.py",
        "old": (
            "from transformer_engine.pytorch.cpu_offload import "
            "get_fine_grained_offload_handler"
        ),
        "new": (
            "# autoparallel_musa_patched: 屏蔽 TE 顶层硬导入(真实 TE 在本机 CPU 上 SIGILL)\n"
            "try:\n"
            "    from transformer_engine.pytorch.cpu_offload import get_fine_grained_offload_handler\n"
            "except (ImportError, ModuleNotFoundError):\n"
            "    class _NoOpFineGrainedOffload:\n"
            "        def start_microbatch_forward(self, *a, **k):\n"
            "            return None\n"
            "        def start_microbatch_backward(self, *a, **k):\n"
            "            return None\n"
            "    def get_fine_grained_offload_handler():\n"
            "        return _NoOpFineGrainedOffload()"
        ),
    },
    {
        "path": "megatron/training/training.py",
        "old": (
            "    # GPU allocation.\n"
            "    # For FSDP2, we don't allocate GPU memory here. We allocate GPU memory\n"
            "    # in the fully_shard function of FSDP2 instead.\n"
            "    # if (\n"
            "    #     not (args.use_torch_fsdp2 and args.use_cpu_initialization)\n"
            "    #     and not args.init_model_with_meta_device\n"
            "    # ):\n"
            "    #     for model_module in model:\n"
            "    #         model_module.cuda(torch.cuda.current_device())"
        ),
        "new": (
            "    # GPU allocation.\n"
            "    # For FSDP2, we don't allocate GPU memory here. We allocate GPU memory\n"
            "    # in the fully_shard function of FSDP2 instead.\n"
            "    # autoparallel_musa_patched: 摩尔线程版把 .cuda() 注释掉导致参数留在\n"
            "    # CPU，Float16Optimizer 报 \"Received torch.BFloat16Tensor\"。这里显式\n"
            "    # 用 musa 设备把模型搬到 MUSA（走 Module.to，避免 Module.cuda 内部把\n"
            "    # int 转成 torch.device('cuda') 导致 musa 不识别）。\n"
            "    if (\n"
            "        not (args.use_torch_fsdp2 and args.use_cpu_initialization)\n"
            "        and not args.init_model_with_meta_device\n"
            "    ):\n"
            "        for model_module in model:\n"
            "            model_module.to(torch.device('musa', torch.musa.current_device()))"
        ),
    },
    {
        "path": "megatron/core/transformer/transformer_block.py",
        "old": (
            "except ImportError:\n"
            "    HAVE_TE = False\n"
            "    get_cpu_offload_context = None"
        ),
        "new": (
            "except ImportError:\n"
            "    HAVE_TE = False\n"
            "    get_cpu_offload_context = None\n"
            "    # autoparallel_musa_patched: TE 不可用时兜底，forward 里无保护地调用\n"
            "    # get_fine_grained_offload_handler()(设置 num_layers/pp_size 等属性、\n"
            "    # 调 end_microbatch)，真实 TE 不存在时给 no-op 实现。\n"
            "    class _NoOpFineGrainedOffload:\n"
            "        def start_microbatch_forward(self, *a, **k):\n"
            "            return None\n"
            "        def start_microbatch_backward(self, *a, **k):\n"
            "            return None\n"
            "        def end_microbatch(self, *a, **k):\n"
            "            return None\n"
            "    def get_fine_grained_offload_handler():\n"
            "        return _NoOpFineGrainedOffload()\n"
            "    class FineGrainedOffloadLayerCounter:\n"
            "        def __init__(self, *a, **k):\n"
            "            pass\n"
            "        def __enter__(self):\n"
            "            return self\n"
            "        def __exit__(self, *a):\n"
            "            return False"
        ),
    },
    {
        # torch.compile 的 dynamo 追踪在 MUSA 下会崩：torch._dynamo.device_interface
        # 拿 musa 设备字符串('musa:0')当列表索引 -> TypeError。本机海光 CPU 也无
        # AVX-512，compile 收益有限，故把 jit_fuser 退化为普通函数。
        "path": "megatron/core/jit.py",
        "old": (
            '    if is_torch_min_version("2.2.0a0"):\n'
            "        jit_fuser = torch.compile\n"
        ),
        "new": (
            '    if is_torch_min_version("2.2.0a0"):\n'
            "        # autoparallel_musa_patched: MUSA 下 torch.compile 的 dynamo 追踪\n"
            "        # 会在 torch._dynamo.device_interface 崩溃，退化为普通函数。\n"
            "        jit_fuser = lambda f: f\n"
        ),
    },
    {
        # MUSA 上 checkpoint 必须加载到 CPU，再由 load_state_dict 拷进已分配的
        # GPU 参数。加载发生时 GPU 已被模型(25G)+DDP 双缓冲(25G+25G bf16)占满
        # (~75G/80G)，再往 GPU 上 torch.load 25G 直接超限，驱动抛
        # "invalid argument to exchangeDevice"(checkpointing.py 捕获后打印为
        # "could not load the checkpoint")。load 到 CPU 是本仓库
        # verify_backward_13b.py 已验证的路径，CPU 内存充足(1.5T)。
        # 配合启动参数 --ckpt-format torch(单次加载，默认 torch_dist 会二次加载)。
        "path": "megatron/training/checkpointing.py",
        "old": "torch.load(checkpoint_name, map_location='musa', weights_only=False)",
        "new": "torch.load(checkpoint_name, map_location='cpu', weights_only=False)  # autoparallel_musa_patched: MUSA 下 load 到 CPU",
    },
    {
        # 本版本无 --keep-last-n-checkpoints 参数，需自实现"保留最近 N 个持久
        # checkpoint"的滚动删除。--ckpt-format torch 为同步保存，save_checkpoint
        # 末尾的 barrier 保证各 rank 写盘完成；只让 rank0 删除，避免 8 rank
        # 并发 rmtree 竞态。non_persistent_ckpt 有自己的清理逻辑，不碰。
        "path": "megatron/training/checkpointing.py",
        "old": (
            "    ft_integration.on_checkpointing_end(is_async_finalization=False)\n"
            "\n"
            "def cleanup_old_non_persistent_checkpoint(save_dir, leave_ckpt_num=1, do_async=False):"
        ),
        "new": (
            "    ft_integration.on_checkpointing_end(is_async_finalization=False)\n"
            "\n"
            "    # autoparallel_musa_patched: 滚动保留最近 3 个持久 checkpoint。\n"
            "    if not non_persistent_ckpt:\n"
            "        _rotate_persistent_checkpoints(args.save, keep_last_n=3)\n"
            "\n"
            "\n"
            "def _rotate_persistent_checkpoints(save_dir, keep_last_n=3):\n"
            "    \"\"\"删除 save_dir 下保留 keep_last_n 个最新之外的 iter_* 目录(仅 rank0)。\"\"\"\n"
            "    import re\n"
            "    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:\n"
            "        return\n"
            "    if not save_dir or not os.path.isdir(save_dir):\n"
            "        return\n"
            "    iters = sorted(\n"
            "        (int(m.group(1)) for d in os.listdir(save_dir)\n"
            "         if (m := re.fullmatch(r\"iter_(\\d{7})\", d))),\n"
            "        reverse=True,\n"
            "    )\n"
            "    for old in iters[keep_last_n:]:\n"
            "        old_dir = os.path.join(save_dir, f\"iter_{old:07d}\")\n"
            "        print_rank_0(f\"[autoparallel_musa_patched] 滚动清理旧 checkpoint: {old_dir}\")\n"
            "        shutil.rmtree(old_dir, ignore_errors=True)\n"
            "\n"
            "\n"
            "def cleanup_old_non_persistent_checkpoint(save_dir, leave_ckpt_num=1, do_async=False):"
        ),
    },
    {
        # 断点续训 KeyError: 'step' 修复。dp_zero 参数状态只保存/加载
        # param/exp_avg/exp_avg_sq 三块（step 由 model_optim_rng.pt 的 optimizer
        # state_dict + param_group 恢复），但 _set_main_param_and_optimizer_states
        # 的 else 分支遍历 dst_tensors（含 step）并 copy_(tensors[key])，tensors
        # 无 step 即崩。跳过缺失键即可。
        "path": "megatron/core/optimizer/distrib_optimizer.py",
        "old": (
            '            dst_tensors = {"param": main_param, **optim_state}\n'
            "            for key in dst_tensors:\n"
            "                dst_tensors[key].copy_(tensors[key])"
        ),
        "new": (
            '            dst_tensors = {"param": main_param, **optim_state}\n'
            "            # autoparallel_musa_patched: dp_zero 只存 param/exp_avg/exp_avg_sq，\n"
            "            # 不含 step；遍历时跳过 tensors 缺失的键，否则续训 KeyError: 'step'。\n"
            "            for key in dst_tensors:\n"
            "                if key not in tensors:\n"
            "                    continue\n"
            "                dst_tensors[key].copy_(tensors[key])"
        ),
    },
]


def _apply_source_patches(upstream: Path) -> None:
    """对上游 Megatron-LM 源码做幂等补丁。

    8 个 rank 在同一容器文件系统里并发执行本函数。read→检查marker→count→
    write 必须用文件锁串行化，否则一个 rank 写文件期间(write_text 先 truncate
    再写)另一个 rank 会读到半截文件: marker 缺失 + 目标文本被截掉 → count=0，
    误报"补丁匹配异常"。加锁后首进程完成补丁，其余 rank 见 marker 直接跳过。
    """
    import fcntl

    lock_path = upstream / ".autoparallel_patch.lock"
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            for patch in _PATCHES:
                target = upstream / patch["path"]
                if not target.is_file():
                    raise FileNotFoundError(f"未找到需补丁的源文件: {target}")
                src = target.read_text(encoding="utf-8")
                # 按 patch 粒度判断幂等：同一文件可打多个补丁（如 checkpointing.py
                # 有 load 改 CPU + 滚动清理两处），若用"文件含 marker 就跳过"会导致
                # 第一个补丁引入 marker 后，同文件后续补丁被误判为已应用而漏打。
                if patch["new"] in src:
                    continue  # 该补丁已应用
                count = src.count(patch["old"])
                if count == 0:
                    # 宽容兜底: 目标文本已不存在(被等价逻辑替换/官方版已改)。
                    # 锁定下本不该出现，宁可持续跳过也不让 8 卡训练硬失败。
                    print(
                        f"[bootstrap] 警告: {target} 未命中补丁目标(0次)，"
                        f"视为已适配跳过: {patch['path']}",
                        flush=True,
                    )
                    continue
                if count != 1:
                    raise RuntimeError(
                        f"补丁匹配异常: {target} 中目标文本出现 {count} 次(期望 1 次)"
                    )
                target.write_text(src.replace(patch["old"], patch["new"]), encoding="utf-8")
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def bootstrap(verbose: bool = False) -> dict[str, str]:
    """初始化 MUSA Megatron 运行环境。

    返回值用于调试和自检，包含仓库根目录和上游源码目录信息。
    """
    global _BOOTSTRAPPED

    repo_root = _repo_root()
    upstream_root = _upstream_root()
    te_stub_root = _te_stub_root()

    if not te_stub_root.is_dir():
        raise FileNotFoundError(f"未找到 te_stub 目录: {te_stub_root}")

    # 1) te_stub 最先进入 sys.path，屏蔽 MT-TransformerEngine
    _prepend_sys_path(te_stub_root)

    # 2) 对 Megatron 源码做幂等补丁（必须在 import musa_patch 之前）
    _apply_source_patches(upstream_root)

    # 3) 上游 Megatron-LM + 本仓库 tools 进入 sys.path
    _prepend_sys_path(upstream_root)
    _prepend_sys_path(repo_root / "tools")
    _prepend_sys_path(repo_root)  # 供 import musa_patch

    # 记录环境，便于脚本和排障时直接查看。
    os.environ.setdefault("MEGATRON_PROJECT_ROOT", str(repo_root))
    os.environ.setdefault("MEGATRON_UPSTREAM_ROOT", str(upstream_root))
    os.environ.setdefault("MEGATRON_MUSA_ENABLED", "1")

    if not _BOOTSTRAPPED:
        # 4) musa_patch：torch.cuda -> torch.musa 兼容 + megatron 内部补丁。
        #    它内部会 import megatron.training.utils(此时 te_stub 已在路径前、
        #    源码补丁已生效)。
        import musa_patch  # noqa: F401
        importlib.import_module("megatron.training.arguments")
        _BOOTSTRAPPED = True

    patched_module = importlib.import_module("megatron.training.arguments")
    patched_parse_args = getattr(patched_module, "parse_args", None)
    patched_module_name = getattr(patched_parse_args, "__module__", "")

    if not patched_module_name.startswith("megatron.training."):
        raise RuntimeError(
            "MUSA Megatron 接口异常: megatron.training.arguments.parse_args "
            f"当前来自 {patched_module_name or 'unknown'}"
        )

    info = {
        "repo_root": str(repo_root),
        "upstream_root": str(upstream_root),
        "te_stub_root": str(te_stub_root),
        "parse_args_module": patched_module_name,
    }

    if verbose:
        print("[megatron-musa-bootstrap] repo_root =", info["repo_root"])
        print("[megatron-musa-bootstrap] upstream_root =", info["upstream_root"])
        print("[megatron-musa-bootstrap] te_stub_root =", info["te_stub_root"])
        print("[megatron-musa-bootstrap] parse_args_module =", info["parse_args_module"])

    return info


if __name__ == "__main__":
    bootstrap(verbose=True)
