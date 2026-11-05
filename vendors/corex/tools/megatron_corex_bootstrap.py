#!/usr/bin/env python3
"""CoreX(天数智芯)版 Megatron 环境 bootstrap。

由摩尔线程版 megatron_musa_bootstrap.py 适配而来。CoreX 走 CUDA 兼容层
（torch 2.7.1+corex 官方构建，torch.cuda.* 原生可用），因此相比 MUSA 版
大幅简化：

  无需 ① torch.cuda→torch.musa 垫片（musa_patch 整层删除）
  无需 ② apex 屏蔽 —— 容器自带 corex 编译版 apex/FusedAdam/FusedLayerNorm，
           本机 CPU(Xeon 6330) 有 AVX-512，无 MUSA 机器的 SIGILL 风险
  无需 ③ NCCL→MCCL 重映射 —— CoreX 自带 libnccl，backend 名就是 nccl
           （2 卡 all_reduce 已实测通过）
  无需 ④ training.py GPU allocation 补丁 —— 本仓库用上游干净源码，
           `model_module.cuda()` 原生工作
  无需 ⑤ checkpointing.py map_location 补丁 —— 上游 0.14.0rc2 本就
           `map_location='cpu'` 加载

保留的处理（本版本上游源码中真实需要）：
  1. te_stub：容器未安装 TransformerEngine（天数无公开 TE 适配），
     stub 让 `import transformer_engine` 抛 ModuleNotFoundError，
     Megatron 全局 HAVE_TE=False，配合 --transformer-impl local。
     上游 0.14.0rc2 的 TE 导入全部有 try/except 或为函数内懒加载，可安全兜底。
  2. 源码补丁（幂等 + 文件锁，8 rank 并发安全）：
     a) distrib_optimizer.py dp_zero 断点续训 KeyError:'step' 修复
        （上游 bug：dp_zero 只存 param/exp_avg/exp_avg_sq，遍历 dst_tensors
         时对缺失键 copy_ 即崩）
     b) jit.py jit_fuser 退化为普通函数（torch.compile/dynamo 在非 NVIDIA
        后端的追踪兼容性不可控，与 MUSA 版同策略）
     c) checkpointing.py 滚动保留最近 N 个持久 checkpoint
        （13B 全量微调每 50 步存档会写满盘）
     d) core/utils.py is_te_min_version 的 None 比较 TypeError 修复
        （TE 缺失(te_stub)时 get_te_version() 返回 None，`None >= 版本`
         直接 TypeError；--optimizer-cpu-offload 配方必经此函数）
     （fused_layer_norm 的 RMSNorm 分发补丁见 _PATCHES 首条注释）
  3. 运行时猴子补丁：legacy fused_kernels.load 改为"尽力加载"，
     失败降级 no-op（corex 的 CUDA_HOME 指向 /usr/local/corex，JIT 编译
     legacy 融合核不保证成功；本训练配方未启用 masked-softmax-fusion 等
     依赖项，降级不影响正确性）。compile_helpers（数据集 C++ 索引器，
     纯 CPU）保持必选。

设计目标：
1. 不改动上游源码目录的其它内容（只幂等补丁 3 个文件）。
2. 运行入口统一走 CoreX Megatron 链路。
3. 在当前仓库任意位置执行时都能得到一致结果。
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


_BOOTSTRAPPED = False


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
                f"[megatron_corex_bootstrap] 未找到上游源码，"
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
                    print(f"[megatron_corex_bootstrap] 自动拉取成功 (tag={tag})")
                    return True
                last_err = (proc.stderr or "").strip()[-200:]
                # 失败时 git 可能留下半截目录，清掉再试下一个 tag。
                if target.exists():
                    import shutil
                    shutil.rmtree(target, ignore_errors=True)
            print(
                "[megatron_corex_bootstrap] 自动拉取失败(内网作业机属正常现象)，"
                f"请人工放置源码包后重试。git 尾部输出: {last_err}"
            )
            return False
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _upstream_root() -> Path:
    """返回上游 Megatron-LM 源码目录（上游干净源码 core_v0.14.0rc2）。"""
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
    # bootstrap 后续的 6 处幂等补丁会把它配成 CoreX 可用状态。
    if _try_fetch_upstream(_repo_root() / "Megatron-LM"):
        return _repo_root() / "Megatron-LM"
    raise FileNotFoundError(
        "未找到上游 Megatron-LM 源码目录。请设置环境变量 MEGATRON_UPSTREAM_ROOT "
        "指向 Megatron-LM 源码(容器内通常为仓库内 Megatron-LM/)；"
        "或把上游 core_v0.14.0rc2 干净源码放到 "
        f"{_repo_root() / 'Megatron-LM'}。"
    )


def _te_stub_root() -> Path:
    """返回 te_stub 目录(屏蔽 TransformerEngine)。"""
    return _repo_root() / "te_stub"


def _prepend_sys_path(path: Path) -> None:
    """把路径放到 sys.path 前部，同时避免重复项。"""
    path_str = str(path)
    sys.path[:] = [entry for entry in sys.path if entry != path_str]
    sys.path.insert(0, path_str)


# ---------------------------------------------------------------------------
# Megatron 源码补丁（幂等，8 rank 并发下用文件锁串行化）
# ---------------------------------------------------------------------------

_PATCHES = [
    {
        # RMSNorm + --transformer-impl local + 无 TE 的路径修复。
        # 上游 gpt_layer_specs.py 的选择逻辑：装了 apex 就用 FusedLayerNorm
        # （只支持 LayerNorm），只有"未装 apex"才回退 WrappedTorchNorm(支持
        # RMSNorm)。CoreX 容器自带 corex 编译版 apex，RMSNorm 模型(llama2)
        # 因此在 FusedLayerNorm.__init__ 的 assert 崩溃。
        # 补丁：FusedLayerNorm.__new__ 按 config.normalization 分发 ——
        # LayerNorm 走原 fused 路径，RMSNorm 回退 torch 原生 RMSNorm
        # （torch>=2.4，容器为 2.7.1+corex）。
        "path": "megatron/core/fusions/fused_layer_norm.py",
        "old": (
            "class FusedLayerNorm(torch.nn.Module):\n"
            '    """Layer Norm, fused into a single CUDA kernel.\n'
        ),
        "new": (
            "class FusedLayerNorm(torch.nn.Module):\n"
            "    # autoparallel_corex_patched: __new__ 按 normalization 分发。\n"
            "    # LayerNorm 走原 fused 内核；RMSNorm 回退 WrappedTorchNorm\n"
            "    # (torch 原生 RMSNorm)。\n"
            "    def __new__(cls, config=None, *args, **kwargs):\n"
            "        if config is not None and getattr(config, 'normalization', 'LayerNorm') != 'LayerNorm':\n"
            "            from megatron.core.transformer.torch_norm import WrappedTorchNorm\n"
            "            return WrappedTorchNorm(config=config, *args, **kwargs)\n"
            "        return super().__new__(cls)\n"
            "\n"
            '    """Layer Norm, fused into a single CUDA kernel.\n'
        ),
    },
    {
        # 断点续训 KeyError: 'step' 修复。dp_zero 参数状态只保存/加载
        # param/exp_avg/exp_avg_sq 三块（step 由 model_optim_rng.pt 的 optimizer
        # state_dict + param_group 恢复），但 _set_main_param_and_optimizer_states
        # 的 else 分支遍历 dst_tensors（含 step）并 copy_(tensors[key])，tensors
        # 无 step 即崩。跳过缺失键即可。（上游 core_v0.14.0rc2 同样存在）
        "path": "megatron/core/optimizer/distrib_optimizer.py",
        "old": (
            '            dst_tensors = {"param": main_param, **optim_state}\n'
            "            for key in dst_tensors:\n"
            "                dst_tensors[key].copy_(tensors[key])"
        ),
        "new": (
            '            dst_tensors = {"param": main_param, **optim_state}\n'
            "            # autoparallel_corex_patched: dp_zero 只存 param/exp_avg/exp_avg_sq，\n"
            "            # 不含 step；遍历时跳过 tensors 缺失的键，否则续训 KeyError: 'step'。\n"
            "            for key in dst_tensors:\n"
            "                if key not in tensors:\n"
            "                    continue\n"
            "                dst_tensors[key].copy_(tensors[key])"
        ),
    },
    {
        # torch.compile 的 dynamo 追踪在非 NVIDIA 后端的兼容性不可控（MUSA 上
        # 实测会在 device_interface 崩溃）。CoreX 首发迁移从稳出发，把
        # jit_fuser 退化为普通函数；后续可试验性移除本补丁评估 torch.compile。
        "path": "megatron/core/jit.py",
        "old": (
            '    if is_torch_min_version("2.2.0a0"):\n'
            "        jit_fuser = torch.compile\n"
        ),
        "new": (
            '    if is_torch_min_version("2.2.0a0"):\n'
            "        # autoparallel_corex_patched: CoreX 下 torch.compile 的 dynamo\n"
            "        # 追踪兼容性未验证，首发迁移退化为普通函数（稳态优先）。\n"
            "        jit_fuser = lambda f: f\n"
        ),
    },
    {
        # 自实现"保留最近 N 个持久 checkpoint"的滚动删除（上游本版本无
        # --keep-last-n-checkpoints 参数）。--ckpt-format torch 为同步保存，
        # save_checkpoint 末尾的 barrier 保证各 rank 写盘完成；只让 rank0
        # 删除，避免 8 rank 并发 rmtree 竞态。non_persistent_ckpt 有自己的
        # 清理逻辑，不碰。
        "path": "megatron/training/checkpointing.py",
        "old": (
            "    ft_integration.on_checkpointing_end(is_async_finalization=False)\n"
            "\n"
            "def cleanup_old_non_persistent_checkpoint(save_dir, leave_ckpt_num=1, do_async=False):"
        ),
        "new": (
            "    ft_integration.on_checkpointing_end(is_async_finalization=False)\n"
            "\n"
            "    # autoparallel_corex_patched: 滚动保留最近 N 个持久 checkpoint。\n"
            "    # N 由环境变量 COREX_KEEP_LAST_CKPT 控制（默认 3）。\n"
            "    if not non_persistent_ckpt:\n"
            "        _rotate_persistent_checkpoints(\n"
            "            args.save, keep_last_n=int(os.environ.get('COREX_KEEP_LAST_CKPT', '3'))\n"
            "        )\n"
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
            "        print_rank_0(f\"[autoparallel_corex_patched] 滚动清理旧 checkpoint: {old_dir}\")\n"
            "        shutil.rmtree(old_dir, ignore_errors=True)\n"
            "\n"
            "\n"
            "def cleanup_old_non_persistent_checkpoint(save_dir, leave_ckpt_num=1, do_async=False):"
        ),
    },
    {
        # 无 TE 环境下 is_te_min_version 的 None 比较崩溃修复。
        # get_te_version() 在 TE 缺失(CoreX te_stub)时返回 None，
        # `None >= PkgVersion(...)` 直接 TypeError。触发路径：
        # --optimizer-cpu-offload + --use-precision-aware-optimizer 时
        # OptimizerConfig.__post_init__ 调 is_te_min_version("2.1.0")。
        # 语义修复：TE 缺失时任何版本下限检查都返回 False（无 TE 可言）。
        "path": "megatron/core/utils.py",
        "old": (
            "    if check_equality:\n"
            "        return get_te_version() >= PkgVersion(version)\n"
            "    return get_te_version() > PkgVersion(version)"
        ),
        "new": (
            "    if check_equality:\n"
            "        # autoparallel_corex_patched: TE 缺失(te_stub)时 get_te_version() 为\n"
            "        # None，直接比较会 TypeError；无 TE 时版本下限恒为 False。\n"
            "        _te_ver = get_te_version()\n"
            "        return _te_ver is not None and _te_ver >= PkgVersion(version)\n"
            "    _te_ver_ = get_te_version()\n"
            "    return _te_ver_ is not None and _te_ver_ > PkgVersion(version)"
        ),
    },
    {
        # clip_grad 的 apex multi_tensor_scale 无 BFloat16 分发（corex 版 apex）。
        # --optimizer-cpu-offload 路径 use_decoupled_grad=True，grad 为 bf16，
        # clip_grad_by_total_norm_fp32 走 apex multi_tensor_scale_cuda 即崩：
        #   RuntimeError: "multi_tensor_scale_cuda" not implemented for 'BFloat16'
        # 该 clip 数学上就是 grads *= clip_coeff 的原地缩放，改用 torch 原生
        # foreach（bf16 原生支持，数值等价，任何后端可用）。
        "path": "megatron/core/optimizer/clip_grads.py",
        "old": (
            "    if clip_coeff < 1.0:\n"
            "        dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')\n"
            "        multi_tensor_applier(\n"
            "            multi_tensor_scale_impl, dummy_overflow_buf, [grads, grads], clip_coeff\n"
            "        )"
        ),
        "new": (
            "    if clip_coeff < 1.0:\n"
            "        # autoparallel_corex_patched: corex 版 apex 的 multi_tensor_scale_cuda\n"
            "        # 无 BFloat16 分发（optimizer-cpu-offload 的 decoupled_grad 为 bf16）。\n"
            "        # 改用 torch 原生 foreach 原地缩放，数值等价。\n"
            "        torch._foreach_mul_(grads, clip_coeff)"
        ),
    },
]


def _apply_source_patches(upstream: Path) -> None:
    """对上游 Megatron-LM 源码做幂等补丁。

    8 个 rank 在同一容器文件系统里并发执行本函数。read→检查marker→count→
    write 必须用文件锁串行化，否则一个 rank 写文件期间(write_text 先 truncate
    再写)另一个 rank 会读到半截文件: marker 缺失 + 目标文本被截掉 → count=0，
    误报"补丁匹配异常"。加锁后首进程完成补丁，其余 rank 见补丁文本直接跳过。
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
                # 按 patch 粒度判断幂等：同一文件可打多个补丁，若用
                # "文件含 marker 就跳过"会导致同文件后续补丁被误判漏打。
                if patch["new"] in src:
                    continue  # 该补丁已应用
                count = src.count(patch["old"])
                if count == 0:
                    # 宽容兜底: 目标文本已不存在(被等价逻辑替换/官方版已改)。
                    # 宁可持续跳过也不让 8 卡训练硬失败。
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


def _patch_fused_kernels_optional() -> None:
    """legacy fused_kernels.load 改为尽力加载（失败降级 no-op）。

    上游 _compile_dependencies 无条件调用 fused_kernels.load(args) JIT 编译
    legacy CUDA 融合核。corex 的 CUDA_HOME(/usr/local/corex) 下 JIT 编译
    不保证成功；本训练配方未启用 masked_softmax_fusion 等依赖 legacy 融合核
    的特性，降级不影响正确性。compile_helpers(数据集 C++ 索引器)保持必选。
    """
    import megatron.legacy.fused_kernels as fused_kernels

    _orig_load = fused_kernels.load

    def _tolerant_load(args):
        try:
            _orig_load(args)
        except Exception as exc:  # noqa: BLE001 —— JIT 编译失败的降级路径
            print(
                f"[autoparallel_corex] fused_kernels 加载失败，降级为 no-op: {exc}",
                flush=True,
            )

    fused_kernels.load = _tolerant_load
    # initialize.py 已 `from megatron.legacy.fused_kernels import load` 持有
    # 原引用，需同步替换。
    import megatron.training.initialize as mInitialize

    if hasattr(mInitialize, "fused_kernels"):
        mInitialize.fused_kernels.load = _tolerant_load
    elif hasattr(mInitialize, "load"):
        mInitialize.load = _tolerant_load


def bootstrap(verbose: bool = False) -> dict[str, str]:
    """初始化 CoreX Megatron 运行环境。

    返回值用于调试和自检，包含仓库根目录和上游源码目录信息。
    """
    global _BOOTSTRAPPED

    repo_root = _repo_root()
    upstream_root = _upstream_root()
    te_stub_root = _te_stub_root()

    if not te_stub_root.is_dir():
        raise FileNotFoundError(f"未找到 te_stub 目录: {te_stub_root}")

    # 1) te_stub 最先进入 sys.path，屏蔽 TransformerEngine
    _prepend_sys_path(te_stub_root)

    # 2) 对 Megatron 源码做幂等补丁
    _apply_source_patches(upstream_root)

    # 3) 上游 Megatron-LM + 本仓库 tools 进入 sys.path
    _prepend_sys_path(upstream_root)
    _prepend_sys_path(repo_root / "tools")
    _prepend_sys_path(repo_root)

    # 记录环境，便于脚本和排障时直接查看。
    os.environ.setdefault("MEGATRON_PROJECT_ROOT", str(repo_root))
    os.environ.setdefault("MEGATRON_UPSTREAM_ROOT", str(upstream_root))
    os.environ.setdefault("MEGATRON_COREX_ENABLED", "1")

    if not _BOOTSTRAPPED:
        # 4) 运行时猴子补丁：fused_kernels 尽力加载。
        #    （此时 megatron 尚未 import，先改 initialize 模块持有的引用，
        #     在 import 后二次生效。）
        importlib.import_module("megatron.training.arguments")
        _patch_fused_kernels_optional()
        _BOOTSTRAPPED = True

    patched_module = importlib.import_module("megatron.training.arguments")
    patched_parse_args = getattr(patched_module, "parse_args", None)
    patched_module_name = getattr(patched_parse_args, "__module__", "")

    if not patched_module_name.startswith("megatron.training."):
        raise RuntimeError(
            "CoreX Megatron 接口异常: megatron.training.arguments.parse_args "
            f"当前来自 {patched_module_name or 'unknown'}"
        )

    info = {
        "repo_root": str(repo_root),
        "upstream_root": str(upstream_root),
        "te_stub_root": str(te_stub_root),
        "parse_args_module": patched_module_name,
    }

    if verbose:
        print("[megatron-corex-bootstrap] repo_root =", info["repo_root"])
        print("[megatron-corex-bootstrap] upstream_root =", info["upstream_root"])
        print("[megatron-corex-bootstrap] te_stub_root =", info["te_stub_root"])
        print("[megatron-corex-bootstrap] parse_args_module =", info["parse_args_module"])

    return info


if __name__ == "__main__":
    bootstrap(verbose=True)
