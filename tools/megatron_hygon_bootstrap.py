#!/usr/bin/env python3
"""海光版 Megatron 环境 bootstrap。

这个模块负责把当前仓库里的 `Megatron-LM` 源码作为上游接口层挂进
`sys.path`，随后再加载海光补丁层（优先 das 仓库自带的 hcu_megatron
源码层，其次环境 pip 安装的 dcu_megatron / tools 下随仓库 wheel）。

设计目标：
1. 不改动官方源码目录。
2. 运行入口统一走海光补丁链路。
3. 在当前仓库任意位置执行时都能得到一致结果。
"""

from __future__ import annotations

import importlib
import os
import sys
import typing
from pathlib import Path


# ── Python 3.11 兼容垫片 ─────────────────────────────────────────────────
# 上游 Megatron-LM（core_v0.18.2，pyproject 声明 requires-python>=3.12）在
# megatron/training/models/hybrid.py:5 使用了 `from typing import override`，
# 而 SCNet DCU 镜像 / 本地环境均为 Python 3.11，会在 import 阶段直接失败。
# 本模块是所有 megatron 导入的唯一前置入口（run_megatron_hygon.py 先 import
# 本模块、后触发 megatron 导入），在这里补齐符号可覆盖主进程及其 fork 子进程。
# 例外：multiprocessing spawn 子进程（checkpoint 异步保存的 Manager/写盘
# worker）不经过本模块，由 tools/sitecustomize.py 在解释器启动期补齐
# （run-8 踩坑：spawn 子进程重新 import pretrain_gpt.py 时缺 override 而崩，
# 主进程随后 EOFError）。
# 全树扫描确认 3.12+ 特性仅 override 这一处，故垫片足够。
# 3.12+ 环境不受影响（hasattr 命中即跳过）。
if not hasattr(typing, "override"):
    try:
        from typing_extensions import override as _override  # torch 自带该依赖
    except ImportError:  # 极端兜底：恒等装饰器，运行期无行为差异

        def _override(method):
            return method

    typing.override = _override


_BOOTSTRAPPED = False


def _repo_root() -> Path:
    """返回当前仓库根目录。"""
    return Path(__file__).resolve().parent.parent


def _upstream_candidates(repo_root: Path) -> list[Path]:
    """返回上游 Megatron-LM 源码目录的候选列表（按优先级）。

    1. 环境变量 MEGATRON_UPSTREAM_ROOT（多机部署时推荐显式指定）。
    2. <repo>/Megatron-LM（旧版仓库布局）。
    3. <repo>/Megatron-LM-das/3rdparty/Megatron-LM
       （hcu-megatron 布局：core_v0.18.2 以 git submodule 提供）。
    """
    candidates: list[Path] = []
    env_root = os.environ.get("MEGATRON_UPSTREAM_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    candidates.append(repo_root / "Megatron-LM")
    candidates.append(repo_root / "Megatron-LM-das" / "3rdparty" / "Megatron-LM")
    # 去重但保序。
    seen: set[str] = set()
    unique: list[Path] = []
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _looks_like_upstream_root(path: Path) -> bool:
    """判断目录是否是完整的 Megatron-LM 上游源码树。"""
    return (path / "megatron" / "training" / "arguments.py").is_file()


def _upstream_root() -> Path:
    """返回本地 Megatron-LM 源码目录。"""
    repo_root = _repo_root()
    for candidate in _upstream_candidates(repo_root):
        if _looks_like_upstream_root(candidate):
            return candidate

    hint = "\n".join(f"  - {candidate}" for candidate in _upstream_candidates(repo_root))
    raise FileNotFoundError(
        "未找到可用的上游 Megatron-LM 源码目录。已尝试:\n"
        f"{hint}\n"
        "请任选其一:\n"
        "  1. 在 Megatron-LM-das 仓库执行: git submodule update --init --recursive\n"
        "  2. 把完整 Megatron-LM (core_v0.18.2) 克隆到 <repo>/Megatron-LM\n"
        "  3. 导出环境变量 MEGATRON_UPSTREAM_ROOT 指向已有源码树"
    )


def _prepend_sys_path(path: Path) -> None:
    """把路径放到 sys.path 前部，同时避免重复项。"""
    path_str = str(path)
    sys.path[:] = [entry for entry in sys.path if entry != path_str]
    sys.path.insert(0, path_str)


def _load_hygon_patch_layer() -> str:
    """加载海光 Megatron 补丁层，返回所用补丁层来源描述。

    优先级：
    1. <repo>/Megatron-LM-das 源码自带的 hcu_megatron（namespace 包）。
       该目录即 hcu-megatron 0.18.2 源码包（见其 setup.py），与
       3rdparty/Megatron-LM 子模块（core_v0.18.2）同版本配套；把 das
       仓库根加入 sys.path 即可导入，无需 pip install，适合共享存储 /
       离线集群。`import hcu_megatron.megatron_adaptor` 会在导入期自动
       完成打补丁（megatron_adaptor.py 模块末尾执行 patch_features(...)）。
    2. 环境已 pip 安装的 dcu_megatron（旧命名补丁层，0.1x 版本线，
       与 0.18.2 源码接口不配套，仅作无 das 源码时的兜底）。
    3. tools/ 下随仓库携带的 dcu_megatron-*.whl（纯 python wheel 可被
       zipimport 直接导入，同样无需 pip install）。
    """
    das_root = _repo_root() / "Megatron-LM-das"
    if (das_root / "hcu_megatron" / "megatron_adaptor.py").is_file():
        # 注意：本函数在 upstream 已入 sys.path、typing 垫片已生效之后调用，
        # megatron_adaptor 顶部的 megatron.* 导入才能成功。
        _prepend_sys_path(das_root)
        importlib.import_module("hcu_megatron.megatron_adaptor")
        print(f"[megatron-hygon] 补丁层: hcu_megatron 源码（与上游 0.18.2 配套）: {das_root}")
        return f"hcu_megatron 源码层（与上游 0.18.2 配套）: {das_root}"

    try:
        importlib.import_module("dcu_megatron")
        print("[megatron-hygon] 补丁层: pip 安装的 dcu_megatron（旧命名，若与上游版本不配套会在训练期报错）")
        return "pip 安装的 dcu_megatron（旧命名，注意与上游版本配套性）"
    except ImportError:
        pass

    for wheel in sorted(Path(__file__).resolve().parent.glob("dcu_megatron-*.whl")):
        if str(wheel) not in sys.path:
            sys.path.append(str(wheel))
        try:
            importlib.import_module("dcu_megatron")
        except ImportError:
            continue
        print(f"[megatron-hygon-bootstrap] 环境未安装 dcu_megatron，已从随仓库 wheel 加载: {wheel.name}")
        return f"随仓库 wheel: {wheel.name}"

    raise ImportError(
        "无法加载海光补丁层：未找到 Megatron-LM-das/hcu_megatron 源码，"
        "环境未安装 dcu_megatron，且 tools/ 下无可用 dcu_megatron-*.whl"
    )


def bootstrap(verbose: bool = False) -> dict[str, str]:
    """初始化海光 Megatron 运行环境。

    返回值用于调试和自检，包含仓库根目录、上游源码目录和补丁模块信息。
    """
    global _BOOTSTRAPPED

    repo_root = _repo_root()
    upstream_root = _upstream_root()

    if not upstream_root.is_dir():
        raise FileNotFoundError(
            f"未找到上游 Megatron-LM 源码目录: {upstream_root}"
        )

    # 先把本地 Megatron-LM 源码放到 import 链路前面，保证
    # `megatron.training` 等官方接口存在，随后再由 dcu_megatron 打补丁。
    _prepend_sys_path(upstream_root)

    # 记录环境，便于脚本和排障时直接查看。
    os.environ.setdefault("MEGATRON_PROJECT_ROOT", str(repo_root))
    os.environ.setdefault("MEGATRON_UPSTREAM_ROOT", str(upstream_root))
    os.environ.setdefault("MEGATRON_HYGON_ENABLED", "1")

    # 先加载上游训练接口，再导入海光补丁层。
    importlib.import_module("megatron.training.arguments")

    patch_layer = "（本进程已 bootstrap 过，沿用先前结果）"
    if not _BOOTSTRAPPED:
        patch_layer = _load_hygon_patch_layer()
        _BOOTSTRAPPED = True

    # 补丁生效校验：hcu_megatron 通过 MegatronPatchesManager 就地替换
    # megatron 内部函数（不同于旧版 dcu_megatron 整体接管 parse_args），
    # 因此以补丁注册表非空为校验依据。
    patches_applied = 0
    patch_utils = sys.modules.get("hcu_megatron.patch_utils")
    if patch_utils is not None:
        patches_applied = len(patch_utils.MegatronPatchesManager.patches_info)

    if "hcu_megatron.megatron_adaptor" in sys.modules and patches_applied == 0:
        raise RuntimeError(
            "海光补丁层已导入但未注册任何补丁"
            "（hcu_megatron.patch_utils.MegatronPatchesManager.patches_info 为空），"
            "请检查 hcu_megatron/features_manager 的默认补丁配置"
        )

    info = {
        "repo_root": str(repo_root),
        "upstream_root": str(upstream_root),
        "patch_layer": patch_layer,
        "patches_applied": patches_applied,
    }

    if verbose:
        print("[megatron-hygon-bootstrap] repo_root =", info["repo_root"])
        print("[megatron-hygon-bootstrap] upstream_root =", info["upstream_root"])
        print("[megatron-hygon-bootstrap] patch_layer =", info["patch_layer"])
        print("[megatron-hygon-bootstrap] patches_applied =", info["patches_applied"])

    return info


if __name__ == "__main__":
    bootstrap(verbose=True)
