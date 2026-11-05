#!/usr/bin/env python3
"""MUSA 版 Megatron 数据预处理包装器。

preprocess_data.py 的 import 链会拉进 TransformerEngine 的 .so（AVX-512 编译，
本机海光 C86 无 AVX-512 → import 即 SIGILL，try/except 捕不到）。本包装器先执行
megatron_musa_bootstrap.bootstrap()（te_stub 屏蔽 TE + 源码补丁 + musa_patch），
再 runpy 运行 preprocess_data.py。

用法:
  python3 tools/preprocess_musa.py /home/Megatron-LM/tools/preprocess_data.py \
      --input <jsonl> --output-prefix <prefix> --tokenizer-type LlamaTokenizer \
      --vocab-file <tokenizer.model> --append-eod --workers N
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_SELF_DIR = Path(__file__).resolve().parent
if str(_SELF_DIR) not in sys.path:
    sys.path.insert(0, str(_SELF_DIR))

from megatron_musa_bootstrap import bootstrap  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print("用法: python3 tools/preprocess_musa.py <preprocess_data.py> [args ...]")
        return 2

    target = Path(sys.argv[1])
    if not target.is_file() and target.name == "preprocess_data.py":
        candidate = Path("/home/Megatron-LM") / "tools" / "preprocess_data.py"
        if candidate.is_file():
            target = candidate
    if not target.is_file():
        print(f"未找到 preprocess_data.py: {target}", file=sys.stderr)
        return 2

    bootstrap(verbose=False)
    sys.argv = [str(target), *sys.argv[2:]]
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
