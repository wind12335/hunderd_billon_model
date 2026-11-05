"""TE 桩包：屏蔽真实 MT-TransformerEngine（顶层导入直接失败）。

真实 MT-TransformerEngine 的 .so 用 AVX-512 指令集编译，本机海光 CPU
(无 AVX-512) 一加载就 SIGILL(硬崩溃)，try/except 无法捕获。

本桩置于 PYTHONPATH 最前，import transformer_engine 时抛出
ModuleNotFoundError，使 Megatron 全局 HAVE_TE=False，走 apex/torch 回退，
彻底绕开真实 TE。唯一例外是 megatron/core/pipeline_parallel/schedules.py
里一处无保护的硬导入 get_fine_grained_offload_handler，需对该行单独打
try/except 补丁（见 scripts/patch_megatron_musa.sh）。
"""
raise ModuleNotFoundError(
    "MT-TransformerEngine disabled: its compiled extension requires AVX-512, "
    "which this Hygon CPU lacks. autoparallel MUSA build uses local impl."
)
