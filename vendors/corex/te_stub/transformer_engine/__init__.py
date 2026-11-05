"""TE 桩包：屏蔽 TransformerEngine（顶层导入直接失败）。

天数智芯 CoreX 容器未提供 TransformerEngine 适配（无公开天数版 TE），
本训练配方使用 --transformer-impl local，不需要 TE。

本桩置于 PYTHONPATH 最前，import transformer_engine 时抛出
ModuleNotFoundError，使 Megatron 全局 HAVE_TE=False，走 apex/torch 回退
（corex 编译版 apex 的 FusedAdam/FusedLayerNorm 均可用，已实测通过）。
上游 core_v0.14.0rc2 的 TE 导入全部有 try/except 或为函数内懒加载，
stub 即可安全兜底，无需额外源码补丁。
"""
raise ModuleNotFoundError(
    "TransformerEngine disabled: CoreX container ships no Iluvatar TE build. "
    "autoparallel COREX build uses transformer-impl=local with corex apex."
)
