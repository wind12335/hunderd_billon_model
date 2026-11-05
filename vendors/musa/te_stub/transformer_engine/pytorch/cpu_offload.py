"""get_fine_grained_offload_handler 的 no-op 桩。

真实实现来自 TE，用于流水线/CPU offload。本机 PP=1、不开 CPU offload，
该 handler 不会在运行时被调用，因此直接返回一个全 no-op 对象即可。
"""


class _NoOpFineGrainedOffloadHandler:
    def start_microbatch_forward(self, *args, **kwargs):
        return None

    def start_microbatch_backward(self, *args, **kwargs):
        return None


def get_fine_grained_offload_handler():
    return _NoOpFineGrainedOffloadHandler()
