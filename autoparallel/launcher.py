"""通用下放程序启动器。"""

from __future__ import annotations

import os
import subprocess

from autoparallel.types import LoweredProgram


def launch_lowered_program(program: LoweredProgram) -> subprocess.CompletedProcess[str]:
    """执行已经下放好的 backend 程序。"""
    env = os.environ.copy()
    env.update(program.env_overrides)
    return subprocess.run(
        program.command,
        cwd=str(program.project_root),
        env=env,
        text=True,
        check=True,
        capture_output=True,
    )
