"""socket 级禁网：沙箱子进程设置 AUTOLAB_NO_NET=1 后生效。

放在 runner 入口第一行调用，任何"顺手联网"的算子/模型（含第三方库）都会被拦下。
"""
from __future__ import annotations

import os
import socket

_installed = False


def install() -> None:
    global _installed
    if _installed or os.environ.get("AUTOLAB_NO_NET") != "1":
        return
    _installed = True

    def _blocked(*_args, **_kwargs):
        raise OSError("AUTOLAB_NO_NET=1：沙箱内禁止网络访问")

    def _blocked_connect(self, address):  # noqa: ANN001
        raise OSError(f"AUTOLAB_NO_NET=1：沙箱内禁止网络访问（socket.connect {address!r}）")

    def _blocked_connect_ex(self, address):  # noqa: ANN001
        raise OSError(f"AUTOLAB_NO_NET=1：沙箱内禁止网络访问（socket.connect_ex {address!r}）")

    socket.socket.connect = _blocked_connect  # type: ignore[method-assign]
    socket.socket.connect_ex = _blocked_connect_ex  # type: ignore[method-assign]
    socket.create_connection = _blocked  # type: ignore[assignment]
    socket.getaddrinfo = _blocked  # type: ignore[assignment]
    socket.gethostbyname = _blocked  # type: ignore[assignment]


def is_network_blocked() -> bool:
    return os.environ.get("AUTOLAB_NO_NET") == "1"
