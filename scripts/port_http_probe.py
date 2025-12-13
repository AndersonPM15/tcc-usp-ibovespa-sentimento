#!/usr/bin/env python
from __future__ import annotations

import socket
import urllib.request


def check_port(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex((host, port)) == 0


def check_http(url: str):
    try:
        with urllib.request.urlopen(url, timeout=2) as r:
            return True, r.status, None
    except Exception as e:  # noqa: BLE001
        return False, None, repr(e)


def main() -> None:
    targets = [("127.0.0.1", 8050), ("127.0.0.1", 8060)]
    for host, port in targets:
        open_flag = check_port(host, port)
        print(f"PORT {host}:{port} OPEN={open_flag}")
    ok, status, err = check_http("http://127.0.0.1:8050/")
    if ok:
        print(f"HTTP 8050 STATUS={status}")
    else:
        print(f"HTTP 8050 FAIL={err}")


if __name__ == "__main__":
    main()
