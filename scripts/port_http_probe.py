#!/usr/bin/env python
from __future__ import annotations

import argparse
import socket
import urllib.request


def check_port(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex((host, port)) == 0


def check_http(host: str, port: int):
    url = f"http://{host}:{port}/"
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            return True, r.status, None
    except Exception as e:  # noqa: BLE001
        return False, None, repr(e)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe de porta/HTTP do dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()

    open_flag = check_port(args.host, args.port)
    print(f"PORT {args.host}:{args.port} OPEN={open_flag}")
    ok, status, err = check_http(args.host, args.port)
    if ok:
        print(f"HTTP {args.host}:{args.port} STATUS={status}")
    else:
        print(f"HTTP {args.host}:{args.port} FAIL={err}")


if __name__ == "__main__":
    main()
