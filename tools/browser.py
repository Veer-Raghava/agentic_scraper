"""Lightweight browser pool abstraction.

Provides a stable interface used by agents. Uses requests-based fallback.
"""

from __future__ import annotations

import requests


class BrowserPool:
    def __init__(self) -> None:
        self.started = False
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        })

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False
        self.session.close()

    def get(self, url: str, timeout: int = 30) -> str:
        r = self.session.get(url, timeout=timeout)
        r.raise_for_status()
        return r.text
