"""Read members of a remote ZIP archive over HTTP range requests.

The RxCADRE archives are 0.7-0.9 GB each but a single burn unit is a small
fraction of that.  `zipfile` only needs a seekable file object, so we give it
one backed by HTTP `Range:` requests with a read-ahead buffer.  Each member is
written atomically (``.part`` then rename), which makes downloads resumable at
file granularity.
"""
from __future__ import annotations

import io
import os
import time
import zipfile
from pathlib import Path

import requests

USER_AGENT = "rxcadre-spread-pipeline/0.1 (research data client)"


class HTTPRangeFile(io.RawIOBase):
    """Seekable, read-only file object over an HTTP resource."""

    def __init__(self, url: str, block: int = 4 << 20, session: requests.Session | None = None):
        self.url = url
        self.block = block
        self.s = session or requests.Session()
        self.s.headers["User-Agent"] = USER_AGENT
        r = self.s.head(url, allow_redirects=True, timeout=60)
        if r.ok and r.headers.get("accept-ranges", "").lower() == "bytes":
            self.size = int(r.headers["content-length"])
            self.url = r.url
        else:  # e.g. signed cloud-storage URLs that only allow GET: probe with a 1-byte range
            g = self.s.get(url, headers={"Range": "bytes=0-0"}, allow_redirects=True, timeout=60)
            if g.status_code != 206 or "content-range" not in g.headers:
                raise IOError(f"server does not support range requests: {url} ({g.status_code})")
            self.size = int(g.headers["content-range"].split("/")[-1])
            self.url = g.url
        self.pos = 0
        self._buf_start = -1
        self._buf = b""
        self.bytes_fetched = 0

    # io.RawIOBase protocol -------------------------------------------------
    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        self.pos = max(0, min(self.pos, self.size))
        return self.pos

    def _fetch(self, start: int, end: int) -> bytes:
        for attempt in range(6):
            try:
                r = self.s.get(self.url, headers={"Range": f"bytes={start}-{end - 1}"}, timeout=120)
                if r.status_code != 206:
                    raise IOError(f"expected 206, got {r.status_code}")
                self.bytes_fetched += len(r.content)
                return r.content
            except (requests.RequestException, IOError):
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError("unreachable")

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        out = bytearray()
        while n > 0:
            in_buf = self._buf_start <= self.pos < self._buf_start + len(self._buf)
            if not in_buf:
                start = self.pos
                end = min(self.size, start + max(self.block, n))
                self._buf = self._fetch(start, end)
                self._buf_start = start
            off = self.pos - self._buf_start
            chunk = self._buf[off: off + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def list_remote_zip(url: str) -> list[zipfile.ZipInfo]:
    with zipfile.ZipFile(HTTPRangeFile(url, block=256 << 10)) as zf:
        return zf.infolist()


def _safe_target(dest_root: Path, member: str) -> Path:
    """Map an archive member to a path inside dest_root (no traversal)."""
    parts = [p for p in member.replace("\\", "/").split("/") if p not in ("", ".", "..")]
    target = dest_root.joinpath(*parts).resolve()
    if not str(target).startswith(str(dest_root.resolve())):
        raise ValueError(f"unsafe member path: {member}")
    return target


def extract_members(url: str, members, dest_root: Path, progress=None) -> list[Path]:
    """Extract selected members (names or ZipInfo) into dest_root, skipping
    files that already exist with the right size."""
    dest_root = Path(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)
    f = HTTPRangeFile(url)
    out = []
    with zipfile.ZipFile(f) as zf:
        infos = [zf.getinfo(m) if isinstance(m, str) else m for m in members]
        # extract in archive order so the read-ahead buffer is used sequentially
        infos.sort(key=lambda i: i.header_offset)
        for i, info in enumerate(infos):
            if info.is_dir():
                continue
            target = _safe_target(dest_root, info.filename)
            out.append(target)
            if target.exists() and target.stat().st_size == info.file_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(target.suffix + ".part")
            with zf.open(info) as src, open(tmp, "wb") as dst:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    dst.write(chunk)
            os.replace(tmp, target)
            if progress:
                progress(i + 1, len(infos), info.filename)
    return out
