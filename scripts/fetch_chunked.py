#!/usr/bin/env python3
"""经代理分段下载大文件：代理连接不稳定时，用 Range 分块 + 重试拿完整文件。
用法: fetch_chunked.py <url> <dest> [chunk_mb=8] [conn=8]
"""
import os, sys, time
import urllib.request, urllib.error

URL, DEST = sys.argv[1], sys.argv[2]
CHUNK = int(sys.argv[3]) if len(sys.argv) > 3 else 8  # MB
CONN = int(sys.argv[4]) if len(sys.argv) > 4 else 8

def head(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])

total = head(URL)
print(f"total {total} bytes, chunk {CHUNK}MB conn {CONN}", flush=True)

ranges = [(s, min(total - 1, s + CHUNK * 1024 * 1024 - 1)) for s in range(0, total, CHUNK * 1024 * 1024)]

def fetch(s, e, idx, tries=60):
    for t in range(tries):
        try:
            req = urllib.request.Request(URL, headers={"Range": f"bytes={s}-{e}"})
            with urllib.request.urlopen(req, timeout=180) as r:
                data = r.read()
            if len(data) != e - s + 1:
                raise OSError(f"short read {len(data)} != {e-s+1}")
            with open(DEST + f".{idx}", "wb") as f:
                f.write(data)
            print(f"chunk {idx} {s}-{e} done", flush=True)
            return
        except Exception as ex:
            if t % 10 == 9:
                print(f"chunk {idx} try {t}: {ex}", flush=True)
            time.sleep(1 + (t % 5))
    raise RuntimeError(f"chunk {idx} failed after {tries} tries")

from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(CONN) as ex:
    list(ex.map(lambda i: fetch(*ranges[i], i), range(len(ranges))))

with open(DEST, "wb") as out:
    for i in range(len(ranges)):
        with open(DEST + f".{i}", "rb") as f:
            out.write(f.read())
        os.remove(DEST + f".{i}")
print(f"done: {DEST}", flush=True)
