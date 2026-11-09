#!/usr/bin/env python
"""Resumable downloader for the CHB-MIT Scalp EEG Database (PhysioNet, v1.0.0).

Examples
--------
  # metadata only (RECORDS, SUBJECT-INFO, all 24 summary files)
  python scripts/download_data.py --metadata-only
  # EDF headers only for every recording (a few KB each; used for the channel/duration scan)
  python scripts/download_data.py --headers-only --patients all
  # full EDFs for a subset
  python scripts/download_data.py --patients chb01 chb02 chb03
  # everything (~43 GB), 4 parallel transfers
  python scripts/download_data.py --patients all --workers 4

Resuming: partially downloaded files are kept as `<name>.part` and continued with an
HTTP Range request (or `wget -c` if wget is installed). Completed files are verified
against the server's Content-Length and skipped on re-runs.
"""
from __future__ import annotations

import argparse
import shutil
import struct
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = "https://physionet.org/files/chbmit/1.0.0/"
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
HDR = ROOT / "data" / "headers"
TOP_FILES = ["RECORDS", "RECORDS-WITH-SEIZURES", "SUBJECT-INFO"]
ALL_CASES = [f"chb{i:02d}" for i in range(1, 25)]
UA = {"User-Agent": "neurowatch-downloader/1.0"}


def _req(url: str, headers: dict | None = None, method: str = "GET"):
    h = dict(UA)
    h.update(headers or {})
    return urllib.request.urlopen(urllib.request.Request(url, headers=h, method=method), timeout=60)


def remote_size(url: str) -> int | None:
    try:
        with _req(url, method="HEAD") as r:
            cl = r.headers.get("Content-Length")
            return int(cl) if cl is not None else None
    except Exception:
        return None


def download(url: str, dest: Path, retries: int = 5) -> str:
    """Download url -> dest, resuming from dest.part. Returns a status string."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = remote_size(url)
    if dest.exists() and (size is None or dest.stat().st_size == size):
        return "skip"
    part = dest.with_suffix(dest.suffix + ".part")
    if dest.exists():  # wrong size -> treat as partial
        dest.rename(part)
    for attempt in range(1, retries + 1):
        try:
            if shutil.which("wget"):
                subprocess.run(["wget", "-q", "-c", "-O", str(part), url], check=True)
            else:
                have = part.stat().st_size if part.exists() else 0
                if size is not None and have >= size:
                    pass
                else:
                    hdr = {"Range": f"bytes={have}-"} if have else {}
                    with _req(url, hdr) as r:
                        mode = "ab" if (have and r.status == 206) else "wb"
                        with open(part, mode) as f:
                            shutil.copyfileobj(r, f, length=1 << 20)
            got = part.stat().st_size
            if size is not None and got != size:
                raise IOError(f"size mismatch {got} != {size}")
            part.rename(dest)
            return "ok"
        except Exception as e:  # network hiccup -> retry, keeping the .part
            if attempt == retries:
                return f"FAIL ({e})"
            time.sleep(2 * attempt)
    return "FAIL"


def fetch_header(url: str, dest: Path) -> str:
    """Fetch only the EDF header (256 + 256*ns bytes) via Range requests."""
    if dest.exists() and dest.stat().st_size > 256:
        return "skip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 6):
        try:
            with _req(url, {"Range": "bytes=0-255"}) as r:
                first = r.read(256)
            ns = int(first[252:256].decode("ascii").strip())
            n = 256 + 256 * ns
            with _req(url, {"Range": f"bytes=0-{n - 1}"}) as r:
                data = r.read(n)
            if len(data) != n:
                raise IOError("short header")
            dest.write_bytes(data)
            return "ok"
        except Exception as e:
            if attempt == 5:
                return f"FAIL ({e})"
            time.sleep(2 * attempt)
    return "FAIL"


def read_records() -> list[str]:
    return [l.strip() for l in (RAW / "RECORDS").read_text().splitlines() if l.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patients", nargs="+", default=["chb01", "chb02", "chb03"],
                    help="case ids (chb01 ...) or 'all'")
    ap.add_argument("--metadata-only", action="store_true")
    ap.add_argument("--headers-only", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)

    cases = ALL_CASES if a.patients == ["all"] else a.patients
    bad = [c for c in cases if c not in ALL_CASES]
    if bad:
        sys.exit(f"unknown case ids: {bad}")

    # metadata: always all 24 summaries (tiny) so counts can be computed on the whole database
    for f in TOP_FILES:
        print(f, download(BASE + f, RAW / f))
    for c in ALL_CASES:
        download(BASE + f"{c}/{c}-summary.txt", RAW / c / f"{c}-summary.txt")
    print("summaries: done")
    if a.metadata_only:
        return

    recs = [r for r in read_records() if r.split("/")[0] in cases]
    jobs = []
    with ThreadPoolExecutor(a.workers) as ex:
        for r in recs:
            if a.headers_only:
                jobs.append(ex.submit(fetch_header, BASE + r, HDR / r))
            else:
                jobs.append(ex.submit(download, BASE + r, RAW / r))
        status = {}
        for i, (r, j) in enumerate(zip(recs, jobs)):
            s = j.result()
            status[r] = s
            if not s.startswith("skip"):
                print(f"[{i + 1}/{len(recs)}] {r}: {s}", flush=True)
    fails = [r for r, s in status.items() if s.startswith("FAIL")]
    print(f"{len(recs)} files, {sum(s == 'ok' for s in status.values())} downloaded, "
          f"{sum(s == 'skip' for s in status.values())} already present, {len(fails)} failed")
    if fails:
        print("failed (re-run to resume):", *fails, sep="\n  ")
        sys.exit(1)


if __name__ == "__main__":
    main()
