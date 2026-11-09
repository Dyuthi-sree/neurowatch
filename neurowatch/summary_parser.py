"""Parse chbXX-summary.txt files into per-file seizure annotations."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_FILE = re.compile(r"^File Name:\s*(\S+)")
_FSTART = re.compile(r"^File Start Time:\s*(\S+)")
_FEND = re.compile(r"^File End Time:\s*(\S+)")
_NSZ = re.compile(r"^Number of Seizures in File:\s*(\d+)")
_SZS = re.compile(r"^Seizure\s*(\d*)\s*Start Time:\s*(\d+(?:\.\d+)?)\s*seconds")
_SZE = re.compile(r"^Seizure\s*(\d*)\s*End Time:\s*(\d+(?:\.\d+)?)\s*seconds")


@dataclass
class FileEntry:
    file: str
    start_clock: str | None = None
    end_clock: str | None = None
    n_seizures_declared: int | None = None
    starts: list = field(default_factory=list)
    ends: list = field(default_factory=list)

    @property
    def seizures(self) -> list[tuple[float, float]]:
        return list(zip(self.starts, self.ends))


def parse_summary_text(text: str) -> list[FileEntry]:
    entries: list[FileEntry] = []
    cur = None
    for raw in text.splitlines():
        line = raw.strip()
        if m := _FILE.match(line):
            cur = FileEntry(file=m.group(1))
            entries.append(cur)
        elif cur is None:
            continue
        elif m := _FSTART.match(line):
            cur.start_clock = m.group(1)
        elif m := _FEND.match(line):
            cur.end_clock = m.group(1)
        elif m := _NSZ.match(line):
            cur.n_seizures_declared = int(m.group(1))
        elif m := _SZS.match(line):
            cur.starts.append(float(m.group(2)))
        elif m := _SZE.match(line):
            cur.ends.append(float(m.group(2)))
    for e in entries:
        if len(e.starts) != len(e.ends):
            raise ValueError(f"{e.file}: {len(e.starts)} starts vs {len(e.ends)} ends")
        if e.n_seizures_declared is not None and e.n_seizures_declared != len(e.starts):
            raise ValueError(f"{e.file}: declared {e.n_seizures_declared}, found {len(e.starts)}")
        for s, t in e.seizures:
            if t <= s:
                raise ValueError(f"{e.file}: seizure end {t} <= start {s}")
    return entries


def parse_summary(path: Path) -> list[FileEntry]:
    return parse_summary_text(Path(path).read_text(errors="replace"))
