#!/usr/bin/env python3
"""Block likely private paths, generated artifacts, and secrets before commit."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


MAX_FINDINGS = 80

_MACOS_HOME_RE = re.compile(b"/" + b"Users" + rb"/[A-Za-z0-9._-]+/")
_SHAREHPC_HOME_RE = re.compile(b"/" + b"share" + b"/" + b"home" + rb"/[A-Za-z0-9._-]+/")
_PUBLIC_HOME_RE = re.compile(b"/" + b"public" + b"/" + b"home" + rb"/[A-Za-z0-9._-]+/")
_CHAT_APP_CONTENT_RE = re.compile(
    rb"(wx"
    + rb"id_[A-Za-z0-9_]+|x"
    + rb"we"
    + rb"chat|We"
    + rb"Ch"
    + rb"at|com\.tencent\.xin"
    + rb"We"
    + rb"Chat)",
    re.I,
)
_CHAT_APP_PATH_RE = re.compile(
    r"(wx"
    + r"id_|x"
    + r"we"
    + r"chat|We"
    + r"Ch"
    + r"at|com\.tencent\.xin"
    + r"We"
    + r"Chat)",
    re.I,
)


@dataclass(frozen=True)
class Finding:
    path: str
    label: str
    line_number: int | None = None
    sample: str | None = None


CONTENT_PATTERNS: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    ("local macOS home path", _MACOS_HOME_RE),
    ("ShareHPC home path", _SHAREHPC_HOME_RE),
    ("public cluster home path", _PUBLIC_HOME_RE),
    ("chat app temp path/id", _CHAT_APP_CONTENT_RE),
    ("private key block", re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("GitHub token", re.compile(rb"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b")),
    ("OpenAI-style API key", re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("Slack token", re.compile(rb"\bxox[baprs]-[A-Za-z0-9-]{20,}\b")),
    ("AWS access key", re.compile(rb"\bAKIA[0-9A-Z]{16}\b")),
    (
        "secret assignment",
        re.compile(
            rb"(?i)\b(api[_-]?key|access[_-]?token|client[_-]?secret|password|secret)\s*[:=]\s*['\"]?[^'\"\s]{8,}"
        ),
    ),
)


FORBIDDEN_PATH_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("local output artifact", re.compile(r"(^|/)output/")),
    ("local dataset artifact", re.compile(r"(^|/)datasets/")),
    ("chat app temp artifact", _CHAT_APP_PATH_RE),
)


FORBIDDEN_SUFFIXES = (
    ".xlsx",
    ".xls",
    ".jsonl",
    ".tar.gz",
    ".tgz",
)


def _run_git(args: list[str]) -> bytes:
    result = subprocess.run(["git", *args], check=True, stdout=subprocess.PIPE)
    return result.stdout


def _staged_paths() -> list[str]:
    raw = _run_git(["diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"])
    return [item.decode("utf-8", "surrogateescape") for item in raw.split(b"\0") if item]


def _tracked_paths() -> list[str]:
    raw = _run_git(["ls-files", "-z"])
    return [item.decode("utf-8", "surrogateescape") for item in raw.split(b"\0") if item]


def _blob_from_index(path: str) -> bytes:
    return _run_git(["show", f":{path}"])


def _blob_from_worktree(path: str) -> bytes:
    return Path(path).read_bytes()


def _line_number(blob: bytes, offset: int) -> int:
    return blob.count(b"\n", 0, offset) + 1


def _line_sample(blob: bytes, offset: int) -> str:
    start = blob.rfind(b"\n", 0, offset) + 1
    end = blob.find(b"\n", offset)
    if end == -1:
        end = len(blob)
    line = blob[start:end].decode("utf-8", "replace").strip()
    return line[:220]


def _path_findings(path: str) -> list[Finding]:
    findings: list[Finding] = []
    for label, pattern in FORBIDDEN_PATH_PATTERNS:
        if pattern.search(path):
            findings.append(Finding(path, label))
    lower = path.lower()
    if any(lower.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES):
        findings.append(Finding(path, "generated/binary artifact extension"))
    return findings


def _content_findings(path: str, blob: bytes) -> list[Finding]:
    findings: list[Finding] = []
    for label, pattern in CONTENT_PATTERNS:
        for match in pattern.finditer(blob):
            findings.append(
                Finding(
                    path,
                    label,
                    _line_number(blob, match.start()),
                    _line_sample(blob, match.start()),
                )
            )
            if len(findings) >= MAX_FINDINGS:
                return findings
    return findings


def scan(paths: list[str], *, staged: bool) -> list[Finding]:
    findings: list[Finding] = []
    for path in paths:
        findings.extend(_path_findings(path))
        try:
            blob = _blob_from_index(path) if staged else _blob_from_worktree(path)
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
        findings.extend(_content_findings(path, blob))
        if len(findings) >= MAX_FINDINGS:
            break
    return findings


def report(findings: list[Finding]) -> None:
    print("Privacy check failed. Remove or redact these staged values before committing:", file=sys.stderr)
    for finding in findings[:MAX_FINDINGS]:
        location = finding.path
        if finding.line_number is not None:
            location += f":{finding.line_number}"
        print(f"- {location}: {finding.label}", file=sys.stderr)
        if finding.sample:
            print(f"  {finding.sample}", file=sys.stderr)
    if len(findings) >= MAX_FINDINGS:
        print(f"- More findings omitted after {MAX_FINDINGS} matches.", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true", help="Scan staged files. This is the pre-commit mode.")
    parser.add_argument("--all-tracked", action="store_true", help="Scan all tracked files in the worktree.")
    args = parser.parse_args()
    staged = True if args.staged or not args.all_tracked else False
    paths = _staged_paths() if staged else _tracked_paths()
    if not paths:
        return 0
    findings = scan(paths, staged=staged)
    if findings:
        report(findings)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
