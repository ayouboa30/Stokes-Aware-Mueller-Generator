"""Fail on common credentials, local paths, identifiers and large artefacts."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".cff",
    ".csv",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}
FORBIDDEN_SUFFIXES = {
    ".ckpt",
    ".h5",
    ".hdf5",
    ".mat",
    ".npy",
    ".npz",
    ".onnx",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
}
PATTERNS = {
    "absolute Windows user path": re.compile(r"[A-Za-z]:[\\/]Users[\\/]", re.I),
    "absolute remote workspace path": re.compile(r"(?<!\w)/workspace/"),
    "SSH command": re.compile(r"\bssh\s+(?:-[A-Za-z]\s+\S+\s+)*[^\n]+@", re.I),
    "private key": re.compile(r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY"),
    "credential assignment": re.compile(
        r"\b(?:api[_-]?key|access[_-]?token|secret[_-]?key|password)\s*[:=]\s*['\"]?[^\s'\"]+",
        re.I,
    ),
    "patient-like code": re.compile(r"\b\d{3}-\d{4}-[A-Z]{2}\b"),
    "unresolved licence author": re.compile(r"Copyright \(c\).*\[Auteur\]", re.I),
}


def tracked_files() -> list[Path]:
    excluded_directories = {
        ".git",
        ".pytest_cache",
        ".ruff_cache",
        ".tmp",
        "__pycache__",
        "build",
        "dist",
    }

    def fallback() -> list[Path]:
        return [
            path
            for path in ROOT.rglob("*")
            if path.is_file()
            and not any(
                part in excluded_directories or part.endswith(".egg-info") for part in path.parts
            )
        ]

    try:
        top_level = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if Path(top_level).resolve() != ROOT.resolve():
            return fallback()
        output = subprocess.check_output(
            ["git", "-C", str(ROOT), "ls-files"], text=True, stderr=subprocess.DEVNULL
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return fallback()
    files = [ROOT / line for line in output.splitlines() if line]
    return files or fallback()


def main() -> int:
    findings: list[str] = []
    for path in tracked_files():
        relative = path.relative_to(ROOT)
        if not path.exists():
            continue
        if path.is_symlink():
            findings.append(f"{relative}: symbolic link requires manual review")
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            findings.append(f"{relative}: forbidden binary artefact")
        if path.stat().st_size > 5 * 1024 * 1024:
            findings.append(f"{relative}: file exceeds 5 MiB")
        if (
            path.suffix.lower() not in TEXT_SUFFIXES
            or relative.as_posix() == "tools/audit_release.py"
        ):
            continue
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            for label, pattern in PATTERNS.items():
                if pattern.search(line):
                    findings.append(f"{relative}:{line_number}: {label}")
    forbidden_names = {"CLAUDE.md", "AGENTS.md", ".env"}
    for path in tracked_files():
        if path.name in forbidden_names:
            findings.append(f"{path.relative_to(ROOT)}: private agent or environment file")
    if findings:
        print("Release audit failed:")
        print("\n".join(f"- {finding}" for finding in findings))
        return 1
    print(f"Release audit passed for {len(tracked_files())} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
