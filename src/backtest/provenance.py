"""Records that tie a report to the exact code, data and settings that produced it."""
import hashlib
import subprocess
from pathlib import Path


def content_sha256(path: Path) -> str:
    """SHA-256 of the file with CRLF normalized to LF.

    Git stores text with LF but a Windows checkout (and pandas on Windows) writes
    CRLF, so hashing the raw bytes would give different values on different OSes.
    """
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def git_state(root: Path) -> dict:
    """Commit checked out and whether tracked or untracked files differ from it (reports/ excluded).

    Call before writing any report, so the run's own outputs do not count as changes.
    """
    def git(*args) -> str:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout

    try:
        return {
            "git_commit": git("rev-parse", "HEAD").strip(),
            "git_dirty": bool(git("status", "--porcelain", "--", ".", ":(exclude)reports").strip()),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}
