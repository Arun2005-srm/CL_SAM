"""Verify that the official CA-SAM submodule is pinned and unmodified."""

from __future__ import annotations

import subprocess
from pathlib import Path


EXPECTED_REVISION = "9a4ee0f71e264343719a42027d71099ea7ccb8d9"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = REPOSITORY_ROOT / "upstream" / "ca_sam_official"


def git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(UPSTREAM_ROOT), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> None:
    if not (UPSTREAM_ROOT / ".git").exists():
        raise SystemExit(
            "Official CA-SAM submodule is missing. Run: "
            "git submodule update --init --recursive"
        )

    actual_revision = git("rev-parse", "HEAD")
    if actual_revision != EXPECTED_REVISION:
        raise SystemExit(
            "Official CA-SAM revision mismatch.\n"
            f"Expected: {EXPECTED_REVISION}\n"
            f"Actual:   {actual_revision}"
        )

    changes = git("status", "--porcelain", "--untracked-files=all")
    if changes:
        raise SystemExit(
            "Official CA-SAM contains local modifications. Restore the "
            "submodule before running experiments:\n" + changes
        )

    print(f"Official CA-SAM verified: {actual_revision}")


if __name__ == "__main__":
    main()

