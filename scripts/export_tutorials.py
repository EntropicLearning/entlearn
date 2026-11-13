"""Render the tutorial notebooks with the checkout's installed entlearn."""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Export all tutorials, failing the build if any notebook fails."""
    root = Path(__file__).resolve().parents[1]
    destination = root / "docs_site" / "tutorials" / "notebooks"
    destination.mkdir(parents=True, exist_ok=True)
    for notebook in sorted((root / "examples").glob("*.py")):
        print(f"Rendering {notebook.name}", flush=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "marimo",
                "export",
                "html",
                "--no-sandbox",
                "--force",
                str(notebook),
                "-o",
                str(destination / f"{notebook.stem}.html"),
            ],
            cwd=root,
            check=True,
        )


if __name__ == "__main__":
    main()
