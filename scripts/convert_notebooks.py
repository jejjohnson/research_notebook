"""Convert Jupytext percent-format notebooks to .ipynb files.

Notebooks in this repository are stored as Jupytext percent-format Python
scripts (.py) instead of raw .ipynb files. This keeps the repository
clean, diff-friendly, and free of committed cell outputs.

Use this script to generate the .ipynb files locally for execution in
JupyterLab or any other Jupyter-compatible environment.

Usage:
    # Convert all notebooks in notebooks/
    pixi run notebooks-to-ipynb

    # Convert a single file
    pixi run jupytext --to notebook notebooks/01_eda.py

    # Pair a .py file with a .ipynb file (keeps both in sync on save)
    pixi run jupytext --set-formats py:percent,ipynb notebooks/01_eda.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def convert_notebooks(
    source_dir: Path,
    *,
    output_format: str = "notebook",
    pattern: str = "*.py",
    verbose: bool = False,
) -> int:
    """Convert Jupytext percent-format scripts to the given output format.

    Args:
        source_dir: Directory containing Jupytext percent-format .py files.
        output_format: Target Jupytext format (e.g. 'notebook' for .ipynb).
        pattern: Glob pattern to match source files within source_dir.
        verbose: If True, print each file being converted.

    Returns:
        Exit code: 0 on success, 1 if any conversion failed.
    """
    source_files = sorted(source_dir.glob(pattern))

    if not source_files:
        print(f"No files matching '{pattern}' found in '{source_dir}'.")
        return 0

    failed: list[Path] = []
    for src in source_files:
        if verbose:
            print(f"  Converting {src} ...")
        result = subprocess.run(
            ["jupytext", "--to", output_format, str(src)],
            capture_output=not verbose,
            text=True,
        )
        if result.returncode != 0:
            print(f"ERROR converting {src}:\n{result.stderr}", file=sys.stderr)
            failed.append(src)
        elif verbose:
            print(f"  -> {src.with_suffix('.ipynb')}")

    if failed:
        print(f"\n{len(failed)} conversion(s) failed.", file=sys.stderr)
        return 1

    print(f"Converted {len(source_files)} notebook(s) to '{output_format}' format.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Jupytext .py notebooks to .ipynb (or other formats).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "source_dir",
        nargs="?",
        default="notebooks",
        help="Directory containing Jupytext .py files (default: notebooks/)",
    )
    parser.add_argument(
        "--to",
        default="notebook",
        metavar="FORMAT",
        help=(
            "Target Jupytext format, e.g. 'notebook' (.ipynb) or 'script'"
            " (default: notebook)"
        ),
    )
    parser.add_argument(
        "--pattern",
        default="*.py",
        help="Glob pattern for source files (default: *.py)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print each file being converted",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    if not source_dir.is_dir():
        print(f"Error: '{source_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    sys.exit(
        convert_notebooks(
            source_dir,
            output_format=args.to,
            pattern=args.pattern,
            verbose=args.verbose,
        )
    )


if __name__ == "__main__":
    main()
