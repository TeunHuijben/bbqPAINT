#!/usr/bin/env python3
"""
Build Jupyter notebooks from raw Python files.

This script converts the raw Python example files to .ipynb notebooks:
- Normal version (src/mcoast/examples/*.ipynb): colab cells removed
- Colab version (src/mcoast/examples/colab/*.ipynb): all cells, pip install uncommented

Usage:
    python build_notebooks.py
"""

import json
import subprocess
import sys
from pathlib import Path


def uncomment_colab_cells(input_path, output_path):
    """Uncomment pip install commands in colab-tagged cells."""
    with open(input_path) as f:
        nb = json.load(f)

    for cell in nb["cells"]:
        if "colab" in cell.get("metadata", {}).get("tags", []):
            cell["source"] = [line.replace("# !pip", "!pip") for line in cell["source"]]

    with open(output_path, "w") as f:
        json.dump(nb, f, indent=1)
        f.write("\n")


def main():
    # Get paths
    raw_dir = Path(__file__).parent
    examples_dir = raw_dir.parent
    colab_dir = examples_dir / "colab"
    tmp_dir = Path("/tmp")

    # Create colab directory
    colab_dir.mkdir(exist_ok=True)

    # Find all raw Python files
    raw_files = sorted(raw_dir.glob("*.py"))
    raw_files = [f for f in raw_files if f.name != "build_notebooks.py"]

    if not raw_files:
        print("No Python files found to convert")
        return

    print(f"Building {len(raw_files)} notebook(s)...\n")

    for raw_file in raw_files:
        filename = raw_file.stem
        print(f"Converting {filename}...")

        tmp_notebook = tmp_dir / f"{filename}.ipynb"
        colab_notebook = colab_dir / f"{filename}.ipynb"

        # Convert to full notebook
        subprocess.run(
            [
                "jupytext",
                "--to",
                "notebook",
                "--output",
                str(tmp_notebook),
                str(raw_file),
            ],
            check=True,
            capture_output=True,
        )

        # Normal version (remove colab cells)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "nbconvert",
                str(tmp_notebook),
                "--TagRemovePreprocessor.enabled=True",
                "--TagRemovePreprocessor.remove_cell_tags",
                "colab",
                "--to",
                "notebook",
                "--output",
                f"{filename}.ipynb",
                "--output-dir",
                str(examples_dir),
                "--log-level=WARN",
            ],
            check=True,
            capture_output=True,
        )

        # Colab version (keep all cells, uncomment pip install)
        uncomment_colab_cells(tmp_notebook, colab_notebook)

        print("  ✓ Created normal and Colab versions")

    print("\n✓ All notebooks built successfully")
    print(f"  Normal: {examples_dir}/*.ipynb")
    print(f"  Colab:  {colab_dir}/*.ipynb")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"\nError: {e}", file=sys.stderr)
        if e.stderr:
            print(e.stderr.decode(), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
