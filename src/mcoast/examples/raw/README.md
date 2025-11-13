# Raw Example Files (Developer Source)

This directory contains the **source files** for the mCOAST example notebooks in jupytext format.

## For Developers

These are the files you should edit when updating examples:
- `1_basic_simulation.py`
- `2_simulated_data.py`
- `3_experimental_data.py`

### File Format

These files use jupytext format with `# %%` cell markers:
- They work as regular Python scripts (can run from command line)
- They work as interactive notebooks in VS Code, PyCharm, etc.
- They get automatically converted to `.ipynb` files by GitHub Actions

### Workflow

1. **Edit** these `.py` files
2. **Test** by running as scripts: `python 1_basic_simulation.py`
3. **Commit and push** to the main branch
4. **GitHub Action automatically**:
   - Converts them to `.ipynb` notebooks
   - Saves to `../notebooks/` directory
   - Commits the generated notebooks

### Local Testing

Test the jupytext conversion locally:
```bash
# Install jupytext
pip install jupytext

# Convert to notebook
jupytext --to notebook 1_basic_simulation.py -o 1_basic_simulation.ipynb

# Run as Python script
python 1_basic_simulation.py
```

## For Users

Users can use the **auto-generated notebooks** in `/examples/` instead of these raw files.
