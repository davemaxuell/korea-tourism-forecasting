# Contributing

Thanks for helping keep this project reproducible and easy to review.

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt
```

## Before Opening a Pull Request

Run the same checks used by CI:

```bash
ruff check src tests
ruff format --check src tests
python src/build_dataset.py
pytest
```

## Data Guidelines

- Do not commit API keys, `.env` files, or private credentials.
- Keep small reproducibility inputs under `data/raw/` when reuse rights allow it.
- Regenerate processed datasets with `python src/build_dataset.py` instead of editing them manually.
- Keep generated model outputs under `outputs/`; these files are ignored by Git by default.

## Reporting Results

If model behavior changes, update `reports/model_results.md` with the new headline metrics and a short interpretation. Prefer concise summaries over committing every generated output file.
