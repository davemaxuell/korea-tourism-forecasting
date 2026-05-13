# Public GitHub Checklist

Use this checklist before pushing the repository to GitHub.

1. Install dependencies and rebuild the clean data.

```bash
pip install -r requirements.txt
python src/build_dataset.py
```

2. Confirm no secrets are present in tracked files.

```bash
rg "API_KEY|ECOS_API_KEY|M8ESS|password|secret"
```

3. Initialize Git if this folder is not already a repository.

```bash
git init
git status --short --ignored
```

4. Add only the clean public project files.

```bash
git add README.md requirements.txt requirements-optional.txt .gitignore .env.example
git add src data docs outputs
git status --short
```

5. Commit and push.

```bash
git commit -m "Clean reproducible tourism analysis project"
git branch -M main
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```

The legacy exploratory folders are intentionally ignored by `.gitignore`.
