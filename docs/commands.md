# Commands cheat sheet

Run everything from the repo root (`~/lab/cv-journey`). Relative paths in `config/hit-uav.yaml` depend on it.

## Environment (uv)

| Command | What it does |
|---|---|
| `uv sync` | Install the exact versions from `uv.lock` into `.venv` |
| `uv add <pkg>` | Add a dependency (updates `pyproject.toml` + `uv.lock`) |
| `uv run python -m cv_journey.<script>` | Run one of my scripts, e.g. `check_env`, `detect`, `show_labels` |
| `uv run <tool>` | Run a tool installed in `.venv`, e.g. `uv run yolo ...` |

## YOLO (Ultralytics CLI)

```bash
# Train: fine-tune a pretrained model on HIT-UAV. Results go to runs/detect/<name>/
uv run yolo detect train data=config/hit-uav.yaml model=yolo11n.pt epochs=100 imgsz=640 batch=16 name=n640

# Validate: metrics + per-class table + plots for one trained model (val split)
uv run yolo detect val model=runs/detect/n640/weights/best.pt data=config/hit-uav.yaml imgsz=640 name=n640-val

# Final score on the TEST split. Run it ONCE, only on the chosen model
uv run yolo detect val model=runs/detect/<chosen>/weights/best.pt data=config/hit-uav.yaml split=test name=<chosen>-test

# Predict on images or a folder, saving the drawn boxes
uv run yolo detect predict model=runs/detect/n640/weights/best.pt source=<image-or-folder>
```

| Argument | Meaning |
|---|---|
| `model=` | Starting weights: `yolo11n.pt` (nano), `yolo11s.pt` (small), ... or my own `best.pt` |
| `imgsz=` | Input size in pixels. Use the same value for val as for train |
| `epochs=` / `batch=` | Number of passes over the training set / images per weight update |
| `name=` | Output folder name under `runs/detect/` |
| `split=` | Which split `val` uses: `val` (default) or `test` |
| `conf=` | Minimum confidence for a box to count. Lower → more recall, less precision |

Useful files in each run folder: `results.csv` (metrics per epoch), `results.png`, `BoxPR_curve.png`,
`confusion_matrix.png`, `weights/best.pt`.

## GPU monitoring

| Command | What it does |
|---|---|
| `nvidia-smi` | GPU usage, memory, temperature (one snapshot) |
| `watch -n 2 nvidia-smi` | Same, refreshed every 2 s (Ctrl+C to exit) |

## Git (my branch workflow)

| Command | What it does |
|---|---|
| `git status` / `git diff` | What changed, and how |
| `git switch -c <branch>` | Create a branch and move to it |
| `git add <files>` then `git commit` | Stage and commit (short subject, blank line, body) |
| `git commit --amend` | Fix the last commit (before pushing, or then push with the next line) |
| `git push --force-with-lease` | Safe force-push after amending |
| `git push -u origin <branch>` | First push of a new branch |
| `git switch main && git merge <branch> && git push` | Merge a finished branch into main |
| `git log --oneline -10` | Last 10 commits, one per line |

## Shell tricks

```bash
# Count objects per class in YOLO labels (first column = class ID)
cat data/hit-uav/yolo_labels/train/*.txt | awk '{print $1}' | sort | uniq -c

# Count images per day/night digit (1st field of the filename)
ls data/hit-uav/normal_json/train | cut -d_ -f1 | sort | uniq -c

# Brace expansion: one command over several splits
ls data/hit-uav/normal_json/{train,val,test} | wc -l
```

## Viewing

| Command | What it does |
|---|---|
| `xdg-open <file>` | Open an image/PDF/HTML with the default app |
| VS Code `Ctrl+Shift+V` | Markdown preview (Mermaid needs the `bierner.markdown-mermaid` extension) |
