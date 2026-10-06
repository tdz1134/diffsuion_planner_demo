---
name: diffusion-planner-workflow
description: Run and modify the occupancy-grid diffusion path planner and map VAE in this repo. Covers the migrated venv (../venv_py38, invoked by absolute path, GPU RTX 4060), the make_dataset -> diffusion_path -> vae_map pipeline, tqdm progress bars for long jobs, dataset_<M>.npz caching, figs/ output layout, and the --quick smoke test. Use when building/regenerating data, training the planner or VAE, running or debugging these scripts, or when the user mentions diffusion_path, vae_map, make_dataset, grid_env, or the venv.
---

# Diffusion Planner — Run & Modify Workflow

Guidance for the training pipeline in `diffusion_planner/` (conditional DDPM path
planner + occupancy-map VAE). Long steps are meant to be **run by the user** with
tqdm progress bars; the agent should wire scripts, keep bars, and avoid blocking.

## Environment (important)

- venv lives **outside** the code folder, as a sibling: `../venv_py38`
  (abs: `/home/t/projects/diffusion_planner_demo/venv_py38`).
- It was migrated from another machine; `source activate` may still be fragile.
  **Always call python by absolute path** instead of activating:
  ```bash
  cd diffusion_planner
  ../venv_py38/bin/python <script>.py ...
  ```
- torch 2.4.1+cu121, CUDA available on RTX 4060. Scripts auto-pick `cuda` if present
  and print `[info] device = cuda`.
- tqdm is installed (4.70.x). Long loops MUST keep a tqdm bar.

## Pipeline & files

| File | Role | Has `main`? |
|------|------|-------------|
| `grid_env.py`  | shared lib: map gen, A*/weighted-A*/greedy, resample, SDF distance field | no (library) |
| `make_dataset.py` | **slowest step**: builds expert data → `dataset_<M>.npz` | yes (`--data --seed --overwrite`) |
| `diffusion_path.py` | conditional DDPM planner: train + sample + render | yes (`--data --train --regen --quick`) |
| `vae_map.py`   | occupancy-map VAE (spatial latent 8×8×8 + residual blocks) | yes (no CLI args; edit config constants) |
| `diffusion_toy.py` | 2D two-moons DDPM intro demo | yes |

Run order: **make_dataset → diffusion_path → vae_map** (planner & VAE reuse the same
`dataset_<M>.npz`; VAE uses channel 0 = occupancy).

## Data & output conventions

- Cache file is named **by sample count**: `dataset_<M>.npz`
  (e.g. `dataset_20000.npz`, `dataset_1500.npz`). Keys: `maps` (M,2,64,64),
  `starts`,`goals` (M,2), `paths` (M,32,2). Maps are 2 channels `[occupancy, normalized SDF]`.
- `diffusion_path`/`vae_map` auto-load the cache; if missing they call
  `make_dataset.generate(...)` on the fly. **Prefer pre-generating** so the long
  data step is a separate, visible progress bar.
- All figures go to **`figs/<script_name>/`** (auto-created):
  - `figs/diffusion_path/fig_planner.png`
  - `figs/vae_map/fig_vae.png`
  - `figs/diffusion_toy/{fig_forward.png,fig_generated.png,fig_denoise.gif}`
- Checkpoints stay in repo root: `planner_ckpt.pt`, `vae_map.pt`.

## Commands (user runs the long ones)

```bash
cd diffusion_planner

# 1) build dataset (slowest, ~14 maps/s; 20k ≈ 20+ min) — one bar, run manually
../venv_py38/bin/python make_dataset.py --data 20000

# 2) train planner + render (long, ~40k steps)
../venv_py38/bin/python diffusion_path.py --data 20000 --train 40000

# 3) train map VAE (reuses dataset; ~minutes)
../venv_py38/bin/python vae_map.py

# fast smoke test (seconds–1 min; quality is NOT representative)
../venv_py38/bin/python diffusion_path.py --quick     # data<=1500, train<=3000
```

Regenerate cache: `make_dataset.py --data M --overwrite`, or
`diffusion_path.py --regen`.

## Working rules for the agent

1. **Do not block on long training.** Write/adjust the code, then hand the user the
   exact command above to run manually. Only run `--quick` yourself for a smoke test.
2. **Keep tqdm on every long loop** (dataset generation, each training epoch/step).
3. **Log buffering gotcha:** when a script's output is redirected to a file
   (`... > log 2>&1`), tqdm/`tqdm.write` lines are block-buffered and may only appear
   at process exit. To read progress, either run in a terminal (user) or use
   `stdbuf -oL ../venv_py38/bin/python <script>.py`, or just wait for completion.
4. **Never hardcode the old path** `/home/t/zizhu/...`. venv abs path is
   `/home/t/projects/diffusion_planner_demo/venv_py38`.
5. Plots use **English labels** (no CJK font installed → Chinese titles render as boxes).
6. Coordinate convention: waypoint `(x,y) = (col,row)`, normalized to `[-1,1]` via
   `x = c/(size-1)*2-1`. Endpoints are pinned each denoise step.

## Known state / caveats

- Planner collision rate is modest (~5/8, length ratio ~0.98–1.02): it sometimes
  shortcuts through obstacles. Planned fixes (not yet done): C-space obstacle
  inflation, per-waypoint local-SDF conditioning, obstacle-avoidance loss /
  sampling guidance, stronger `repair_path`.
- VAE was upgraded from a 32-d global vector to a **spatial latent (8×8×8) +
  residual blocks + weighted BCE (pos_weight=3) + low KL (beta=1e-2)**, ~5M params.
  U-Net skip connections were deliberately **not** used to avoid latent collapse
  (keeps the latent usable for a future latent-diffusion).
