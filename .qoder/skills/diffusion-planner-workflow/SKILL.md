---
name: diffusion-planner-workflow
description: Run and modify the two conditional-DDPM planners in this repo - the occupancy-grid path planner and map VAE in diffusion_planner/, and the parking planner in parking/ (Hybrid A* teacher -> dataset -> map VAE -> SE(2) trajectory diffusion -> repair/evaluate). Covers the migrated venv (venv_py38, invoked by absolute path, GPU RTX 4060, numba 0.58.1), the make_dataset -> diffusion_path -> vae_map and dataset -> map_vae -> train -> evaluate pipelines, tqdm progress bars, npz caching, figs/ layout, numba JIT warm-up, and --quick smoke tests. Use when building/regenerating data, training the planner/VAE/diffusion, running or debugging these scripts, or when the user mentions diffusion_path, vae_map, make_dataset, grid_env, parking, hybrid_a_star, HA*, scenarios, repair, evaluate, or the venv.
---

# Diffusion Planner — Run & Modify Workflow

Guidance for the two conditional-DDPM planners in this repo:
- **`diffusion_planner/`** - occupancy-grid path planner + map VAE (the demo, sections below).
- **`parking/`** - parking planner (Hybrid A* teacher -> dataset -> map VAE -> SE(2)
  trajectory diffusion -> repair/evaluate). See the "Parking planner" section; the full
  narrative lives in `PARKING_NOTES.md`.

Long steps are meant to be **run by the user** with tqdm progress bars; the agent
should wire scripts, keep bars, and avoid blocking.

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
- `parking/` additionally needs **numba 0.58.1** (installed in the venv) for the Hybrid A*
  hot loop; without it the search falls back to slow pure-Python (~240x slower, unusable
  for dataset generation).

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

## Parking planner (`parking/`)

Independent package (does **not** modify `diffusion_planner/`; only reuses `vae_map.VAE`
via `sys.path`). Every swappable piece is behind an ABC + dataclass (`interfaces.py` /
`config.py`): motion planner, scenario generator, post-processor, map conditioner,
generative planner.

Run everything from the **repo root** with `-m` (NOT from inside `parking/`). Pipeline
order is **dataset -> map_vae -> train -> evaluate**:

```bash
cd /home/t/projects/diffusion_planner_demo
venv_py38/bin/python -m parking.dataset  --n 4000 --proc 8              # HA* teacher -> cache/parking_<M>.npz
venv_py38/bin/python -m parking.map_vae  --npz cache/parking_4000.npz   # retrain+freeze map VAE -> parking/cache/vae_parking.pt
venv_py38/bin/python -m parking.train    --npz cache/parking_4000.npz --steps 12000   # -> parking/cache/diffusion_parking.pt
venv_py38/bin/python -m parking.evaluate --npz cache/parking_4000.npz   # metrics + figs/parking/m6_eval_compare.png
venv_py38/bin/python -m parking.<mod>                                 # each module also has a __main__ self-test
```

| Module | Role |
|--------|------|
| `hybrid_a_star.py` | **teacher**: Hybrid A*, whole search in one `@njit` (array states + hand-written heap + inlined footprint collision). ~0.078 s/pose |
| `scenarios.py` | random perpendicular (reverse-in) + parallel slots with neighbor cars |
| `postprocess.py` | resample dense HA* path -> **exactly N=40** `(x,y,cos,sin)`, endpoints pinned |
| `dataset.py` | multiprocessing build + npz cache + success-rate stats |
| `map_vae.py` | reuse `diffusion_planner/vae_map.py` VAE on parking occupancy -> freeze |
| `conditioner.py` / `diffusion.py` / `train.py` | frozen-VAE latent (+) SDF condition; DDPM over (N,4); AMP + big-batch + precomputed latent; pins endpoints each step |
| `repair.py` / `evaluate.py` | SDF-push + Laplacian-smooth; footprint-collision / curvature / length-ratio metrics (raw vs repaired) |

**npz layout** (`cache/parking_<M>.npz`; `*.npz` is gitignored): `maps` uint8
`(M,2,72,128)` = `[occupancy(0/1), SDF x 255]`; `traj` `(M,40,4)` = **world coords**
`[x,y,cos,sin]`; `start_pose`/`goal_pose` `(M,3)` = `[x,y,theta]`; `gear (M,40) int8`;
`scene_type uint8` (0=perp, 1=par); `length`, `n_switches`, `split` (0=train, 1=eval),
`bbox (4,)`. Normalization is **global via `bbox`**, so `evaluate` MUST use the same npz
the ckpt was trained on.

Rules specific to `parking/`:
- **numba warm-up**: the first `plan()` compiles the JIT (~1.4 s) then caches
  (`cache=True`). `dataset.py` deliberately runs one search in the **main process before**
  spawning the Pool, so workers reuse the cache instead of each recompiling.
- Maps are **72x128** (both multiples of 8) so the reused VAE's 3 stride-2 downsamples work.
- Heading uses `(cos,sin)` (no wrap issues); `gear` is metadata only (not diffused).
- Plots use English labels (same no-CJK-font constraint as the demo).

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
- VAE was upgraded from a 32-d global vector to a **spatial latent (8x8x8) +
  residual blocks + weighted BCE (pos_weight=3) + low KL (beta=1e-2)** - now
  **~444K params** (down from ~5M). U-Net skip connections were deliberately **not**
  used to avoid latent collapse (keeps the latent usable for a future latent-diffusion).

## Parking planner - known state

- **HA* teacher works well**: ~0.078 s/pose, 100% feasible/collision-free (it is the
  data source). Dataset solve rate 0.76-0.80 (failures discarded + resampled).
- **Map VAE on parking maps**: pixel acc 99.80%, obstacle IoU 99.13% (frozen).
- **Diffusion Phase 1 (epsilon-MSE + MLP) is NOT yet kinodynamically feasible**: raw
  footprint-collision-free ~1.7%, feasibility 0%, curvature far above the 1/r_min cap.
  `repair.py` lifts collision-free to ~20-30% but cannot fix feasibility; retraining on
  12k data / 30k steps did **not** recover it -> bottleneck is objective / architecture
  (averaging over sharp, multimodal maneuvers), not data volume.
- **Phase 2 (attempted, mostly negative; see PARKING_NOTES M8)**:
  - *In-training penalty* (`penalty.py`, wired into `train.py` via `--w-nh/--w-curv/--w-coll`, gated by `sqrt(abar)`, default 0): **both weight groups diverge the DDPM sampler** (spirals, length ratio 600-870x, collision 0%) - trajectory geometry cost is structurally incompatible with the epsilon-MSE objective. Checkpoint rolled back.
  - *Sampling guidance* (`evaluate --g-scale/--g-coll/--g-curv/--g-min-abar`, late-step only, coll-only): **a real, safe Pareto win** - raw collision-free 1.7%->25% (repair ~32%), max-kappa drops, length ratio stays stable/no divergence. But adding curvature/heading terms diverges instantly, and it does NOT reach kinematic feasibility -> end-to-end success still 0%.
  - **The blocker is feasibility (reversing geometry), not collision.** Real next steps: MLP -> 1D-Conv/Transformer denoiser; proper classifier-guidance in score/eps space (train a feasibility discriminator), not geometry cost stuffed into epsilon-MSE or x0 gradients.
- **Phase 3 (Denoiser upgrade + metric fix; PARKING_NOTES M9) - real breakthrough**:
  - *MLP -> 1D temporal dilated-conv denoiser* (`CondDenoiserConv`, `config.denoiser=conv`, `train --denoiser conv`, ~419K params; ckpt now stores a `denoiser` field, `evaluate` rebuilds it; default `mlp` keeps backward compat). Same 12k/30k retrain improves every axis: raw collision-free 1.7%->32% (->70% with coll guidance), length ratio 2.34->0.95, mean lateral-slip 0.64->0.15 (expert 0.015).
  - *The old curvature-based feasibility/success metric was BROKEN*: on the fixed N=40 chord-densified rep, gear-switch cusps blow up |dtheta/ds|, so even the 100%-feasible HA* expert scored 0% feasible. Replaced with a cusp/gear-agnostic **nonholonomy lateral-slip** criterion (`evaluate.nonholonomy_slip`, feasibility = max|slip|<=SLIP_TOL=0.2). Under the corrected metric, end-to-end success is nonzero for the first time (conv+repair 6.7%). Raw max_kappa is now diagnostic-only.
  - Remaining: strict curvature feasibility needs per-gear segmentation (model does not output gear); optional small-Transformer denoiser + score-space classifier guidance.
- **Phase 4 (Cusp segmentation + raise N; PARKING_NOTES M10/M11) - metric now trustworthy, model still the bottleneck**:
  - *Cusp segmentation* (`evaluate.segment_by_cusps` splits where motion reverses along heading; `segment_metrics` = per-segment position/Menger curvature + gear-switch count). Hard feasibility gate stays the lateral-slip criterion (only sampling-density-agnostic quantity); segmentation adds diagnostics `raw_seg_kappa` (relative) and **`raw_gear_switches`** (a clean quality signal: MLP ~16 jittery vs CONV ~2.2 ~= expert 2.6).
  - *N is now configurable & auto-derived*: `dataset --n_wp 80` builds a fixed-length-80 npz; `train`/`evaluate` infer N from the data `traj` shape / ckpt `n_wp` field (old N=40 data+ckpts still yield 40, backward compat). ckpt `parking/cache/diffusion_parking_conv_n80.pt`, data `cache/parking_12000_n80.npz`.
  - *Result (honest)*: raising N 40->80 makes **curvature measurable** (expert per-seg kappa 1.45->0.43, within-limit 43%->78%) BUT does NOT fix the model (conv@N80 still reads kappa~4.6 / 0% curvature-feasible; raw collision even dropped 0.32->0.18; end-to-end comparable). => bottleneck is the objective/model, not the representation. NEW LEVER now unblocked: at N=80 curvature is measurable, so proper score-space classifier guidance / differentiable curvature penalty become evaluable (impossible before).
- **Phase 4b (Classifier guidance; PARKING_NOTES M12) - HONEST NEGATIVE, do not pursue post-hoc guidance**:
  - Trained `parking/critic.py` (`FeasibilityCritic`, 1D dilated-conv; positives=HA* expert, negatives=the diffusion model's own samples) and guided sampling by ascending `log p(feasible)` on x_t (`sample(critic=, critic_scale=, critic_min_abar=)`; CLI `evaluate --critic --c-scale --c-min-abar`).
  - The critic separates expert vs model samples PERFECTLY (acc 1.0) - the two are far apart in feature space - yet **guidance still DIVERGES** (even c-scale=0.02, late-only min_abar=0.95: len_ratio 0.95->3~19, collisions->0). Same failure as M8's hand-written geometry cost. Fixing the obvious bug (train critic on clean-only -> must train on noise-augmented x_t via q_sample, and take grad on x_t not x0_hat) did NOT rescue it.
  - **Lesson (consistent across M8/M12)**: for this DDPM, **any inference-time gradient guidance from a separately-trained feasibility signal (geometry cost OR learned classifier) is ill-conditioned** - the gradient pushes samples off-manifold into adversarial high-logit regions. => **feasibility must be baked into the denoiser/objective at training time** (gear-aware / hybrid representation or architecture), not bolted on at sampling. Code kept as tested-then-rejected artifact, OFF by default (c-scale=0 keeps the baseline intact).
- Authoritative docs: `PARKING_NOTES.md` (milestone log) and `diffusion_planner/README.md` section 7.
