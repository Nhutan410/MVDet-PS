# Missing-annotation robust loss for MVDet (`--loss mal`)

Implementation of the spec *"Loss chống nhiễu thiếu nhãn (Missing-Annotation Robust Loss) cho MVDet"*
on top of the original MVDet code (fork `Nhutan410/MVDet`, which is left untouched). Everything
below is additive: `--loss mse` (the default) is byte-for-byte the original pipeline.

```
python main.py -d wildtrack --loss mal                       # defaults below
python main.py -d wildtrack --loss mal --mal_warmup_epochs 3 --mal_consensus percentile25 \
       --mal_min_dist_easy_neg 10 --mal_lambda_q 1 --mal_lambda_s 1 --mal_ev_backbone frozen_dinov2
python -m pytest tests/ -q      # or: python tests/test_missing_annotation_loss.py   (CPU, no data)
```

## 1. What was added

| spec item | file | what |
|---|---|---|
| `EvidenceHead` (`s_v`) | `multiview_detector/models/missing_annotation.py` → `EvidenceBranch` | independent backbone (**frozen DINOv2 ViT-S/14** via `torch.hub`, or a **separate ImageNet ResNet-18**, never shared with the main backbone) + `1x1 conv → ReLU → 1x1 conv` head, output logits at the same 270×480 resolution the main per-view features are projected from |
| reuse the homography | `persp_trans_detector.py` → `project_views_to_bev(..., self.proj_mats, ...)` | the **same** `proj_mats` that warp the main features warp `sigmoid(s_v)` to the BEV; `proj_mats_inv` warps BEV masks back into each view (`model.project_bev_to_views`) |
| `aggregate_consensus` | `missing_annotation.py` → `aggregate_consensus(s_bev, visible, method, min_visible_views)` | `median` (default) / `percentile25` / `min` over **visible views only**; `c = 0` when fewer than `min_visible_views` (2) cameras see the cell; returned `.detach()`ed. `mean` is deliberately not offered |
| view visibility | `missing_annotation.py` → `compute_visibility_masks(dataset)` | precomputed `[num_cam, X, Y]` bool buffer: cell centre `(x, y, 0)` → extrinsic → **depth > 0** → intrinsic → pixel inside the image. Wildtrack: mean 3.75 views / cell, 97.8 % of cells have ≥ 2 views |
| `GateHead` (`q`) | `missing_annotation.py` → `GateHead` | 2 convs on the main BEV trunk (output of `map_classifier[:-1]`), **input detached**, last bias init −4 so `q ≈ 0.02` at the switch-on |
| loss | `multiview_detector/loss/missing_annotation_loss.py` → `MissingAnnotationLoss` | returns a dict `{'total','map','q','s','s_pos','s_neg','prior','stats'}` |
| warm-up | `trainer.py` (`mal_warmup_epochs`) | epochs `≤ warmup`: map term = plain `GaussianMSE`, `L_q = L_prior = 0`; `L_s` trains from epoch 1 so `c` is meaningful when `q` switches on |
| logging | `trainer.py` | every `log_interval` batches and per epoch: `L_map, L_q, L_s (pos/neg), q_bg, c_bg, c`; per epoch a 10-bin histogram of `q`; all written to `<logdir>/mal_stats.jsonl`; `map.jpg` gets `q` and `c` rows, `cam1_evidence.jpg` shows `s_1` |
| unit test | `tests/test_missing_annotation_loss.py` | gradient isolation (main ↔ gate ↔ evidence), `c` detached, `q = 0` ⇒ `GaussianMSE`, background minimiser `p* = q`, positives never doubted, consensus rules, confusion-zone exclusion |

## 2. Loss, adapted to MVDet's regression head

MVDet does **not** use sigmoid + focal/BCE: `map_classifier` regresses the BEV map to a Gaussian
soft target with MSE (`GaussianMSE`). The spec's likelihood terms were therefore written in their
MSE counterparts, which keep the two properties that matter: at `q = 0` the term is *exactly* the
original loss, and on background the minimiser is `p* = q` (same as for the BCE form).

```
soft_gt = Gaussian(map_gt)                       # MVDet's own soft target, in [0, 1]
q_eff   = stopgrad(q) · (1 − soft_gt)            # doubt only where the label says background
L_map   = mean[ (1 − q_eff) (p − soft_gt)² + q_eff (p − 1)² ]      # 4.1 + 4.2 blended, no threshold
L_q     = mean[ (q − stopgrad(c))² ]                                # 4.3
L_s     = mean_v [ Σ w_pos·BCE(s_v, 1) / Σ w_pos  +  Σ m_neg·BCE(s_v, 0) / Σ m_neg ]   # 4.4, balanced
L_prior = ( mean_bg(q) − π )²   with mean_bg weighted by (1 − soft_gt)                  # 4.5, optional
L_total = L_map + λ_q L_q + λ_s L_s + λ_prior L_prior  (+ MVDet's unchanged per-view head/foot loss)
```

* **`w_pos`** for view `v` = MVDet's own Gaussian **foot** target of the kept annotations in that
  view (foot pixels are what the homography maps to the correct ground cell, so `s_v` is trained as
  a foot detector, exactly like MVDet's per-view foot head).
* **`m_neg`** = pixels whose ground point is ≥ `min_dist_easy_neg` reduced-grid cells (Euclidean;
  Wildtrack 1 cell = 10 cm, default 10 cells = 1 m) from *every* kept annotation — computed on the
  BEV (`far_mask`) and warped into the view with the inverse homography — and not inside a foot
  window. Everything in between is the **confusion zone** and is ignored by `L_s`. Pixels that do
  not map into the grid at all (sky, walls above the horizon) are ignored too: they never reach `c`.
* Means (not sums) over pixels keep `L_map` on the same scale as the baseline.

### Where the gradients go (constraint 2 of the spec)

```
image ──► main backbone ──► BEV trunk ──► p  ◄── L_map (+ per-view loss)      main SGD optimizer
                                 │ detach
                                 └──────► GateHead ──► q  ◄── L_q (+ L_prior)  aux Adam optimizer
image ──► evidence backbone ──► s_v ──► warp ──► consensus ──► c (detached)
                                  ▲
                                  └── L_s (kept feet = 1, far pixels = 0)        aux Adam optimizer
```

* `c` is detached; the gate's trunk input is detached; `q` is detached inside `L_map`. So the main
  branch is shaped only by `L_map` (and MVDet's per-view loss), the gate only by `L_q`, the
  evidence only by `L_s`. `tests/test_gradient_isolation` asserts this.
* `--mal_conf_grad_q` (ablation only) lets `L_map` pull on `q` (the "model judges itself" loop the
  spec forbids). It still never reaches the evidence branch.
* The evidence head and the gate head are trained by a **separate Adam optimizer**
  (`--mal_aux_lr`, default 1e-3); the main SGD + OneCycle schedule excludes them. The DINOv2
  backbone is frozen and kept in `eval()` even inside `model.train()`.

## 3. Config (`main.py` flags ↔ spec §7)

| spec | flag | default |
|---|---|---|
| `enabled` | `--loss mal` | off (`mse`) |
| `warmup_epochs` | `--mal_warmup_epochs` | 3 |
| `min_dist_easy_neg` | `--mal_min_dist_easy_neg` | 10 (reduced cells) |
| `consensus_agg` | `--mal_consensus` | `median` (torch lower median: with 2 views it equals `min`) |
| `min_visible_views` | `--mal_min_visible_views` | 2 |
| `lambda_q`, `lambda_s` | `--mal_lambda_q`, `--mal_lambda_s` | 1.0, 1.0 |
| `lambda_prior`, `pi_prior` | `--mal_lambda_prior`, `--mal_pi_prior` | 0.0, 0.0 (off) |
| `s_v_backbone` | `--mal_ev_backbone` | `frozen_dinov2` (`separate_trainable` = own ResNet-18) |
| — | `--mal_dino_name`, `--mal_dino_input H W` | `dinov2_vits14`, `504 896` (multiples of 14) |
| — | `--mal_aux_lr` | 1e-3 |
| — | `--mal_ev_device` | `auto` (= GPU of `base_pt1`) |
| — | `--mal_conf_grad_q` | off |

Every term can be switched off independently for ablations: `--mal_lambda_q 0` leaves `q` at its
init (≈ 0.02 ⇒ practically the baseline), `--mal_lambda_s 0` stops `s_v` from learning,
`--mal_lambda_prior 0` disables the prior. The log directory encodes the config:
`logs/wildtrack_frame/mal_<backbone>_<consensus>_w<warmup>_d<dist>_lq<λq>_ls<λs>[...]/default/<time>/`.

## 4. What to watch during training (spec §6.4)

`mal_stats.jsonl` (one JSON per epoch) and the console carry:

* `q_bg_mean` — mean gate on the background region. Should stay ≈ 0 during warm-up and then grow
  towards `c_bg_mean`, **not** race to 1 everywhere. A fast rise to 1 right after warm-up means the
  gradient cut between `s_v` and `p` is broken (or `--mal_conf_grad_q` is on).
* `c_bg_mean`, `c_mean` — multi-view evidence; with 60 % of labels dropped and a low-capacity head,
  expect values around the fraction of *labelled* feet rather than 1 at every real person: the
  "easy negatives" unavoidably contain the feet of dropped people, and the head can only learn
  `P(labelled | foot-like)`. The consensus still separates real people from background.
* `q_hist` — 10-bin histogram of `q` over BEV cells.
* `L_s pos / neg` — should both decrease; `pos` stalling near `log 2` means the evidence head is not
  seeing the kept feet (check `cam1_evidence.jpg`).

## 5. Deviations from the spec, and why

1. **MSE form instead of log-likelihoods** (§2 above) — MVDet's map head is a regression head; the
   sigmoid/BCE form would have changed the main branch, which the spec forbids.
2. **`q` is detached inside `L_conf` by default** — the spec writes `q` into `L_conf` and adds `L_q`
   "otherwise `q` learns nothing", which only holds if `L_conf` does not back-propagate into `q`;
   letting it would re-open the self-judging loop of constraint 2. The ablation flag keeps the
   literal reading available.
3. **Evidence + gate heads on their own Adam optimizer** — the spec asks for at least a separate
   optimizer for `s_v`; the gate is added there because with the main SGD (gradient averaged over
   43 k cells) `q` moved far too slowly to track `c`.
4. **Per-view head/foot loss of MVDet is unchanged** — the spec covers only the BEV map.
5. **Chebyshev → Euclidean**: the "far" test uses a Euclidean disk of radius `min_dist_easy_neg`.

## 6. Device layout (1 or 2 GPUs)

`PerspTransDetector` decides at construction time from `torch.cuda.device_count()`; do **not** set
`CUDA_VISIBLE_DEVICES=0` if you want both Kaggle T4s.

| | 2 GPUs (Kaggle T4 x2) | 1 GPU | no CUDA |
|---|---|---|---|
| `base_pt1` (ResNet layers 1-3, high-res activations) | `cuda:1` | `cuda:0` | `cpu` |
| **evidence branch `s_v`** (DINOv2 / own ResNet + head) | `cuda:1` (`--mal_ev_device auto`) | `cuda:0` | `cpu` |
| `base_pt2`, `img_classifier`, warp, `map_classifier`, gate `q`, consensus `c` | `cuda:0` | `cuda:0` | `cpu` |

The evidence branch shares nothing with the main branch, so on 2 GPUs its forward/backward on
`cuda:1` overlaps with the main branch on `cuda:0`; `s_logits` stay on `cuda:1` (L_s is computed
there and moved next to `L_map` before summing), `c` is moved to `cuda:0` next to `q`. The epoch
summary prints `GPU peak memory this epoch: cuda:0 … cuda:1 …`. `--mal_ev_device cuda:N` overrides
the placement. The cross-device path was verified locally with the main branch on CPU and the
evidence branch on Apple MPS (identical losses to the single-device run).

## 7. Local CPU smoke test

`PerspTransDetector` falls back to CPU when no CUDA device exists, so the whole `main.py` path can
be exercised on a laptop (2-frame dataset, ~20 s / iteration on an M4 Pro). Real training is done on
Kaggle / Colab — see `notebooks/mvdet/mvdet_wildtrack_drop60_mal_kaggle.ipynb` in the capstone repo.
