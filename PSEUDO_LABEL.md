# MVDet with pseudo labels from a 2D detector (`--loss pseudo`, branch `pseudo-label`)

This branch trains MVDet when only some pedestrians have a BEV label, for example the Wildtrack drop60 split.
The unlabeled people get **pseudo labels**: an off-the-shelf COCO detector finds them in each view, and their
foot points are projected onto the ground plane.

The idea follows `docs/mvdet_pseudo_label_guide.md` in the capstone repo. The error of a pseudo label is in its
**location**, not in its value.

## Pipeline

```
tools/pseudo/detect_2d.py     images -> person boxes (Faster R-CNN v2, COCO), score >= 0.05, raw json
tools/pseudo/build_pseudo.py  score / bottom-edge / height filter -> foot (u=(x1+x2)/2, v=y2) -> Z=0 -> grid
                              -> average-linkage merge over views (0.5 m) -> drop clusters within 0.5 m of a KEPT label
                              -> <out>/<frame>.json [{grid, score, n_views, n_visible, n_dets, max_dets_per_cam}]
                              -> meta.json with a leak-free projection-error estimate and a suggested r
tools/pseudo/eval_pseudo.py   P/R vs the HIDDEN people (full minus kept, by personID) at 0.5 m / 1 m,
                              breakdown by n_views / score, BEV overlays. Evaluation only.
tools/pseudo/make_oracle.py   hidden positions + known noise (upper-bound experiment, not a method)
```

- Projection reuses `multiview_detector.utils.projection` (the same `K[r1 r2 t]` homography as MVDet's warp) and
  the dataset's `get_worldgrid_from_worldcoord`. Wildtrack images are already undistorted (`intrinsic_zero`).
- Pseudo points are only generated for, and loaded on, the **train** split. The hidden labels are only read by
  `eval_pseudo.py` and `make_oracle.py`, never by the training code.
- Sanity check with the GT boxes as the "detector" (Wildtrack drop60, 360 train frames):
  - P@0.5m = 1.00 and R = 0.97 on the 5148 hidden people.
  - The projection error measured on the kept labels (p80 14.6 cm) matches the error on the hidden people
    (p80 14.1 cm), so the leak-free estimate of `r` is valid.
- Real Faster R-CNN v2, local run on 2 frames only:
  - P@0.5m = 0.39 and R = 0.87 over all pseudo points.
  - Single-view clusters have precision 0.17. Keeping `n_views >= 2` gives P = 0.78 and R = 0.62,
    hence `--ps_min_views 2`.

## Loss (`multiview_detector/loss/pseudo_gaussian_mse.py`)

Kept labels are handled exactly as in `GaussianMSE`. Each pseudo point has an output-map position and a
confidence alpha, and gets a positive target near that position:

| `--ps_variant` | guide | target |
|---|---|---|
| `gauss` | B (r=0: E2 basic form) | GT-kernel Gaussian at a random point of the r-disk, pixel weight lambda * alpha |
| `point` | A | one pixel at a random point of the disk pushed to 1: alpha (P - 1)^2 |
| `maxval` | C | alpha (max_disk P - 1)^2 |
| `mil` | C with the GT shape | Gaussian at the argmax of P inside the disk (argmax detached) |

Design decisions:

- **Normalisation.** Every term is summed and divided by H * W, like the mean of `GaussianMSE`. A point target
  carries the weight `sum(kernel^2)`, the squared mass of one GT Gaussian. So at P = 0, one pseudo person weighs
  exactly as much as one kept person, in every variant. The guide's `mean over N points` form is about
  1000 x larger per pixel than the GT term on the 120 x 360 map and would dominate it.
- **Consistency with GaussianMSE.** With `gauss`, r = 0, lambda * alpha = 1 and r_ignore = 0, the loss is exactly
  GaussianMSE with the pseudo points added to `map_gt` (unit-tested).
- **Background.** There is no separate L_neg: the background is part of the same MSE. Background pixels within
  `r_ignore` (default 1.5 r) of a pseudo point and not on a kept label get weight 0. The pseudo target pixels are
  never counted as background, even while lambda = 0.
- **Jitter** is applied only in train mode; the trainer switches the criterion to eval mode for testing.
- **Lambda schedule.** `lambda_ps(epoch) = ps_lambda * min(1, (epoch - warmup) / ramp)` is 0 during the warm-up
  and scales only the pseudo term.
- **alpha** (set in `frameDataset.load_pseudo`):
  - `const`: `--ps_alpha_const`.
  - `score`: the detector score.
  - `views`: `score * min(1, n_views / min(k, n_visible))`. Agreement is counted relative to the cameras that
    can see that spot at all.
- **Per-view head/foot loss** is still the original loss on the kept labels only. Hidden people therefore remain
  negatives in the image-level loss. Possible follow-up: use the detector boxes there.

Distances in the flags are in **output cells** (`grid_reduce 4`, 1 cell = 10 cm). The json files use
full-resolution grid cells (2.5 cm).

## Usage

```bash
python -m tools.pseudo.detect_2d -d wildtrack --root ~/Data/Wildtrack --out pseudo/raw_frcnn_v2.json
python -m tools.pseudo.build_pseudo -d wildtrack --root ~/Data/Wildtrack --raw pseudo/raw_frcnn_v2.json \
    --kept_ann <drop60>/annotations_positions --out pseudo/wt_drop60_frcnn_v2_s0.5 --score_thr 0.5
python -m tools.pseudo.eval_pseudo -d wildtrack --root ~/Data/Wildtrack --full_ann <full>/annotations_positions \
    --kept_ann <drop60>/annotations_positions --pseudo pseudo/wt_drop60_frcnn_v2_s0.5 --viz_dir pseudo/viz
# ~/Data/Wildtrack/annotations_positions must point at the drop60 annotations for training
python main.py -d wildtrack --loss pseudo --pseudo_dir pseudo/wt_drop60_frcnn_v2_s0.5 \
    --ps_variant gauss --ps_r 0 --ps_r_ignore 0 --ps_min_views 2          # E2
python tests/test_pseudo_loss.py
```

Per-epoch stats go to `<logdir>/pseudo_stats.jsonl`: `l_gt`, `l_ps`, `l_view`, `p_at_ps`, `ignored_frac`,
`lam_ps`, and train precision / recall.

The Kaggle notebook is `notebooks/mvdet/mvdet_wildtrack_drop60_pseudo_kaggle.ipynb` in the capstone repo. It runs
detection, the quality table, the oracle, and the E2 / E3 / ORACLE runs.
