"""
Stage 2: 2D boxes -> foot points on the ground plane -> multi-view merge -> pseudo BEV points.

    python -m tools.pseudo.build_pseudo -d wildtrack --root ~/Data/Wildtrack \
        --raw pseudo_raw/wildtrack_frcnn_v2.json \
        --kept_ann <Wildtrack_dropped>/drop60/annotations_positions \
        --out pseudo_labels/wildtrack_drop60_frcnn_v2_s0.5

Per train frame:
  1. keep boxes with score >= --score_thr, not touching the image bottom (feet cut off), tall enough;
  2. foot point u = (x1+x2)/2, v = y2 -> grid [gx, gy] (dataset helpers), drop points outside the grid;
  3. merge the points of all views (average linkage, --merge_dist cells): one cluster ~ one person,
     n_views = number of distinct cameras in the cluster, n_visible = cameras that can see the centre;
  4. drop clusters closer than --d_gt cells to a KEPT label (that person is already annotated).

Only the kept (dropped-split) annotations are read. The hidden people are never touched here -- use
eval_pseudo.py for that. Clusters that land on kept labels give a leak-free estimate of the projection
error (guide sec. 2.4); its percentiles and a suggested jitter radius r are written to meta.json.

Output: <out>/<frame:08d>.json = [{"grid": [gx, gy], "score", "n_views", "n_visible", "n_dets",
"max_dets_per_cam"}, ...] + <out>/meta.json.
"""
import argparse
import json
import math
import os

import numpy as np

from tools.pseudo.common import (CELL_CM, image_to_grid, in_grid, load_ann, load_base, match_points,
                                 n_visible_cams, ped_grid, pseudo_fname)


def merge_multiview(points, cams, scores, dist_thr):
    """average-linkage clustering of foot points (grid cells); returns list of clusters"""
    if len(points) == 0:
        return []
    if len(points) == 1 or dist_thr <= 0:
        labels = np.arange(len(points))
    else:
        from scipy.cluster.hierarchy import fcluster, linkage
        labels = fcluster(linkage(points, 'average'), dist_thr, 'distance')
    out = []
    for c in np.unique(labels):
        m = labels == c
        w = scores[m] / scores[m].sum()
        _, per_cam = np.unique(cams[m], return_counts=True)
        out.append({'grid': (points[m] * w[:, None]).sum(0), 'score': float(scores[m].mean()),
                    'n_views': int(len(per_cam)), 'n_dets': int(m.sum()), 'max_dets_per_cam': int(per_cam.max())})
    return out


def pct(values, qs=(50, 80, 90)):
    if len(values) == 0:
        return {}
    return {**{f'p{q}': float(np.percentile(values, q)) for q in qs},
            'mean': float(np.mean(values)), 'n': int(len(values))}


def main(args):
    base = load_base(args.dataset, args.root)
    with open(args.raw) as f:
        raw = json.load(f)
    H = base.img_shape[0]
    os.makedirs(args.out, exist_ok=True)

    frames = sorted(int(k) for k in raw['dets'])
    stats = {'boxes_in': 0, 'boxes_kept': 0, 'off_grid': 0, 'clusters': 0, 'near_gt_removed': 0, 'pseudo': 0}
    calib_err = []  # cluster -> kept-label distance (grid cells), leak-free projection error estimate
    per_frame = []
    for frame in frames:
        d = np.array(raw['dets'][str(frame)], dtype=np.float64).reshape(-1, 6)
        stats['boxes_in'] += len(d)
        keep = (d[:, 5] >= args.score_thr) & (d[:, 4] < H - args.bottom_margin) & \
               (d[:, 4] - d[:, 2] >= args.min_height)
        d = d[keep]
        stats['boxes_kept'] += len(d)
        cams = d[:, 0].astype(int)
        grids = np.zeros([len(d), 2])
        for cam in np.unique(cams):
            m = cams == cam
            grids[m] = image_to_grid(base, cam, np.stack([(d[m, 1] + d[m, 3]) / 2, d[m, 4]], 1))
        ok = in_grid(base, grids)
        stats['off_grid'] += int((~ok).sum())
        clusters = merge_multiview(grids[ok], cams[ok], d[ok, 5], args.merge_dist)
        stats['clusters'] += len(clusters)

        kept = np.array([ped_grid(base, p) for p in load_ann(args.kept_ann, frame)]).reshape(-1, 2)
        centers = np.array([c['grid'] for c in clusters]).reshape(-1, 2)
        calib_err += [dist for _, _, dist in match_points(centers, kept, args.calib_thr)]
        if len(kept) and len(centers):
            near = np.linalg.norm(centers[:, None] - kept[None], axis=-1).min(1) <= args.d_gt
        else:
            near = np.zeros(len(centers), bool)
        stats['near_gt_removed'] += int(near.sum())
        clusters = [c for c, n in zip(clusters, near) if not n]
        if clusters:
            vis = n_visible_cams(base, np.array([c['grid'] for c in clusters]))
            for c, v in zip(clusters, vis):
                c['n_visible'] = int(v)
                c['grid'] = [round(float(x), 2) for x in c['grid']]
        stats['pseudo'] += len(clusters)
        per_frame.append(len(clusters))
        with open(pseudo_fname(args.out, frame), 'w') as f:
            json.dump(clusters, f)

    err_cells = np.array(calib_err)
    calib = {'cells': pct(err_cells), 'cm': pct(err_cells * CELL_CM)}
    reduce = args.grid_reduce
    suggested_r = math.ceil(np.percentile(err_cells, 80) / reduce) if len(err_cells) else None
    meta = {'args': vars(args), 'detector_meta': raw.get('meta'), 'num_frames': len(frames), 'stats': stats,
            'pseudo_per_frame_mean': float(np.mean(per_frame)) if per_frame else 0.0,
            'projection_error_on_kept_labels': calib,
            f'suggested_r_output_cells_p80 (grid_reduce={reduce})': suggested_r}
    with open(os.path.join(args.out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(json.dumps({k: v for k, v in meta.items() if k != 'args'}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--dataset', default='wildtrack', choices=['wildtrack', 'multiviewx'])
    p.add_argument('--root', default='~/Data/Wildtrack')
    p.add_argument('--raw', required=True, help='output of detect_2d.py')
    p.add_argument('--kept_ann', required=True, help='annotations_positions of the partial (dropped) split')
    p.add_argument('--out', required=True)
    p.add_argument('--score_thr', type=float, default=0.5)
    p.add_argument('--bottom_margin', type=float, default=2, help='boxes with y2 >= H - margin have cut-off feet')
    p.add_argument('--min_height', type=float, default=50, help='px on the original image')
    p.add_argument('--merge_dist', type=float, default=20, help='grid cells (2.5 cm): 20 = 0.5 m')
    p.add_argument('--d_gt', type=float, default=20, help='grid cells: drop pseudo points this close to a kept label')
    p.add_argument('--calib_thr', type=float, default=40,
                   help='grid cells: max cluster <-> kept-label distance counted in the projection-error stats')
    p.add_argument('--grid_reduce', type=int, default=4, help='only used to express the suggested r in output cells')
    main(p.parse_args())
