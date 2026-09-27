"""
Stage 3: quality of the pseudo labels against the HIDDEN people (evaluation only -- the hidden labels
are never used for training, and nothing here feeds back into the training code).

    python -m tools.pseudo.eval_pseudo -d wildtrack --root ~/Data/Wildtrack \
        --full_ann <Wildtrack>/annotations_positions --kept_ann <drop60>/annotations_positions \
        --pseudo pseudo_labels/wildtrack_drop60_frcnn_v2_s0.5 --viz_dir pseudo_viz

Reports precision / recall at 0.5 m (the MODA threshold) and 1 m (Hungarian matching), localisation
error of the matched points, and precision broken down by n_views / score so --ps_min_views and
--ps_min_score can be chosen. Writes <pseudo>/eval.json and optional BEV overlays.
"""
import argparse
import json
import os

import numpy as np

from tools.pseudo.common import CELL_CM, grid_shape_xy, load_ann, load_base, match_points, ped_grid, pseudo_fname
from tools.pseudo.make_oracle import hidden_people


def pr(n_match, n_pred, n_gt):
    return {'precision': n_match / max(n_pred, 1), 'recall': n_match / max(n_gt, 1),
            'matched': n_match, 'pred': n_pred, 'gt': n_gt}


def viz_frame(base, frame, kept, hidden, pseudo, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sx, sy = grid_shape_xy(base)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, sy)
    ax.set_ylim(sx, 0)
    ax.set_aspect('equal')
    ax.set_xlabel('gy')
    ax.set_ylabel('gx')
    if len(kept):
        ax.scatter(kept[:, 1], kept[:, 0], c='tab:green', s=40, label=f'kept GT ({len(kept)})')
    if len(hidden):
        ax.scatter(hidden[:, 1], hidden[:, 0], c='tab:red', marker='x', s=50, label=f'hidden GT ({len(hidden)})')
    if pseudo:
        g = np.array([p['grid'] for p in pseudo])
        nv = np.array([p['n_views'] for p in pseudo])
        ax.scatter(g[:, 1], g[:, 0], s=30 + 30 * nv, facecolors='none', edgecolors='tab:blue',
                   label=f'pseudo ({len(pseudo)}, size ~ n_views)')
        for gi in g:  # 0.5 m match radius
            ax.add_patch(plt.Circle((gi[1], gi[0]), 20, color='tab:blue', fill=False, lw=0.3))
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(f'frame {frame}')
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(args):
    base = load_base(args.dataset, args.root)
    frames = sorted(int(f.split('.')[0]) for f in os.listdir(args.pseudo) if f[0].isdigit())
    thrs = {'0.5m': 0.5 * 100 / CELL_CM, '1m': 1.0 * 100 / CELL_CM}
    counts = {k: [0, 0, 0] for k in thrs}
    err = []
    by_views, by_score = {}, {}
    mv_counts = {k: [0, 0, 0] for k in (1, 2, 3)}  # P/R@0.5m if training kept only points with n_views >= k
    if args.viz_dir:
        os.makedirs(args.viz_dir, exist_ok=True)
    viz_frames = set(frames[::max(1, len(frames) // args.viz_n)][:args.viz_n]) if args.viz_dir else set()
    for frame in frames:
        with open(pseudo_fname(args.pseudo, frame)) as f:
            pseudo = json.load(f)
        hidden = np.array([ped_grid(base, p) for p in hidden_people(base, args.full_ann, args.kept_ann, frame)])
        hidden = hidden.reshape(-1, 2)
        g = np.array([p['grid'] for p in pseudo]).reshape(-1, 2)
        for k, thr in thrs.items():
            m = match_points(g, hidden, thr)
            counts[k][0] += len(m)
            counts[k][1] += len(g)
            counts[k][2] += len(hidden)
            if k == '0.5m':
                err += [d for _, _, d in m]
                hit = np.zeros(len(g), bool)
                hit[[i for i, _, _ in m]] = True
                for p, h in zip(pseudo, hit):
                    kv = min(p['n_views'], 4)
                    ks = f'{min(int(p["score"] * 10), 9) / 10:.1f}'
                    by_views.setdefault(kv, [0, 0])
                    by_score.setdefault(ks, [0, 0])
                    by_views[kv][0] += int(h)
                    by_views[kv][1] += 1
                    by_score[ks][0] += int(h)
                    by_score[ks][1] += 1
        for k in mv_counts:
            gk = np.array([p['grid'] for p in pseudo if p['n_views'] >= k]).reshape(-1, 2)
            mv_counts[k][0] += len(match_points(gk, hidden, thrs['0.5m']))
            mv_counts[k][1] += len(gk)
            mv_counts[k][2] += len(hidden)
        if frame in viz_frames:
            kept = np.array([ped_grid(base, p) for p in load_ann(args.kept_ann, frame)]).reshape(-1, 2)
            viz_frame(base, frame, kept, hidden, pseudo, os.path.join(args.viz_dir, f'bev_{frame:08d}.png'))

    err = np.array(err)
    rep = {'pseudo_dir': args.pseudo, 'num_frames': len(frames),
           **{f'@{k}': pr(*v) for k, v in counts.items()},
           'loc_error_matched_cm': ({f'p{q}': float(np.percentile(err, q) * CELL_CM) for q in (50, 80, 90)}
                                    if len(err) else {}),
           'min_views@0.5m': {str(k): pr(*v) for k, v in mv_counts.items()},
           'precision_by_n_views@0.5m': {('4+' if k == 4 else str(k)): {'precision': v[0] / v[1], 'n': v[1]}
                                         for k, v in sorted(by_views.items())},
           'precision_by_score@0.5m': {k: {'precision': v[0] / v[1], 'n': v[1]} for k, v in sorted(by_score.items())}}
    with open(os.path.join(args.pseudo, 'eval.json'), 'w') as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--dataset', default='wildtrack', choices=['wildtrack', 'multiviewx'])
    p.add_argument('--root', default='~/Data/Wildtrack')
    p.add_argument('--full_ann', required=True)
    p.add_argument('--kept_ann', required=True)
    p.add_argument('--pseudo', required=True)
    p.add_argument('--viz_dir', default=None)
    p.add_argument('--viz_n', type=int, default=4)
    main(p.parse_args())
