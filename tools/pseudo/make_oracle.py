"""
ORACLE pseudo labels (upper-bound experiment only, never a real method): the true positions of the
hidden people, displaced by a known amount of noise. Separates "is the loss design good" from
"is the detector good".

    python -m tools.pseudo.make_oracle -d wildtrack --root ~/Data/Wildtrack \
        --full_ann ~/Data/Wildtrack_full/annotations_positions --kept_ann <drop60>/annotations_positions \
        --noise_r 20 --out pseudo_labels/wildtrack_drop60_oracle_n20

hidden = full minus kept, matched by personID. Noise is uniform in a disk of --noise_r grid cells
(2.5 cm; 20 = 0.5 m), seeded. Same file format as build_pseudo.py (score = 1).
"""
import argparse
import json
import os

import numpy as np

from tools.pseudo.common import (grid_shape_xy, list_frames, load_ann, load_base, n_visible_cams, ped_grid,
                                 ped_in_any_cam, pseudo_fname)


def hidden_people(base, full_ann, kept_ann, frame):
    kept_ids = {p['personID'] for p in load_ann(kept_ann, frame)}
    return [p for p in load_ann(full_ann, frame)
            if p['personID'] not in kept_ids and ped_in_any_cam(p, base.num_cam)]


def main(args):
    base = load_base(args.dataset, args.root)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    sx, sy = grid_shape_xy(base)
    frames = list_frames(args.kept_ann, base, 'train')
    total = 0
    for frame in frames:
        hid = hidden_people(base, args.full_ann, args.kept_ann, frame)
        out = []
        if hid:
            g = np.array([ped_grid(base, p) for p in hid])
            theta = rng.uniform(0, 2 * np.pi, len(g))
            rad = args.noise_r * np.sqrt(rng.uniform(0, 1, len(g)))
            g = g + np.stack([rad * np.cos(theta), rad * np.sin(theta)], 1)
            g[:, 0] = g[:, 0].clip(0, sx - 1)
            g[:, 1] = g[:, 1].clip(0, sy - 1)
            vis = n_visible_cams(base, g)
            out = [{'grid': [round(float(x), 2) for x in gi], 'score': 1.0, 'n_views': int(v), 'n_visible': int(v),
                    'n_dets': int(v), 'max_dets_per_cam': 1} for gi, v in zip(g, vis)]
        total += len(out)
        with open(pseudo_fname(args.out, frame), 'w') as f:
            json.dump(out, f)
    meta = {'args': vars(args), 'oracle': True, 'num_frames': len(frames), 'pseudo': total}
    with open(os.path.join(args.out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'oracle: {total} points over {len(frames)} train frames, noise r = {args.noise_r} cells -> {args.out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--dataset', default='wildtrack', choices=['wildtrack', 'multiviewx'])
    p.add_argument('--root', default='~/Data/Wildtrack')
    p.add_argument('--full_ann', required=True, help='annotations_positions with ALL people')
    p.add_argument('--kept_ann', required=True, help='annotations_positions of the dropped split')
    p.add_argument('--noise_r', type=float, default=0, help='grid cells (2.5 cm)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', required=True)
    main(p.parse_args())
