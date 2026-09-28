"""
Self-training step: a trained MVDet (the "teacher", e.g. the best pseudo-label run) re-scores the pseudo points
of the TRAIN frames and assigns each one a tier for the next training round.

    python -m tools.pseudo.teacher_rescore -d wildtrack --root ~/Data/Wildtrack --ckpt <run>/MultiviewDetector.pth \
        --pseudo pseudo/wt_drop60_frcnn_v2_s0.5 --out pseudo/wt_drop60_frcnn_v2_s0.5_teacher \
        --pos_thr 0.4 --ign_thr 0.1

teacher = max of the teacher's BEV map within --radius output cells of the point (fused over all cameras, so a
projection ghost seen by one camera tends to score low). Tier: "pos" if teacher >= pos_thr, "ign" if
>= ign_thr, else "drop" (background). Every input field is kept and "teacher" / "tier" are added; frameDataset
honours "tier" over its own view filters. --add_peaks {none,ign,pos} also adds teacher peaks (>= pos_thr) that
are farther than --d_gt from every kept label and every pseudo point.

Caveat: the teacher was trained on these very frames with the same pseudo labels, so it may have memorised
some ghosts. Check the result with eval_pseudo.py (precision of the "pos" tier) before training on it.
"""
import argparse
import json
import os
import shutil

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from multiview_detector.datasets import frameDataset
from multiview_detector.models.persp_trans_detector import PerspTransDetector
from multiview_detector.utils.nms import nms
from tools.pseudo.common import load_ann, load_base, pseudo_fname


def main(args):
    base = load_base(args.dataset, args.root)
    normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    trans = T.Compose([T.Resize([720, 1280]), T.ToTensor(), normalize])
    ds = frameDataset(base, train=True, transform=trans, grid_reduce=4)
    model = PerspTransDetector(ds, 'resnet18')
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
    model.eval()
    red = ds.grid_reduce
    frames = sorted(int(f.split('.')[0]) for f in os.listdir(args.pseudo) if f[0].isdigit())
    frame_to_idx = {f: i for i, f in enumerate(ds.map_gt.keys())}
    os.makedirs(args.out, exist_ok=True)
    counts = {'pos': 0, 'ign': 0, 'drop': 0, 'peaks_added': 0}
    R = int(args.radius)

    def to_rc(g):
        return (g[1] / red, g[0] / red) if base.indexing == 'xy' else (g[0] / red, g[1] / red)

    def to_grid(r, c):
        return [c * red, r * red] if base.indexing == 'xy' else [r * red, c * red]

    for n, frame in enumerate(frames):
        with open(pseudo_fname(args.pseudo, frame)) as f:
            pts = json.load(f)
        imgs = ds[frame_to_idx[frame]][0][None]
        with torch.no_grad():
            m = model(imgs)[0].detach().cpu()[0, 0]  # (H, W)
        H, W = m.shape
        pooled = F.max_pool2d(m[None, None], 2 * R + 1, stride=1, padding=R)[0, 0]
        for p in pts:
            r, c = to_rc(p['grid'])
            t = float(pooled[min(max(int(r), 0), H - 1), min(max(int(c), 0), W - 1)])
            p['teacher'] = round(t, 4)
            p['tier'] = 'pos' if t >= args.pos_thr else ('ign' if t >= args.ign_thr else 'drop')
            counts[p['tier']] += 1
        if args.add_peaks != 'none':
            ij = (m > args.pos_thr).nonzero().float()
            if len(ij):
                ids, cnt = nms(ij, m[m > args.pos_thr], 20 / red, np.inf)
                peaks = ij[ids[:cnt]].numpy()
                known = np.array([np.array(to_rc(p['grid'])) for p in pts] +
                                 [np.array(to_rc(base.get_worldgrid_from_pos(q['positionID'])))
                                  for q in load_ann(args.kept_ann, frame)]).reshape(-1, 2)
                for r, c in peaks:
                    if len(known) and np.linalg.norm(known - [r, c], axis=1).min() * red <= args.d_gt:
                        continue
                    pts.append({'grid': [round(float(v), 2) for v in to_grid(r, c)], 'score': float(m[int(r), int(c)]),
                                'n_views': 0, 'n_visible': 0, 'n_dets': 0, 'max_dets_per_cam': 0,
                                'teacher': round(float(m[int(r), int(c)]), 4), 'tier': args.add_peaks,
                                'source': 'teacher_peak'})
                    counts['peaks_added'] += 1
        with open(pseudo_fname(args.out, frame), 'w') as f:
            json.dump(pts, f)
        if n % 50 == 0:
            print(f'  {n + 1}/{len(frames)} frames, {counts}', flush=True)

    src_meta = os.path.join(args.pseudo, 'meta.json')
    if os.path.isfile(src_meta):
        shutil.copyfile(src_meta, os.path.join(args.out, 'meta_source.json'))
    with open(os.path.join(args.out, 'meta.json'), 'w') as f:
        json.dump({'args': vars(args), 'counts': counts, 'num_frames': len(frames)}, f, indent=2)
    print('done:', counts)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--dataset', default='wildtrack', choices=['wildtrack', 'multiviewx'])
    p.add_argument('--root', default='~/Data/Wildtrack')
    p.add_argument('--ckpt', required=True, help='MultiviewDetector.pth of the teacher run')
    p.add_argument('--pseudo', required=True, help='pseudo dir to re-score (build_pseudo output, no view filter)')
    p.add_argument('--kept_ann', default=None, help='kept (dropped-split) annotations; needed for --add_peaks')
    p.add_argument('--out', required=True)
    p.add_argument('--radius', type=float, default=4, help='output cells searched around each point')
    p.add_argument('--pos_thr', type=float, default=0.4)
    p.add_argument('--ign_thr', type=float, default=0.1)
    p.add_argument('--add_peaks', default='none', choices=['none', 'ign', 'pos'])
    p.add_argument('--d_gt', type=float, default=20, help='grid cells (2.5 cm) for --add_peaks')
    a = p.parse_args()
    if a.add_peaks != 'none' and a.kept_ann is None:
        p.error('--add_peaks needs --kept_ann')
    main(a)
