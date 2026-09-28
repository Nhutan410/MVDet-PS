"""
Evaluate trained MVDet checkpoints on the test set at several cls_thres values (one forward pass per checkpoint,
the BEV maps are cached and re-thresholded -- same threshold -> NMS -> evaluate path as trainer.test()).

    python tools/eval_thresholds.py -d wildtrack --ckpt <logdir>/MultiviewDetector.pth [more ckpts] \
        --thresholds 0.3 0.4 0.5 0.6 0.7 --out sweep.json

Note: choosing cls_thres on the test set is test-set tuning -- report it as such and apply the same threshold
to the baseline for a fair comparison.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torchvision.transforms as T

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multiview_detector.datasets import MultiviewX, Wildtrack, frameDataset
from multiview_detector.evaluation.evaluate import evaluate
from multiview_detector.models.persp_trans_detector import PerspTransDetector
from multiview_detector.utils.nms import nms


def main(args):
    base = Wildtrack(os.path.expanduser(args.root)) if args.dataset == 'wildtrack' else \
        MultiviewX(os.path.expanduser(args.root))
    trans = T.Compose([T.Resize([720, 1280]), T.ToTensor(),
                       T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    train_set = frameDataset(base, train=True, transform=trans, grid_reduce=4)
    test_set = frameDataset(base, train=False, transform=trans, grid_reduce=4)
    loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=args.num_workers)
    os.makedirs(args.work_dir, exist_ok=True)
    res_fpath = os.path.join(args.work_dir, 'test_sweep.txt')
    out = {}
    for ckpt in args.ckpt:
        name = args.names[args.ckpt.index(ckpt)] if args.names else ckpt
        model = PerspTransDetector(train_set, 'resnet18')
        model.load_state_dict(torch.load(ckpt, map_location='cpu'))
        model.eval()
        maps = []
        for data, _, _, frame in loader:
            with torch.no_grad():
                maps.append((int(frame[0]), model(data)[0].detach().cpu().squeeze()))
        out[name] = {}
        print(f'\n{name}\n{"cls_thres":>10} | {"MODA":>6} {"MODP":>6} {"Prec":>6} {"Rec":>6} | #det')
        for t in args.thresholds:
            res_list = []
            for frame, m in maps:
                grid_ij = (m > t).nonzero()
                v = m[m > t]
                if len(v) == 0:
                    continue
                grid_xy = grid_ij[:, [1, 0]] if base.indexing == 'xy' else grid_ij
                pos = grid_xy.float() * test_set.grid_reduce
                ids, count = nms(pos, v, 20, np.inf)
                res_list.append(torch.cat([torch.ones([count, 1]) * frame, pos[ids[:count]]], dim=1))
            res = torch.cat(res_list, 0).numpy() if res_list else np.empty([0, 3])
            np.savetxt(res_fpath, res, '%d')
            if len(res):
                recall, precision, moda, modp = evaluate(os.path.abspath(res_fpath), os.path.abspath(test_set.gt_fpath),
                                                         base.__name__)
            else:
                recall = precision = moda = modp = 0.0
            out[name][f'{t:g}'] = {'moda': float(moda), 'modp': float(modp), 'precision': float(precision),
                                   'recall': float(recall), 'n_det': int(len(res))}
            print(f'{t:10.2f} | {moda:6.1f} {modp:6.1f} {precision:6.1f} {recall:6.1f} | {len(res)}'
                  + ('   <- training default' if abs(t - 0.4) < 1e-9 else ''), flush=True)
        del model
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(out, f, indent=2)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--dataset', default='wildtrack', choices=['wildtrack', 'multiviewx'])
    p.add_argument('--root', default='~/Data/Wildtrack')
    p.add_argument('--ckpt', nargs='+', required=True)
    p.add_argument('--names', nargs='*', default=None)
    p.add_argument('--thresholds', type=float, nargs='+', default=[0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7])
    p.add_argument('--work_dir', default='sweep_tmp')
    p.add_argument('--out', default=None)
    p.add_argument('-j', '--num_workers', type=int, default=2)
    main(p.parse_args())
