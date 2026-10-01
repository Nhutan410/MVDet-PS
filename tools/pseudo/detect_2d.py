"""
Stage 1 of the pseudo-label pipeline: run an off-the-shelf COCO person detector on every view.

    python -m tools.pseudo.detect_2d -d wildtrack --root ~/Data/Wildtrack --out pseudo_raw/wildtrack_frcnn_v2.json

Keeps every person box with score >= --min_score (low on purpose: the real score threshold is chosen
later in build_pseudo.py without re-running the detector). Output:
    {"meta": {...}, "dets": {"<frame>": [[cam, x1, y1, x2, y2, score], ...]}}
in ORIGINAL image pixels. Only the images are read -- no annotation is used here.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image

from tools.pseudo.common import list_frames, load_base

COCO_PERSON = 1  # torchvision COCO label id


def build_detector(name, device, imgsz=1280):
    if name == 'frcnn_v2':
        from torchvision.models.detection import (FasterRCNN_ResNet50_FPN_V2_Weights,
                                                  fasterrcnn_resnet50_fpn_v2)
        model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
                                           box_score_thresh=0.0, box_detections_per_img=300)
        model.eval().to(device)

        @torch.no_grad()
        def run(imgs):
            import torchvision.transforms.functional as TF
            batch = [TF.to_tensor(im).to(device) for im in imgs]
            outs = model(batch)
            res = []
            for o in outs:
                keep = o['labels'] == COCO_PERSON
                res.append(torch.cat([o['boxes'][keep], o['scores'][keep, None]], 1).cpu().numpy())
            return res
        return run
    if name.startswith('yolo'):
        # e.g. --detector yolo:yolo26s.pt   (needs `pip install ultralytics`); person class only, at --imgsz
        from ultralytics import YOLO
        model = YOLO(name.split(':', 1)[1] if ':' in name else 'yolo26s.pt')

        def run(imgs):
            res = []
            for r in model.predict(imgs, classes=[0], conf=0.01, iou=0.6, imgsz=imgsz, verbose=False, device=device):
                b = r.boxes
                res.append(np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None]], 1))
            return res
        return run
    raise ValueError(name)


def main(args):
    base = load_base(args.dataset, args.root)
    ann_dir = os.path.join(base.root, 'annotations_positions')  # only used to know which frames exist
    frames = list_frames(ann_dir, base, args.split)
    if args.max_frames:
        frames = frames[:args.max_frames]
    img_fpaths = base.get_image_fpaths(frames)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    run = build_detector(args.detector, device, args.imgsz)
    print(f'{args.detector} on {device}: {len(frames)} frames x {base.num_cam} cams, split={args.split}')

    jobs = [(frame, cam) for frame in frames for cam in range(base.num_cam)]
    dets = {str(f): [] for f in frames}
    t0 = time.time()
    n_boxes = 0
    for s in range(0, len(jobs), args.batch):
        chunk = jobs[s:s + args.batch]
        imgs = [Image.open(img_fpaths[cam][frame]).convert('RGB') for frame, cam in chunk]
        for (frame, cam), boxes in zip(chunk, run(imgs)):
            boxes = boxes[boxes[:, 4] >= args.min_score]
            n_boxes += len(boxes)
            dets[str(frame)] += [[cam] + [round(float(v), 2) for v in b[:4]] + [round(float(b[4]), 4)] for b in boxes]
        if (s // args.batch) % 50 == 0:
            print(f'  {s + len(chunk)}/{len(jobs)} images, {n_boxes} boxes, {time.time() - t0:.0f}s')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    meta = {'dataset': args.dataset, 'detector': args.detector, 'split': args.split, 'min_score': args.min_score,
            'num_frames': len(frames), 'num_cam': base.num_cam, 'img_shape': list(base.img_shape),
            'num_boxes': n_boxes}
    with open(args.out, 'w') as f:
        json.dump({'meta': meta, 'dets': dets}, f)
    print(f'saved {args.out}: {n_boxes} boxes ({n_boxes / max(len(jobs), 1):.1f}/image), {time.time() - t0:.0f}s')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--dataset', default='wildtrack', choices=['wildtrack', 'multiviewx'])
    p.add_argument('--root', default='~/Data/Wildtrack')
    p.add_argument('--out', required=True)
    p.add_argument('--detector', default='frcnn_v2', help='frcnn_v2 | yolo[:weights.pt]')
    p.add_argument('--split', default='train', choices=['train', 'test', 'all'],
                   help='pseudo labels are only ever used on the train split')
    p.add_argument('--min_score', type=float, default=0.05)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--max_frames', type=int, default=0, help='debug: only the first N frames')
    p.add_argument('--device', default=None)
    p.add_argument('--imgsz', type=int, default=1280, help='YOLO inference size (Faster R-CNN uses the full image)')
    main(p.parse_args())
