"""
Merge the raw detections of several 2D detectors (detect_2d.py outputs) into one raw json, per image:
keep each detector's boxes above its own score threshold, then greedy NMS (IoU >= --iou) across the union so a person
found by both detectors is counted once. Scores of different detectors are not comparable, so the merged score is
1.0 for boxes found by >= 2 detectors and the detector's own score otherwise; run build_pseudo.py on the result with
--score_thr 0 (the per-detector thresholds were already applied here).

    python -m tools.pseudo.merge_dets --inputs raw_frcnn.json:0.5 raw_yolo.json:0.2 --out raw_merged.json
"""
import argparse
import json

import numpy as np


def iou(a, b):
    x1, y1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    x2, y2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[..., 2] - z[..., 0]) * (z[..., 3] - z[..., 1])
    return inter / (area(a) + area(b) - inter + 1e-9)


def merge_image(boxes, thr):
    """boxes: list of (x1, y1, x2, y2, score, source); returns merged list [x1, y1, x2, y2, score, n_sources]"""
    if not boxes:
        return []
    b = np.array([x[:5] for x in boxes], dtype=np.float64)
    src = np.array([x[5] for x in boxes])
    order = np.argsort(-b[:, 4])
    used = np.zeros(len(b), bool)
    out = []
    for i in order:
        if used[i]:
            continue
        group = [j for j in order if not used[j] and iou(b[i, :4], b[j:j + 1, :4])[0] >= thr]
        used[group] = True
        n_src = len(set(src[group].tolist()))
        out.append(b[i, :4].tolist() + [1.0 if n_src >= 2 else float(b[i, 4]), n_src])
    return out


def main(args):
    inputs = []
    for spec in args.inputs:
        path, thr = spec.rsplit(':', 1)
        with open(path) as f:
            inputs.append((json.load(f), float(thr)))
    frames = sorted({int(k) for raw, _ in inputs for k in raw['dets']})
    merged, n_in, n_out, n_both = {}, 0, 0, 0
    for frame in frames:
        per_cam = {}
        for src, (raw, thr) in enumerate(inputs):
            for cam, x1, y1, x2, y2, score in raw['dets'].get(str(frame), []):
                if score >= thr:
                    per_cam.setdefault(int(cam), []).append((x1, y1, x2, y2, score, src))
                    n_in += 1
        rows = []
        for cam, boxes in sorted(per_cam.items()):
            for x1, y1, x2, y2, score, n_src in merge_image(boxes, args.iou):
                rows.append([cam, round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2), round(score, 4)])
                n_both += n_src >= 2
        merged[str(frame)] = rows
        n_out += len(rows)
    meta = {'merged_from': args.inputs, 'iou': args.iou, 'boxes_in': n_in, 'boxes_out': n_out,
            'found_by_both': n_both, 'num_frames': len(frames),
            'detector': '+'.join(raw.get('meta', {}).get('detector', '?') for raw, _ in inputs)}
    with open(args.out, 'w') as f:
        json.dump({'meta': meta, 'dets': merged}, f)
    print(json.dumps(meta))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', nargs='+', required=True, help='raw.json:min_score per detector')
    p.add_argument('--iou', type=float, default=0.5)
    p.add_argument('--out', required=True)
    main(p.parse_args())
