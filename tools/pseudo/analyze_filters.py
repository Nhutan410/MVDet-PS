"""
Offline comparison of pseudo-label filters (no training): which rule for POSITIVE / IGNORE / background gives the
cleanest supervision. Uses the hidden people for evaluation only (like eval_pseudo.py).

    python -m tools.pseudo.analyze_filters -d wildtrack --root ~/Data/Wildtrack --full_ann <full>/annotations_positions \
        --kept_ann <drop60>/annotations_positions --pseudo <build_pseudo output without view filter>

Positive rule = n_views >= k and n_views / n_visible >= ratio and >= border cells from the grid edge and pseudo
support in >= temporal of the two neighbouring annotated frames. Ignore rule 'all' = every other point becomes an
ignore region. Columns: P / R of the positives vs hidden people @0.5 m, positives and ghosts per frame, ignore
regions per frame and the fraction of them that really cover a hidden person, hidden people left as background.
Also breaks the ghosts of the n_views >= 2 rule down by cause. Writes <pseudo>/filter_analysis.json.
"""
import argparse
import itertools
import json
import os

import numpy as np

from tools.pseudo.common import grid_shape_xy, load_ann, load_base, match_points, ped_grid, pseudo_fname
from tools.pseudo.make_oracle import hidden_people

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', default='wildtrack', choices=['wildtrack', 'multiviewx'])
ap.add_argument('--root', default='~/Data/Wildtrack')
ap.add_argument('--full_ann', required=True)
ap.add_argument('--kept_ann', required=True)
ap.add_argument('--pseudo', required=True)
ap.add_argument('--top', type=int, default=40, help='rows of the ranked table to print')
A = ap.parse_args()
FULL, KEPT = A.full_ann, A.kept_ann
THR = 20  # 0.5 m in grid cells

base = load_base(A.dataset, A.root)
sx, sy = grid_shape_xy(base)
pdir = A.pseudo
frames = sorted(int(f.split('.')[0]) for f in os.listdir(pdir) if f[0].isdigit())
P = {f: json.load(open(pseudo_fname(pdir, f))) for f in frames}
HID = {f: np.array([ped_grid(base, p) for p in hidden_people(base, FULL, KEPT, f)]).reshape(-1, 2) for f in frames}
KEP = {f: np.array([ped_grid(base, p) for p in load_ann(KEPT, f)]).reshape(-1, 2) for f in frames}
fidx = {f: i for i, f in enumerate(frames)}


def temporal_support(f, g, dist=40):
    """number of neighbouring annotated frames (+-1 step) that have a pseudo point within dist cells"""
    n = 0
    for nb in (fidx[f] - 1, fidx[f] + 1):
        if 0 <= nb < len(frames):
            q = np.array([p['grid'] for p in P[frames[nb]]]).reshape(-1, 2)
            if len(q) and np.linalg.norm(q - g, axis=1).min() <= dist:
                n += 1
    return n


def border_dist(g):
    return min(g[0], sx - 1 - g[0], g[1], sy - 1 - g[1])


# precompute per-point features
for f in frames:
    for p in P[f]:
        g = np.array(p['grid'])
        p['ratio'] = p['n_views'] / max(1, p['n_visible'])
        p['temporal'] = temporal_support(f, g)
        p['border'] = border_dist(g)


def evaluate(pos_rule, ign_rule):
    tp = npos = nign = ign_hidden = hid_bg = nhid = 0
    for f in frames:
        pos = [p for p in P[f] if pos_rule(p)]
        ign = [p for p in P[f] if not pos_rule(p) and ign_rule(p)]
        hid = HID[f]
        nhid += len(hid)
        gp = np.array([p['grid'] for p in pos]).reshape(-1, 2)
        gi = np.array([p['grid'] for p in ign]).reshape(-1, 2)
        m = match_points(gp, hid, THR)
        tp += len(m)
        npos += len(gp)
        nign += len(gi)
        matched_h = {j for _, j, _ in m}
        rest = np.array([h for j, h in enumerate(hid) if j not in matched_h]).reshape(-1, 2)
        mi = match_points(gi, rest, 40)  # ignore disk ~ 1 m
        ign_hidden += len(mi)
        hid_bg += len(rest) - len(mi)
    nf = len(frames)
    return dict(P=tp / max(npos, 1), R=tp / max(nhid, 1), pos_f=npos / nf, fp_f=(npos - tp) / nf, ign_f=nign / nf,
                ign_hid_frac=ign_hidden / max(nign, 1), hidden_as_bg_f=hid_bg / nf)


rows = []
for k, ratio, border, temp in itertools.product([2, 3, 4], [0, 0.5, 0.75], [0, 20, 40], [0, 1, 2]):
    pos_rule = lambda p, k=k, ratio=ratio, border=border, temp=temp: (
        p['n_views'] >= k and p['ratio'] >= ratio and p['border'] >= border and p['temporal'] >= temp)
    for ign_name, ign_rule in [('none', lambda p: False), ('all', lambda p: True)]:
        r = evaluate(pos_rule, ign_rule)
        rows.append(dict(k=k, ratio=ratio, border=border, temporal=temp, ignore=ign_name, **r))

print('(ranked by F1 of the positives, P >= 0.75 only; top rows without and with the ignore tier)')
print(f'{len(frames)} frames, hidden/frame {sum(len(h) for h in HID.values()) / len(frames):.1f}')
print(f"{'k':>2} {'ratio':>5} {'bord':>4} {'temp':>4} {'ign':>4} | {'P':>5} {'R':>5} {'pos/f':>5} {'FP/f':>5} "
      f"{'ign/f':>5} {'ign=hid':>7} {'hid->bg/f':>9}")
ranked = sorted(rows, key=lambda r: -r['P'] * r['R'] / (r['P'] + r['R'] + 1e-9))
for r in [r for r in ranked if r['P'] >= 0.75 and r['ignore'] == 'none'][:A.top // 2] + \
        [r for r in ranked if r['P'] >= 0.75 and r['ignore'] == 'all'][:A.top // 2]:
    print(f"{r['k']:>2} {r['ratio']:>5} {r['border']:>4} {r['temporal']:>4} {r['ignore']:>4} | {r['P']:.3f} {r['R']:.3f} "
          f"{r['pos_f']:5.1f} {r['fp_f']:5.1f} {r['ign_f']:5.1f} {r['ign_hid_frac']:7.3f} {r['hidden_as_bg_f']:9.1f}")

# ghost breakdown for the currently used rule (n_views >= 2)
cause = {'dup_of_kept (0.5-1m)': 0, 'near_hidden (0.5-1m)': 0, 'border<1m': 0, 'isolated': 0}
by_views_temporal = {}
n_ghost = 0
for f in frames:
    pos = [p for p in P[f] if p['n_views'] >= 2]
    gp = np.array([p['grid'] for p in pos]).reshape(-1, 2)
    matched = {i for i, _, _ in match_points(gp, HID[f], THR)}
    for i, p in enumerate(pos):
        key = (min(p['n_views'], 4), p['temporal'])
        by_views_temporal.setdefault(key, [0, 0])
        by_views_temporal[key][1] += 1
        if i in matched:
            by_views_temporal[key][0] += 1
            continue
        n_ghost += 1
        g = gp[i]
        dk = np.linalg.norm(KEP[f] - g, axis=1).min() if len(KEP[f]) else 1e9
        dh = np.linalg.norm(HID[f] - g, axis=1).min() if len(HID[f]) else 1e9
        if dk <= 40:
            cause['dup_of_kept (0.5-1m)'] += 1
        elif dh <= 40:
            cause['near_hidden (0.5-1m)'] += 1
        elif p['border'] < 40:
            cause['border<1m'] += 1
        else:
            cause['isolated'] += 1
print(f'\nghosts among n_views>=2 positives: {n_ghost} ({n_ghost / len(frames):.1f}/frame)')
for k, v in cause.items():
    print(f'  {k:24s} {v:6d}  {v / max(n_ghost, 1):.1%}')
print('\nprecision by (n_views, temporal support 0/1/2):')
for (v, t), (a, n) in sorted(by_views_temporal.items()):
    print(f'  views {v}{"+" if v == 4 else " "} temporal {t}: {a / n:.3f}  (n={n})')
json.dump(rows, open(os.path.join(pdir, 'filter_analysis.json'), 'w'), indent=1)
