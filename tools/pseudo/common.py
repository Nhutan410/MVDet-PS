"""
Shared helpers of the pseudo-label pipeline.

Coordinate conventions -- everything reuses the dataset classes, nothing is re-derived:
  * "grid" = the dataset's full-resolution world grid [gx, gy], exactly what
    base.get_worldgrid_from_pos(positionID) returns (1 cell = 2.5 cm on Wildtrack and MultiviewX).
  * image -> world uses multiview_detector.utils.projection (the same K[r1 r2 t] homography MVDet
    uses for its own perspective warp), world -> grid uses base.get_worldgrid_from_worldcoord.
  * Wildtrack images are already undistorted (calibrations/intrinsic_zero), so no distortion step.
"""
import json
import os

import numpy as np

from multiview_detector.datasets import MultiviewX, Wildtrack
from multiview_detector.utils.projection import get_imagecoord_from_worldcoord, get_worldcoord_from_imagecoord

CELL_CM = 2.5  # full-resolution grid cell, both datasets


def load_base(dataset, root):
    root = os.path.expanduser(root)
    if dataset == 'wildtrack':
        return Wildtrack(root)
    if dataset == 'multiviewx':
        return MultiviewX(root)
    raise ValueError(dataset)


def train_cutoff(base, train_ratio=0.9):
    """frames < cutoff are the train split -- same rule as frameDataset"""
    return int(base.num_frame * train_ratio)


def list_frames(ann_dir, base, split='train', train_ratio=0.9):
    cutoff = train_cutoff(base, train_ratio)
    frames = sorted(int(f.split('.')[0]) for f in os.listdir(ann_dir) if f.endswith('.json'))
    if split == 'train':
        return [f for f in frames if f < cutoff]
    if split == 'test':
        return [f for f in frames if f >= cutoff]
    return frames


def load_ann(ann_dir, frame):
    with open(os.path.join(ann_dir, f'{frame:08d}.json')) as f:
        return json.load(f)


def ped_grid(base, ped):
    return base.get_worldgrid_from_pos(ped['positionID']).astype(np.float64)


def ped_in_any_cam(ped, num_cam):
    """frameDataset.prepare_gt drops people with no box in any view -- mirror it"""
    return any(not (v['xmin'] == -1 and v['xmax'] == -1 and v['ymin'] == -1 and v['ymax'] == -1)
               for v in ped['views'][:num_cam])


def grid_shape_xy(base):
    """(size along gx, size along gy)"""
    if base.indexing == 'xy':
        return base.worldgrid_shape[1], base.worldgrid_shape[0]
    return base.worldgrid_shape[0], base.worldgrid_shape[1]


def in_grid(base, grid):
    """grid: (N, 2) [gx, gy]"""
    sx, sy = grid_shape_xy(base)
    grid = np.asarray(grid).reshape(-1, 2)
    return (grid[:, 0] >= 0) & (grid[:, 0] < sx) & (grid[:, 1] >= 0) & (grid[:, 1] < sy)


def image_to_grid(base, cam, uv):
    """uv: (N, 2) pixel coords on the ORIGINAL image -> (N, 2) grid [gx, gy] on the ground plane Z = 0"""
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    world = get_worldcoord_from_imagecoord(uv.T, base.intrinsic_matrices[cam], base.extrinsic_matrices[cam])
    return base.get_worldgrid_from_worldcoord(world).T.astype(np.float64)


def grid_to_world(base, grid):
    return base.get_worldcoord_from_worldgrid(np.asarray(grid, dtype=np.float64).reshape(-1, 2).T)


def n_visible_cams(base, grid):
    """number of cameras whose image contains the ground point (in front of the camera)"""
    world = grid_to_world(base, grid)  # (2, N)
    H, W = base.img_shape
    count = np.zeros(world.shape[1], dtype=int)
    hom = np.concatenate([world, np.zeros([1, world.shape[1]]), np.ones([1, world.shape[1]])], 0)
    for cam in range(base.num_cam):
        depth = (base.extrinsic_matrices[cam] @ hom)[2]
        uv = get_imagecoord_from_worldcoord(world, base.intrinsic_matrices[cam], base.extrinsic_matrices[cam])
        count += (depth > 0) & (uv[0] >= 0) & (uv[0] < W) & (uv[1] >= 0) & (uv[1] < H)
    return count


def match_points(pred, gt, thr):
    """Hungarian matching of two point sets (N,2)/(M,2) in grid cells, pairs with distance <= thr.
    Returns list of (i_pred, j_gt, dist)."""
    from scipy.optimize import linear_sum_assignment
    pred, gt = np.asarray(pred).reshape(-1, 2), np.asarray(gt).reshape(-1, 2)
    if len(pred) == 0 or len(gt) == 0:
        return []
    d = np.linalg.norm(pred[:, None] - gt[None], axis=-1)
    cost = np.where(d <= thr, d, 1e6)
    rows, cols = linear_sum_assignment(cost)
    return [(i, j, d[i, j]) for i, j in zip(rows, cols) if d[i, j] <= thr]


def pseudo_fname(out_dir, frame):
    return os.path.join(out_dir, f'{frame:08d}.json')
