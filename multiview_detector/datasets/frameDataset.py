import os
import json
from scipy.stats import multivariate_normal
from PIL import Image
from scipy.sparse import coo_matrix
from torchvision.datasets import VisionDataset
import torch
from torchvision.transforms import ToTensor
from multiview_detector.utils.projection import *


class frameDataset(VisionDataset):
    def __init__(self, base, train=True, transform=ToTensor(), target_transform=ToTensor(),
                 reID=False, grid_reduce=4, img_reduce=4, train_ratio=0.9, force_download=True,
                 pseudo_dir=None, pseudo_cfg=None, view_ignore_dets=None, view_ignore_score=0.5):
        super().__init__(base.root, transform=transform, target_transform=target_transform)

        map_sigma, map_kernel_size = 20 / grid_reduce, 20
        img_sigma, img_kernel_size = 10 / img_reduce, 10
        self.reID, self.grid_reduce, self.img_reduce = reID, grid_reduce, img_reduce

        self.base = base
        self.root, self.num_cam, self.num_frame = base.root, base.num_cam, base.num_frame
        self.img_shape, self.worldgrid_shape = base.img_shape, base.worldgrid_shape  # H,W; N_row,N_col
        self.reducedgrid_shape = list(map(lambda x: int(x / self.grid_reduce), self.worldgrid_shape))

        if train:
            frame_range = range(0, int(self.num_frame * train_ratio))
        else:
            frame_range = range(int(self.num_frame * train_ratio), self.num_frame)

        self.img_fpaths = self.base.get_image_fpaths(frame_range)
        self.map_gt = {}
        self.imgs_head_foot_gt = {}
        self.download(frame_range)

        # $MVDET_CACHE_DIR: write gt.txt outside a read-only dataset mount, one dir per parallel run
        cache_root = os.environ.get('MVDET_CACHE_DIR')
        self.gt_fpath = os.path.join(os.path.expanduser(cache_root), self.base.__name__.lower(), 'gt.txt') \
            if cache_root else os.path.join(self.root, 'gt.txt')
        if not os.path.exists(self.gt_fpath) or force_download:
            self.prepare_gt()

        # pseudo labels (train split only): frame -> (N, 3) [row, col, alpha] on the
        # output map; alpha < 0 marks an ignore-only point
        self.pseudo = None
        if pseudo_dir is not None:
            assert train, 'pseudo labels are only ever used on the train split'
            self.load_pseudo(pseudo_dir, pseudo_cfg or {})

        # per-view ignore boxes (train split only): 2D detector boxes (score >= view_ignore_score) whose background
        # pixels get weight 0 in the per-view head/foot loss -- unlabelled people are not taught as "no person"
        self.view_ignore = None
        if view_ignore_dets is not None:
            assert train, 'view-ignore boxes are only ever used on the train split'
            self.load_view_ignore(view_ignore_dets, view_ignore_score)

        x, y = np.meshgrid(np.arange(-map_kernel_size, map_kernel_size + 1),
                           np.arange(-map_kernel_size, map_kernel_size + 1))
        pos = np.stack([x, y], axis=2)
        map_kernel = multivariate_normal.pdf(pos, [0, 0], np.identity(2) * map_sigma)
        map_kernel = map_kernel / map_kernel.max()
        kernel_size = map_kernel.shape[0]
        self.map_kernel = torch.zeros([1, 1, kernel_size, kernel_size], requires_grad=False)
        self.map_kernel[0, 0] = torch.from_numpy(map_kernel)

        x, y = np.meshgrid(np.arange(-img_kernel_size, img_kernel_size + 1),
                           np.arange(-img_kernel_size, img_kernel_size + 1))
        pos = np.stack([x, y], axis=2)
        img_kernel = multivariate_normal.pdf(pos, [0, 0], np.identity(2) * img_sigma)
        img_kernel = img_kernel / img_kernel.max()
        kernel_size = img_kernel.shape[0]
        self.img_kernel = torch.zeros([2, 2, kernel_size, kernel_size], requires_grad=False)
        self.img_kernel[0, 0] = torch.from_numpy(img_kernel)
        self.img_kernel[1, 1] = torch.from_numpy(img_kernel)
        pass

    def prepare_gt(self):
        og_gt = []
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                all_pedestrians = json.load(json_file)
            for single_pedestrian in all_pedestrians:
                def is_in_cam(cam):
                    return not (single_pedestrian['views'][cam]['xmin'] == -1 and
                                single_pedestrian['views'][cam]['xmax'] == -1 and
                                single_pedestrian['views'][cam]['ymin'] == -1 and
                                single_pedestrian['views'][cam]['ymax'] == -1)

                in_cam_range = sum(is_in_cam(cam) for cam in range(self.num_cam))
                if not in_cam_range:
                    continue
                grid_x, grid_y = self.base.get_worldgrid_from_pos(single_pedestrian['positionID'])
                og_gt.append(np.array([frame, grid_x, grid_y]))
        og_gt = np.stack(og_gt, axis=0)
        os.makedirs(os.path.dirname(self.gt_fpath), exist_ok=True)
        np.savetxt(self.gt_fpath, og_gt, '%d')

    def download(self, frame_range):
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            if frame in frame_range:
                with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                    all_pedestrians = json.load(json_file)
                i_s, j_s, v_s = [], [], []
                head_row_cam_s, head_col_cam_s = [[] for _ in range(self.num_cam)], \
                                                 [[] for _ in range(self.num_cam)]
                foot_row_cam_s, foot_col_cam_s, v_cam_s = [[] for _ in range(self.num_cam)], \
                                                          [[] for _ in range(self.num_cam)], \
                                                          [[] for _ in range(self.num_cam)]
                for single_pedestrian in all_pedestrians:
                    x, y = self.base.get_worldgrid_from_pos(single_pedestrian['positionID'])
                    if self.base.indexing == 'xy':
                        i_s.append(int(y / self.grid_reduce))
                        j_s.append(int(x / self.grid_reduce))
                    else:
                        i_s.append(int(x / self.grid_reduce))
                        j_s.append(int(y / self.grid_reduce))
                    v_s.append(single_pedestrian['personID'] + 1 if self.reID else 1)
                    for cam in range(self.num_cam):
                        x = max(min(int((single_pedestrian['views'][cam]['xmin'] +
                                         single_pedestrian['views'][cam]['xmax']) / 2), self.img_shape[1] - 1), 0)
                        y_head = max(single_pedestrian['views'][cam]['ymin'], 0)
                        y_foot = min(single_pedestrian['views'][cam]['ymax'], self.img_shape[0] - 1)
                        if x > 0 and y > 0:
                            head_row_cam_s[cam].append(y_head)
                            head_col_cam_s[cam].append(x)
                            foot_row_cam_s[cam].append(y_foot)
                            foot_col_cam_s[cam].append(x)
                            v_cam_s[cam].append(single_pedestrian['personID'] + 1 if self.reID else 1)
                occupancy_map = coo_matrix((v_s, (i_s, j_s)), shape=self.reducedgrid_shape)
                self.map_gt[frame] = occupancy_map
                self.imgs_head_foot_gt[frame] = {}
                for cam in range(self.num_cam):
                    img_gt_head = coo_matrix((v_cam_s[cam], (head_row_cam_s[cam], head_col_cam_s[cam])),
                                             shape=self.img_shape)
                    img_gt_foot = coo_matrix((v_cam_s[cam], (foot_row_cam_s[cam], foot_col_cam_s[cam])),
                                             shape=self.img_shape)
                    self.imgs_head_foot_gt[frame][cam] = [img_gt_head, img_gt_foot]

    def load_pseudo(self, pseudo_dir, cfg):
        """<pseudo_dir>/<frame:08d>.json from tools/pseudo (grid = full-res [gx, gy], same convention as the
        annotations). cfg: alpha ('const' | 'score' | 'views'), alpha_const, views_k, min_score, min_views,
        min_view_ratio, ignore_min_views, border.

        Three tiers per point: POSITIVE (passes min_score / min_views / min_view_ratio, or "tier": "pos" in the
        json) -> row alpha > 0; IGNORE (fails them but n_views >= ignore_min_views > 0, or "tier": "ign") -> row
        alpha = -1, the loss neither pushes it up nor treats it as background; otherwise dropped (background).
        Points closer than `border` full-res grid cells to the edge of the annotated area are always dropped: most
        projection ghosts are people standing just outside it, so they must stay background (not even ignore)."""
        mode, a_const = cfg.get('alpha', 'const'), cfg.get('alpha_const', 0.5)
        views_k, min_score, min_views = cfg.get('views_k', 3), cfg.get('min_score', 0.0), cfg.get('min_views', 1)
        min_view_ratio, ignore_min_views = cfg.get('min_view_ratio', 0.0), cfg.get('ignore_min_views', 0)
        border = cfg.get('border', 0.0)
        # extent of the grid along gx / gy (same convention as the annotations)
        sx, sy = (self.worldgrid_shape[1], self.worldgrid_shape[0]) if self.base.indexing == 'xy' else \
            (self.worldgrid_shape[0], self.worldgrid_shape[1])
        self.pseudo, missing, n_raw, alphas, n_ign, n_border = {}, 0, 0, [], 0, 0
        for frame in self.map_gt:
            fpath = os.path.join(pseudo_dir, f'{frame:08d}.json')
            if not os.path.isfile(fpath):
                missing += 1
                self.pseudo[frame] = np.zeros([0, 3], np.float32)
                continue
            with open(fpath) as f:
                pts = json.load(f)
            n_raw += len(pts)
            rows = []
            for p in pts:
                gx, gy = p['grid']
                if border > 0 and min(gx, sx - 1 - gx, gy, sy - 1 - gy) < border:
                    n_border += 1
                    continue
                if 'tier' in p:
                    tier = p['tier']
                else:
                    ratio = p['n_views'] / max(1, p.get('n_visible', p['n_views']))
                    if p['score'] >= min_score and p['n_views'] >= min_views and ratio >= min_view_ratio:
                        tier = 'pos'
                    elif ignore_min_views > 0 and p['n_views'] >= ignore_min_views:
                        tier = 'ign'
                    else:
                        tier = 'drop'
                if tier == 'drop':
                    continue
                if tier == 'ign':
                    alpha = -1.0
                elif mode == 'const':
                    alpha = a_const
                elif mode == 'score':
                    alpha = p['score']
                elif mode == 'views':
                    # agreement relative to how many cameras can see that spot at all
                    alpha = p['score'] * min(1.0, p['n_views'] / max(1, min(views_k, p.get('n_visible', views_k))))
                else:
                    raise ValueError(mode)
                # same index convention as download(): row/col on the reduced map (float, floored in the loss)
                if self.base.indexing == 'xy':
                    rows.append([gy / self.grid_reduce, gx / self.grid_reduce, alpha])
                else:
                    rows.append([gx / self.grid_reduce, gy / self.grid_reduce, alpha])
                if alpha > 0:
                    alphas.append(alpha)
                else:
                    n_ign += 1
            self.pseudo[frame] = np.array(rows, np.float32).reshape(-1, 3)
        if missing == len(self.map_gt):
            raise FileNotFoundError(f'no pseudo-label file for any train frame in {pseudo_dir}')
        n, nf = len(alphas), max(len(self.pseudo), 1)
        print(f'[pseudo] {pseudo_dir}: {n_raw} points -> {n} positive ({n / nf:.1f}/frame) + {n_ign} ignore-only '
              f'({n_ign / nf:.1f}/frame) over {len(self.pseudo)} train frames (min_score {min_score}, min_views '
              f'{min_views}, min_view_ratio {min_view_ratio}, ignore_min_views {ignore_min_views}, border {border}: '
              f'{n_border} dropped), '
              f'alpha={mode} mean {np.mean(alphas) if alphas else 0:.3f}, frames without file: {missing}')

    def load_view_ignore(self, dets_json, min_score):
        """dets_json: tools/pseudo/detect_2d.py output {"dets": {frame: [[cam, x1, y1, x2, y2, score], ...]}}"""
        with open(dets_json) as f:
            dets = json.load(f)['dets']
        self.view_ignore, n = {}, 0
        for frame in self.map_gt:
            boxes = [[] for _ in range(self.num_cam)]
            for cam, x1, y1, x2, y2, score in dets.get(str(frame), []):
                if score >= min_score:
                    boxes[int(cam)].append((x1, y1, x2, y2))
                    n += 1
            self.view_ignore[frame] = boxes
        print(f'[view-ignore] {dets_json}: {n} boxes (score >= {min_score}) over {len(self.view_ignore)} train frames '
              f'({n / max(len(self.view_ignore) * self.num_cam, 1):.1f}/image)')

    def view_ignore_mask(self, frame):
        """(num_cam, H / img_reduce, W / img_reduce) bool: True inside a detector box"""
        h, w = self.img_shape[0] // self.img_reduce, self.img_shape[1] // self.img_reduce
        m = torch.zeros(self.num_cam, h, w, dtype=torch.bool)
        for cam, boxes in enumerate(self.view_ignore[frame]):
            for x1, y1, x2, y2 in boxes:
                c1, r1 = max(int(x1 // self.img_reduce), 0), max(int(y1 // self.img_reduce), 0)
                c2, r2 = min(int(np.ceil(x2 / self.img_reduce)), w), min(int(np.ceil(y2 / self.img_reduce)), h)
                if c2 > c1 and r2 > r1:
                    m[cam, r1:r2, c1:c2] = True
        return m

    def __getitem__(self, index):
        frame = list(self.map_gt.keys())[index]
        imgs = []
        for cam in range(self.num_cam):
            fpath = self.img_fpaths[cam][frame]
            img = Image.open(fpath).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)
        imgs = torch.stack(imgs)
        map_gt = self.map_gt[frame].toarray()
        if self.reID:
            map_gt = (map_gt > 0).int()
        if self.target_transform is not None:
            map_gt = self.target_transform(map_gt)
        imgs_gt = []
        for cam in range(self.num_cam):
            img_gt_head = self.imgs_head_foot_gt[frame][cam][0].toarray()
            img_gt_foot = self.imgs_head_foot_gt[frame][cam][1].toarray()
            img_gt = np.stack([img_gt_head, img_gt_foot], axis=2)
            if self.reID:
                img_gt = (img_gt > 0).int()
            if self.target_transform is not None:
                img_gt = self.target_transform(img_gt)
            imgs_gt.append(img_gt.float())
        extra = []
        if self.pseudo is not None or self.view_ignore is not None:
            extra.append(torch.from_numpy(self.pseudo[frame]) if self.pseudo is not None else torch.zeros(0, 3))
        if self.view_ignore is not None:
            extra.append(self.view_ignore_mask(frame))
        return (imgs, map_gt.float(), imgs_gt, frame, *extra)

    def __len__(self):
        return len(self.map_gt.keys())


def test():
    from multiview_detector.datasets.Wildtrack import Wildtrack
    # from multiview_detector.datasets.MultiviewX import MultiviewX
    from multiview_detector.utils.projection import get_worldcoord_from_imagecoord
    dataset = frameDataset(Wildtrack(os.path.expanduser('~/Data/Wildtrack')))
    # test projection
    world_grid_maps = []
    xx, yy = np.meshgrid(np.arange(0, 1920, 20), np.arange(0, 1080, 20))
    H, W = xx.shape
    image_coords = np.stack([xx, yy], axis=2).reshape([-1, 2])
    import matplotlib.pyplot as plt
    for cam in range(dataset.num_cam):
        world_coords = get_worldcoord_from_imagecoord(image_coords.transpose(), dataset.base.intrinsic_matrices[cam],
                                                      dataset.base.extrinsic_matrices[cam])
        world_grids = dataset.base.get_worldgrid_from_worldcoord(world_coords).transpose().reshape([H, W, 2])
        world_grid_map = np.zeros(dataset.worldgrid_shape)
        for i in range(H):
            for j in range(W):
                x, y = world_grids[i, j]
                if dataset.base.indexing == 'xy':
                    if x in range(dataset.worldgrid_shape[1]) and y in range(dataset.worldgrid_shape[0]):
                        world_grid_map[int(y), int(x)] += 1
                else:
                    if x in range(dataset.worldgrid_shape[0]) and y in range(dataset.worldgrid_shape[1]):
                        world_grid_map[int(x), int(y)] += 1
        world_grid_map = world_grid_map != 0
        plt.imshow(world_grid_map)
        plt.show()
        world_grid_maps.append(world_grid_map)
        pass
    plt.imshow(np.sum(np.stack(world_grid_maps), axis=0))
    plt.show()
    pass
    imgs, map_gt, imgs_gt, _ = dataset.__getitem__(0)
    pass


if __name__ == '__main__':
    test()
