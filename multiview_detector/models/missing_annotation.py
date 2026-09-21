"""
Building blocks for the missing-annotation robust loss (see MISSING_ANNOTATION_LOSS.md):

  * EvidenceBranch  -- per-view evidence s_v(u, v): an INDEPENDENT backbone (frozen DINOv2 or a
                       separate trainable ResNet-18) + a tiny conv head. Shares NO parameter with
                       the main MVDet backbone, so its output can be used as an external judge of
                       the main branch without a self-referential loop.
  * GateHead        -- q(x, y) on the main BEV trunk (input detached): "how likely is this y_obs=0
                       cell a missing annotation". Trained only by L_q = (q - stopgrad(c))^2.
  * compute_visibility_masks / aggregate_consensus -- project (x, y, 0) through the calibration to
                       decide which cameras see each BEV cell, then take the median / 25th
                       percentile / min of the projected evidence over those views -> c(x, y).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.geometry.transform import warp_perspective

from multiview_detector.models.resnet import resnet18

CONSENSUS_METHODS = ('median', 'percentile25', 'min')
EVIDENCE_BACKBONES = ('frozen_dinov2', 'separate_trainable')


def compute_visibility_masks(dataset, grid_reduce=None):
    """
    Boolean tensor [num_cam, X, Y] on the reduced BEV grid: True where camera `cam` sees the
    ground point of that cell. A cell is visible when its world point (x, y, z=0) lies in front of
    the camera (depth > 0) and projects inside the image. Same intrinsic / extrinsic matrices that
    build the feature-projection homographies, so the mask agrees with what the warp samples --
    plus the depth test the homography alone cannot express.
    """
    base = dataset.base
    grid_reduce = dataset.grid_reduce if grid_reduce is None else grid_reduce
    X, Y = dataset.reducedgrid_shape
    H, W = dataset.img_shape
    ii, jj = np.meshgrid(np.arange(X), np.arange(Y), indexing='ij')
    # cell centre on the full-resolution world grid; frameDataset.download maps (x, y) -> (i, j)
    # as i = x / gr, j = y / gr for 'ij' indexing and i = y / gr, j = x / gr for 'xy' indexing.
    ci, cj = (ii + 0.5) * grid_reduce, (jj + 0.5) * grid_reduce
    if base.indexing == 'xy':
        grid_x, grid_y = cj, ci
    else:
        grid_x, grid_y = ci, cj
    grid_h = np.stack([grid_x.ravel(), grid_y.ravel(), np.ones(X * Y)], axis=0)  # [3, XY]
    world_xy = base.worldgrid2worldcoord_mat @ grid_h  # [3, XY] homogeneous (x, y, 1)
    world_xyz1 = np.stack([world_xy[0] / world_xy[2], world_xy[1] / world_xy[2],
                           np.zeros(X * Y), np.ones(X * Y)], axis=0)  # z = 0 ground plane
    masks = np.zeros([dataset.num_cam, X, Y], dtype=bool)
    for cam in range(dataset.num_cam):
        cam_xyz = np.asarray(base.extrinsic_matrices[cam], dtype=np.float64) @ world_xyz1  # [3, XY]
        depth = cam_xyz[2]
        img = np.asarray(base.intrinsic_matrices[cam], dtype=np.float64) @ cam_xyz
        with np.errstate(divide='ignore', invalid='ignore'):
            u, v = img[0] / img[2], img[1] / img[2]
        vis = (depth > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        masks[cam] = vis.reshape(X, Y)
    return torch.from_numpy(masks)


def aggregate_consensus(s_bev, visible, method='median', min_visible_views=2):
    """
    s_bev:   [B, N, X, Y] per-view evidence already projected on the BEV grid (probabilities).
    visible: [N, X, Y] or [B, N, X, Y] bool.
    Returns (c, n_vis): c is [B, 1, X, Y], DETACHED; n_vis is [B, X, Y] (number of visible views).
    Cells seen by fewer than `min_visible_views` cameras get c = 0 (not enough independent
    evidence -> default to trusting the background label). Mean is deliberately not offered: one
    or two hallucinating views would drag it up; median / low percentile / min need agreement.
    'median' is torch's LOWER median (even view count -> the smaller middle value), i.e. with two
    visible views it equals 'min' -- the conservative side.
    """
    if method not in CONSENSUS_METHODS:
        raise ValueError(f'consensus_agg must be one of {CONSENSUS_METHODS}, got {method}')
    s_bev = s_bev.detach()
    if visible.dim() == 3:
        visible = visible.unsqueeze(0)
    visible = visible.to(s_bev.device).expand_as(s_bev)
    n_vis = visible.sum(dim=1)
    if method == 'min':
        c = s_bev.masked_fill(~visible, float('inf')).amin(dim=1)
    else:
        x = s_bev.masked_fill(~visible, float('nan'))
        if method == 'median':
            c = x.nanmedian(dim=1).values
        else:
            c = torch.nanquantile(x, 0.25, dim=1)
    c = torch.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    c = c.masked_fill(n_vis < min_visible_views, 0.0)
    return c.unsqueeze(1).detach(), n_vis


class GateHead(nn.Module):
    """q(x, y) in (0, 1) from the main BEV trunk. The trunk input is detached inside forward so
    L_q never shapes the main features; only these few conv weights learn to predict c."""

    def __init__(self, in_channels=512, hidden=64, init_bias=-4.0):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_channels, hidden, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(hidden, 1, 3, padding=1))
        # start at q = sigmoid(-4) ~ 0.02 so switching L_conf on after warm-up is smooth
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, init_bias)

    def forward(self, trunk, out_shape=None):
        logit = self.net(trunk.detach())
        if out_shape is not None and tuple(logit.shape[-2:]) != tuple(out_shape):
            logit = F.interpolate(logit, out_shape, mode='bilinear', align_corners=False)
        return torch.sigmoid(logit)


class EvidenceBranch(nn.Module):
    """
    Per-view evidence s_v. Returns LOGITS at `out_shape` (the same resolution the main per-view
    features are projected from, so the same proj_mats can warp them to the BEV).

    backbone='frozen_dinov2'      : torch.hub facebookresearch/dinov2 ViT-S/14 (or `dino_name`),
                                    frozen and always in eval mode; only `head` trains.
    backbone='separate_trainable' : ImageNet ResNet-18 (dilated like the main one) with its own
                                    weights; trained by L_s only, through its own optimizer.
    """

    def __init__(self, backbone='frozen_dinov2', out_shape=(270, 480), dino_name='dinov2_vits14',
                 dino_input=(504, 896), hidden=64):
        super().__init__()
        if backbone not in EVIDENCE_BACKBONES:
            raise ValueError(f's_v_backbone must be one of {EVIDENCE_BACKBONES}, got {backbone}')
        self.backbone_type = backbone
        self.out_shape = list(out_shape)
        self.dino_input = list(dino_input)
        if backbone == 'frozen_dinov2':
            if any(s % 14 for s in self.dino_input):
                raise ValueError(f'dino_input {self.dino_input} must be a multiple of the patch size 14')
            self.backbone = torch.hub.load('facebookresearch/dinov2', dino_name)
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.eval()
            feat_dim = self.backbone.embed_dim
        else:
            self.backbone = nn.Sequential(*list(resnet18(pretrained=True, replace_stride_with_dilation=[False, True, True])
                                               .children())[:-2])
            feat_dim = 512
        self.head = nn.Sequential(nn.Conv2d(feat_dim, hidden, 1), nn.ReLU(), nn.Conv2d(hidden, 1, 1))
        nn.init.constant_(self.head[-1].bias, -2.0)  # start around s ~ 0.12 (people are rare)

    @property
    def frozen(self):
        return self.backbone_type == 'frozen_dinov2'

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.backbone.eval()  # never let BN / dropout state of the frozen backbone drift
        return self

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def features(self, img):
        if self.frozen:
            x = F.interpolate(img, self.dino_input, mode='bilinear', align_corners=False)
            with torch.no_grad():
                if x.is_cuda:
                    with torch.autocast('cuda', dtype=torch.float16):
                        tokens = self.backbone.forward_features(x)['x_norm_patchtokens']
                else:
                    tokens = self.backbone.forward_features(x)['x_norm_patchtokens']
            B, _, C = tokens.shape
            h, w = self.dino_input[0] // 14, self.dino_input[1] // 14
            return tokens.float().reshape(B, h, w, C).permute(0, 3, 1, 2).contiguous()
        return self.backbone(img)

    def forward(self, img):
        logit = self.head(self.features(img))
        return F.interpolate(logit, self.out_shape, mode='bilinear', align_corners=False)


def project_views_to_bev(view_maps, proj_mats, bev_shape):
    """view_maps: list of N tensors [B, C, h, w]; proj_mats: list of N [3, 3]. -> [B, N*C, X, Y]"""
    out = []
    for cam, vm in enumerate(view_maps):
        B = vm.shape[0]
        M = proj_mats[cam].repeat([B, 1, 1]).float().to(vm.device)
        out.append(warp_perspective(vm, M, bev_shape))
    return torch.cat(out, dim=1)


def project_bev_to_views(bev_map, proj_mats_inv, view_shape):
    """bev_map: [B, C, X, Y]; proj_mats_inv: list of N [3, 3] (inverse of proj_mats). -> list of N [B, C, h, w]"""
    out = []
    B = bev_map.shape[0]
    for M_inv in proj_mats_inv:
        M = M_inv.repeat([B, 1, 1]).float().to(bev_map.device)
        out.append(warp_perspective(bev_map, M, view_shape))
    return out
