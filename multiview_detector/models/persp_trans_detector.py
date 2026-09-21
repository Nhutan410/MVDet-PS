import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.geometry.transform import warp_perspective
from torchvision.models.vgg import vgg11
from multiview_detector.models.resnet import resnet18
from multiview_detector.models.missing_annotation import (EvidenceBranch, GateHead, compute_visibility_masks,
                                                          aggregate_consensus, project_views_to_bev,
                                                          project_bev_to_views)

import matplotlib.pyplot as plt


class PerspTransDetector(nn.Module):
    def __init__(self, dataset, arch='resnet18', mal_cfg=None):
        """
        mal_cfg: None -> original MVDet. Otherwise a dict enabling the missing-annotation branches:
            s_v_backbone ('frozen_dinov2' | 'separate_trainable'), consensus_agg, min_visible_views,
            dino_name, dino_input, ev_device -- see multiview_detector/models/missing_annotation.py.

        Device layout (2 GPUs, e.g. Kaggle T4 x2):
            cuda:1  base_pt1 (ResNet layers 1-3, the high-resolution activations)
                    + the whole evidence branch s_v (own backbone + head)  <- ev_device='auto'
            cuda:0  base_pt2, img_classifier, warp, map_classifier (heaviest), gate head, consensus
        The evidence branch shares nothing with the main branch, so its forward on cuda:1 overlaps
        with the main branch on cuda:0. With 1 GPU everything is on cuda:0; with no CUDA, on cpu.
        """
        super().__init__()
        # Original code hardcodes a 2-GPU split (backbone half on cuda:1, the rest on cuda:0) --
        # a VRAM workaround from the original paper's hardware, not a correctness requirement.
        # Colab (and some Kaggle sessions) only expose 1 GPU, so fall back to putting everything
        # on cuda:0 there; keeps the original 2-GPU split when a second GPU is actually available.
        # No CUDA at all (local CPU smoke tests) -> everything on cpu.
        if torch.cuda.is_available():
            self.device_pt1 = 'cuda:1' if torch.cuda.device_count() >= 2 else 'cuda:0'
            self.device_pt2 = 'cuda:0'
        else:
            self.device_pt1 = self.device_pt2 = 'cpu'
        self.num_cam = dataset.num_cam
        self.img_shape, self.reducedgrid_shape = dataset.img_shape, dataset.reducedgrid_shape
        imgcoord2worldgrid_matrices = self.get_imgcoord2worldgrid_matrices(dataset.base.intrinsic_matrices,
                                                                           dataset.base.extrinsic_matrices,
                                                                           dataset.base.worldgrid2worldcoord_mat)
        self.coord_map = self.create_coord_map(self.reducedgrid_shape + [1])
        # img
        self.upsample_shape = list(map(lambda x: int(x / dataset.img_reduce), self.img_shape))
        img_reduce = np.array(self.img_shape) / np.array(self.upsample_shape)
        img_zoom_mat = np.diag(np.append(img_reduce, [1]))
        # map
        map_zoom_mat = np.diag(np.append(np.ones([2]) / dataset.grid_reduce, [1]))
        # projection matrices: img feat -> map feat
        self.proj_mats = [torch.from_numpy(map_zoom_mat @ imgcoord2worldgrid_matrices[cam] @ img_zoom_mat)
                          for cam in range(self.num_cam)]
        # map feat -> img feat (used to warp BEV masks into each view for the evidence loss)
        self.proj_mats_inv = [torch.linalg.inv(M) for M in self.proj_mats]

        if arch == 'vgg11':
            base = vgg11().features
            base[-1] = nn.Sequential()
            base[-4] = nn.Sequential()
            split = 10
            self.base_pt1 = base[:split].to(self.device_pt1)
            self.base_pt2 = base[split:].to(self.device_pt2)
            out_channel = 512
        elif arch == 'resnet18':
            base = nn.Sequential(*list(resnet18(replace_stride_with_dilation=[False, True, True]).children())[:-2])
            split = 7
            self.base_pt1 = base[:split].to(self.device_pt1)
            self.base_pt2 = base[split:].to(self.device_pt2)
            out_channel = 512
        else:
            raise Exception('architecture currently support [vgg11, resnet18]')
        # 2.5cm -> 0.5m: 20x
        self.img_classifier = nn.Sequential(nn.Conv2d(out_channel, 64, 1), nn.ReLU(),
                                            nn.Conv2d(64, 2, 1, bias=False)).to(self.device_pt2)
        self.map_classifier = nn.Sequential(nn.Conv2d(out_channel * self.num_cam + 2, 512, 3, padding=1), nn.ReLU(),
                                            # nn.Conv2d(512, 512, 5, 1, 2), nn.ReLU(),
                                            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(),
                                            nn.Conv2d(512, 1, 3, padding=4, dilation=4, bias=False)).to(self.device_pt2)

        # Missing-annotation branches (MISSING_ANNOTATION_LOSS.md). map_classifier keeps its
        # parameter names so checkpoints trained without them still load with strict=False.
        self.mal_cfg = mal_cfg
        self.gate_head = None
        self.evidence = None
        if mal_cfg is not None:
            self.gate_head = GateHead(512).to(self.device_pt2)
            ev_device = mal_cfg.get('ev_device', 'auto')
            self.device_ev = self.device_pt1 if ev_device in (None, 'auto') else ev_device
            self.evidence = EvidenceBranch(backbone=mal_cfg.get('s_v_backbone', 'frozen_dinov2'),
                                           out_shape=self.upsample_shape,
                                           dino_name=mal_cfg.get('dino_name', 'dinov2_vits14'),
                                           dino_input=mal_cfg.get('dino_input', (504, 896))).to(self.device_ev)
            print(f'[mal] devices: base_pt1 {self.device_pt1} | base_pt2 + map_classifier + gate {self.device_pt2} '
                  f'| evidence branch {self.device_ev}')
            self.consensus_agg = mal_cfg.get('consensus_agg', 'median')
            self.min_visible_views = mal_cfg.get('min_visible_views', 2)
            self.register_buffer('visible_masks', compute_visibility_masks(dataset), persistent=False)
            n_vis = self.visible_masks.sum(0).float()
            print(f'[mal] visibility: mean {n_vis.mean():.2f} views / cell, '
                  f'{(n_vis >= self.min_visible_views).float().mean() * 100:.1f}% of cells have >= '
                  f'{self.min_visible_views} views')
        pass

    def aux_parameters(self):
        """Trainable params of the evidence branch + gate head. Both sit on graphs cut off from the
        main branch (own backbone / detached trunk), so they get their own (Adam) optimizer."""
        if self.evidence is None:
            return []
        return self.evidence.trainable_parameters() + list(self.gate_head.parameters())

    def main_parameters(self):
        """Everything the original MVDet SGD optimizer trains: all params except the evidence branch
        (trainable or frozen) and the gate head."""
        skip = set(id(p) for p in self.aux_parameters())
        if self.evidence is not None:
            skip |= set(id(p) for p in self.evidence.parameters())
        return [p for p in self.parameters() if id(p) not in skip]

    def project_bev_to_views(self, bev_map, view_shape=None):
        return project_bev_to_views(bev_map.to(self.device_pt2), self.proj_mats_inv,
                                    self.upsample_shape if view_shape is None else list(view_shape))

    def forward(self, imgs, visualize=False):
        B, N, C, H, W = imgs.shape
        assert N == self.num_cam
        world_features = []
        imgs_result = []
        s_logits = []
        for cam in range(self.num_cam):
            img_feature = self.base_pt1(imgs[:, cam].to(self.device_pt1))
            img_feature = self.base_pt2(img_feature.to(self.device_pt2))
            img_feature = F.interpolate(img_feature, self.upsample_shape, mode='bilinear')
            img_res = self.img_classifier(img_feature.to(self.device_pt2))
            imgs_result.append(img_res)
            proj_mat = self.proj_mats[cam].repeat([B, 1, 1]).float().to(self.device_pt2)
            world_feature = warp_perspective(img_feature.to(self.device_pt2), proj_mat, self.reducedgrid_shape)
            if visualize:
                plt.imshow(torch.norm(img_feature[0].detach(), dim=0).cpu().numpy())
                plt.show()
                plt.imshow(torch.norm(world_feature[0].detach(), dim=0).cpu().numpy())
                plt.show()
            world_features.append(world_feature.to(self.device_pt2))
            if self.evidence is not None:
                # independent branch: raw image in, evidence logits out; nothing shared with base_pt*.
                # Runs on device_ev (cuda:1 with 2 GPUs) and overlaps with the main branch on cuda:0.
                s_logits.append(self.evidence(imgs[:, cam].to(self.device_ev)))

        world_features = torch.cat(world_features + [self.coord_map.repeat([B, 1, 1, 1]).to(self.device_pt2)], dim=1)
        if visualize:
            plt.imshow(torch.norm(world_features[0].detach(), dim=0).cpu().numpy())
            plt.show()
        map_trunk = self.map_classifier[:-1](world_features.to(self.device_pt2))
        map_result = self.map_classifier[-1](map_trunk)
        map_result = F.interpolate(map_result, self.reducedgrid_shape, mode='bilinear')

        if visualize:
            plt.imshow(torch.norm(map_result[0].detach(), dim=0).cpu().numpy())
            plt.show()
        if self.evidence is None:
            return map_result, imgs_result

        q = self.gate_head(map_trunk, self.reducedgrid_shape)
        # same homographies as the features (proj_mats reuse), applied to the evidence probabilities;
        # consensus lives on device_pt2 next to q (L_q = (q - c)^2). s_logits stay on device_ev: the
        # evidence loss L_s is computed there and its masks are moved over inside the loss.
        s_bev = project_views_to_bev([torch.sigmoid(s.detach()).to(self.device_pt2) for s in s_logits],
                                     self.proj_mats, self.reducedgrid_shape)  # [B, N, X, Y]
        c, n_vis = aggregate_consensus(s_bev, self.visible_masks, self.consensus_agg, self.min_visible_views)
        mal_out = {'q': q, 'c': c, 's_logits': s_logits, 's_bev': s_bev, 'n_vis': n_vis}
        return map_result, imgs_result, mal_out

    def get_imgcoord2worldgrid_matrices(self, intrinsic_matrices, extrinsic_matrices, worldgrid2worldcoord_mat):
        projection_matrices = {}
        for cam in range(self.num_cam):
            worldcoord2imgcoord_mat = intrinsic_matrices[cam] @ np.delete(extrinsic_matrices[cam], 2, 1)

            worldgrid2imgcoord_mat = worldcoord2imgcoord_mat @ worldgrid2worldcoord_mat
            imgcoord2worldgrid_mat = np.linalg.inv(worldgrid2imgcoord_mat)
            # image of shape C,H,W (C,N_row,N_col); indexed as x,y,w,h (x,y,n_col,n_row)
            # matrix of shape N_row, N_col; indexed as x,y,n_row,n_col
            permutation_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
            projection_matrices[cam] = permutation_mat @ imgcoord2worldgrid_mat
            pass
        return projection_matrices

    def create_coord_map(self, img_size, with_r=False):
        H, W, C = img_size
        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
        grid_x = torch.from_numpy(grid_x / (W - 1) * 2 - 1).float()
        grid_y = torch.from_numpy(grid_y / (H - 1) * 2 - 1).float()
        ret = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
        if with_r:
            rr = torch.sqrt(torch.pow(grid_x, 2) + torch.pow(grid_y, 2)).view([1, 1, H, W])
            ret = torch.cat([ret, rr], dim=1)
        return ret


def test():
    from multiview_detector.datasets.frameDataset import frameDataset
    from multiview_detector.datasets.Wildtrack import Wildtrack
    from multiview_detector.datasets.MultiviewX import MultiviewX
    import torchvision.transforms as T
    from torch.utils.data import DataLoader

    transform = T.Compose([T.Resize([720, 1280]),  # H,W
                           T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset = frameDataset(Wildtrack(os.path.expanduser('~/Data/Wildtrack')), transform=transform)
    dataloader = DataLoader(dataset, 1, False, num_workers=0)
    imgs, map_gt, imgs_gt, frame = next(iter(dataloader))
    model = PerspTransDetector(dataset)
    map_res, img_res = model(imgs, visualize=True)
    pass


if __name__ == '__main__':
    test()
