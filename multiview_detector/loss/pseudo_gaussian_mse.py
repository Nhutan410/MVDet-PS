import math

import torch
import torch.nn.functional as F
from torch import nn

VARIANTS = ('gauss', 'point', 'maxval', 'mil')


def disk_offsets(r, device):
    """integer (drow, dcol) offsets with drow^2 + dcol^2 <= r^2, (D, 2)"""
    R = int(math.ceil(r))
    d = torch.arange(-R, R + 1, device=device)
    off = torch.stack(torch.meshgrid(d, d, indexing='ij'), -1).reshape(-1, 2)
    return off[(off.float() ** 2).sum(1) <= r ** 2 + 1e-6]


def sample_in_disk(n, r, device):
    theta = torch.rand(n, device=device) * 2 * math.pi
    rad = r * torch.sqrt(torch.rand(n, device=device))
    return torch.stack([rad * torch.cos(theta), rad * torch.sin(theta)], -1)


class PseudoGaussianMSE(nn.Module):
    """
    GaussianMSE + pseudo-labelled people (off-the-shelf 2D detector -> ground plane).

    Kept (real) labels are handled exactly like GaussianMSE. Each pseudo point k (map row/col, confidence
    alpha_k) is uncertain in LOCATION, not in value, so it gets a positive target near -- not exactly at --
    its projected position:

      gauss  (guide variant B)  Gaussian target (same kernel as GT) centred at a random point of the disk of
                                radius r around the pseudo point, pixel weight lambda * alpha.
                                r = 0, lambda * alpha = 1, r_ignore = 0  ==  GaussianMSE with the pseudo points
                                added to map_gt (exactly; the unit test pins it).
      point  (guide variant A, "alpha * l(P(i+di, j+dj), 1)")  a single pixel at a random point of the disk
                                is pushed to 1.
      maxval (guide variant C)  alpha * (max_{disk r} P - 1)^2: only the currently highest pixel inside the disk
                                is pushed to 1 (gradient of the max).
      mil    (C with the GT shape)  Gaussian target centred at the argmax of P inside the disk.

    Normalisation: everything is summed and divided by H * W, like the mean of GaussianMSE. A point target
    carries the weight sum(kernel^2) (the squared mass of one GT Gaussian), so that one pseudo person weighs
    as much as one kept person at initialisation (P ~ 0) in every variant. Without this, a mean over the N
    pseudo points is ~1e3 x larger per pixel than the GT term and dominates it.

    Background around pseudo points: pixels within r_ignore of a pseudo point that are not on a kept label get
    weight 0 (someone is probably there, do not push them to 0); the pseudo target pixels themselves are
    never counted as background. self.lam (set per epoch by the trainer) scales only the pseudo term.

    Ignore-only points (alpha < 0 rows, the uncertain tier): no positive target at all, only a weight-0 disk of
    radius r_ignore_only (default: the Gaussian support + location error) -- the model is neither taught that a
    person is there nor punished for predicting one.

    Optional background recalibration (BRL, MSE form of BRLFocalLoss_v2; off when brl_beta = 0): among the pixels
    that are still trained as background (weight > 0, not a pseudo target, soft GT of the kept labels < brl_pos_thr,
    optionally outside a brl_border band of output cells), those the model predicts >= brl_conf_thr (detached) are
    "confuse" -- possible unlabelled people -- and get brl_beta * (P - 1)^2 instead of (P - 0)^2.
    """

    def __init__(self, variant='gauss', r=0.0, r_ignore=0.0, bg_eps=1e-2, r_ignore_only=10.0,
                 brl_beta=0.0, brl_conf_thr=0.3, brl_pos_thr=0.1, brl_border=0.0):
        super().__init__()
        assert variant in VARIANTS, variant
        self.variant, self.r, self.r_ignore, self.bg_eps = variant, float(r), float(r_ignore), bg_eps
        self.r_ignore_only = float(r_ignore_only)
        self.brl_beta, self.brl_conf_thr, self.brl_pos_thr = float(brl_beta), float(brl_conf_thr), float(brl_pos_thr)
        self.brl_border = int(brl_border)
        self.lam = 1.0

    def _traget_transform(self, x, target, kernel):
        target = F.adaptive_max_pool2d(target, x.shape[2:])
        with torch.no_grad():
            target = F.conv2d(target, kernel.float().to(target.device), padding=int((kernel.shape[-1] - 1) / 2))
        return target

    def _gather(self, P, anchor, off):
        """values of P at anchor + off for every point, (N, D); out-of-map -> -inf. Also returns the indices."""
        H, W = P.shape
        idx = anchor[:, None, :] + off[None]
        valid = (idx[..., 0] >= 0) & (idx[..., 0] < H) & (idx[..., 1] >= 0) & (idx[..., 1] < W)
        idx_c = torch.stack([idx[..., 0].clamp(0, H - 1), idx[..., 1].clamp(0, W - 1)], -1)
        vals = P[idx_c[..., 0], idx_c[..., 1]].masked_fill(~valid, float('-inf'))
        return vals, idx_c, valid

    def _confuse(self, P, S, w, exclude=None):
        """BRL confuse mask (None when BRL is off); assignment uses the detached prediction"""
        if self.brl_beta <= 0:
            return None
        m = (S < self.brl_pos_thr) & (w > 0) & (P.detach() >= self.brl_conf_thr)
        if exclude is not None:
            m = m & ~exclude
        if self.brl_border > 0:
            b, (H, W) = self.brl_border, P.shape
            inner = torch.zeros_like(m)
            inner[b:H - b, b:W - b] = True
            m = m & inner
        return m

    def _bg_split(self, P, w, sq, C):
        """(background term, BRL term) sums; C = confuse mask or None"""
        if C is None:
            return (w * sq).sum(), P.new_zeros(())
        return (w * sq * ~C).sum(), self.brl_beta * (C * (P - 1) ** 2).sum()

    @staticmethod
    def _anchor(pos, H, W):
        return torch.stack([pos[:, 0].floor().long().clamp(0, H - 1), pos[:, 1].floor().long().clamp(0, W - 1)], -1)

    def _centres(self, P, pos, anchor):
        H, W = P.shape
        if self.variant in ('gauss', 'point'):
            if self.training and self.r > 0:
                c = torch.floor(pos + sample_in_disk(len(pos), self.r, pos.device)).long()
            else:
                c = anchor
        else:  # maxval / mil: best pixel of the current prediction inside the disk (no gradient through argmax)
            vals, idx, _ = self._gather(P.detach(), anchor, disk_offsets(self.r, P.device))
            c = idx[torch.arange(len(anchor), device=P.device), vals.argmax(1)]
        return torch.stack([c[:, 0].clamp(0, H - 1), c[:, 1].clamp(0, W - 1)], -1)

    def forward(self, x, target, kernel, pseudo=None):
        """x, target: (B, 1, H, W); pseudo: (B, N, 3) [row, col, alpha] on the output map (alpha = 0 rows are
        padding, alpha < 0 rows are ignore-only) or None. Returns (loss, stats dict or None)."""
        soft_gt = self._traget_transform(x, target, kernel)
        if pseudo is None:
            return F.mse_loss(x, soft_gt), None
        B, _, H, W = x.shape
        kernel = kernel.float().to(x.device)
        pad = int((kernel.shape[-1] - 1) / 2)
        kernel_mass = (kernel ** 2).sum()
        total = x.new_zeros(())
        st = {'l_gt': 0., 'l_ps': 0., 'n_ps': 0., 'n_ign': 0., 'p_at_ps': 0., 'alpha_mean': 0., 'ignored_frac': 0.,
              'l_brl': 0., 'confuse_frac': 0.}
        n_with_ps, n_frac = 0, 0
        for b in range(B):
            P, S = x[b, 0], soft_gt[b, 0]
            rows = pseudo[b].to(x.device).float().reshape(-1, 3)
            pts, ign_pts = rows[rows[:, 2] > 0], rows[rows[:, 2] < 0]
            gt_region = S > self.bg_eps
            w = torch.ones_like(S)
            ign = torch.zeros_like(gt_region)
            if len(ign_pts) and self.r_ignore_only > 0:
                _, idx, valid = self._gather(S, self._anchor(ign_pts[:, :2], H, W), disk_offsets(self.r_ignore_only, x.device))
                ign[idx[..., 0][valid], idx[..., 1][valid]] = True
                st['n_ign'] += len(ign_pts)
            pos, alpha = pts[:, :2], pts[:, 2]
            anchor = self._anchor(pos, H, W)
            if len(pts) and self.r_ignore > 0:
                _, idx, valid = self._gather(S, anchor, disk_offsets(self.r_ignore, x.device))
                ign[idx[..., 0][valid], idx[..., 1][valid]] = True
            w = w.masked_fill(ign & ~gt_region, 0.)
            if len(pts) == 0:
                C = self._confuse(P, S, w)
                l_gt, l_brl = self._bg_split(P, w, (P - S) ** 2, C)
                l_gt, l_brl = l_gt / (H * W), l_brl / (H * W)
                total = total + l_gt + l_brl
                st['l_gt'] += l_gt.item()
                st['l_brl'] += l_brl.item()
                st['confuse_frac'] += C.float().mean().item() if C is not None else 0.
                if len(ign_pts):
                    st['ignored_frac'] += (w == 0).float().mean().item()
                    n_frac += 1
                continue
            c = self._centres(P, pos, anchor)

            if self.variant in ('gauss', 'mil'):
                delta = torch.zeros_like(S).index_put_((c[:, 0], c[:, 1]), torch.ones_like(alpha), accumulate=True)
                a_delta = torch.zeros_like(S).index_put_((c[:, 0], c[:, 1]), alpha, accumulate=True)
                T = F.conv2d(torch.stack([delta, a_delta])[:, None], kernel, padding=pad)[:, 0]
                T, A = T[0], T[1] / T[0].clamp_min(1e-12)  # A = alpha of the pseudo point(s) covering the pixel
                ps_region = (T > self.bg_eps) & ~gt_region
                sq = (P - (S + T)) ** 2
                C = self._confuse(P, S, w, exclude=ps_region)
                l_gt, l_brl = self._bg_split(P, w * ~ps_region, sq, C)
                l_ps = (A * sq * ps_region).sum() / (H * W)
            else:  # point / maxval
                w = w.index_put((c[:, 0], c[:, 1]), torch.zeros_like(alpha))
                C = self._confuse(P, S, w)
                l_gt, l_brl = self._bg_split(P, w, (P - S) ** 2, C)
                l_ps = (alpha * kernel_mass * (P[c[:, 0], c[:, 1]] - 1) ** 2).sum() / (H * W)
            l_gt, l_brl = l_gt / (H * W), l_brl / (H * W)
            total = total + l_gt + self.lam * l_ps + l_brl
            st['l_brl'] += l_brl.item()
            st['confuse_frac'] += C.float().mean().item() if C is not None else 0.
            n_with_ps += 1
            st['l_gt'] += l_gt.item()
            st['l_ps'] += l_ps.item()
            st['n_ps'] += len(pts)
            st['p_at_ps'] += P.detach()[c[:, 0], c[:, 1]].mean().item()
            st['alpha_mean'] += alpha.mean().item()
            st['ignored_frac'] += (w == 0).float().mean().item()
            n_frac += 1
        for k in ('l_gt', 'l_ps', 'n_ps', 'n_ign', 'l_brl', 'confuse_frac'):
            st[k] /= B
        for k in ('p_at_ps', 'alpha_mean'):
            st[k] = st[k] / n_with_ps if n_with_ps else float('nan')
        st['ignored_frac'] = st['ignored_frac'] / n_frac if n_frac else float('nan')
        st['lam'] = self.lam
        return total / B, st
