"""
Missing-annotation robust loss for the MVDet BEV heatmap (spec: MISSING_ANNOTATION_LOSS.md).

MVDet regresses the BEV map to a Gaussian soft target with MSE (GaussianMSE), not sigmoid+BCE, so
the spec's log-likelihood terms are written in their MSE counterparts. Both share the same
minimiser p* = q on background, and at q = 0 the map term is EXACTLY the original GaussianMSE:

    y_obs = 1 (soft_gt > 0):  L_pos  = (p - soft_gt)^2                                  (unchanged)
    y_obs = 0 (soft_gt = 0):  L_conf = (1 - q) * p^2  +  q * (p - 1)^2                  (mirror branch)

blended continuously with the Gaussian target (no threshold on soft_gt):

    q_eff = stopgrad(q) * (1 - soft_gt)
    L_map = mean[ (1 - q_eff) (p - soft_gt)^2 + q_eff (p - 1)^2 ]

so positive labels (soft_gt -> 1) are never doubted. q is detached inside L_map by default: the
gate is learned ONLY from the independent multi-view evidence through
    L_q = mean[ (q - stopgrad(c))^2 ]
(pass conf_grad_q=True to let L_map also pull on q -- ablation only; it re-opens the
self-judging loop the spec forbids).

Evidence supervision (independent branch, never touches y_obs = 0 cells near a label):
    L_s = mean_v [ sum(w_pos * BCE(s_v, 1)) / sum(w_pos)  +  sum(m_neg * BCE(s_v, 0)) / sum(m_neg) ]
    w_pos  = Gaussian foot target of the KEPT annotations in view v (MVDet's own per-view foot GT)
    m_neg  = pixels whose ground point is >= min_dist_easy_neg BEV cells from every kept
             annotation (BEV "far" mask warped into view v with the inverse homography), and not
             inside a foot-target window. Everything in between is the confusion zone: ignored.
Optional prior: L_prior = ( mean_bg(q) - pi )^2 with mean_bg weighted by (1 - soft_gt).
"""
import torch
from torch import nn
import torch.nn.functional as F


class MissingAnnotationLoss(nn.Module):
    def __init__(self, lambda_q=1.0, lambda_s=1.0, lambda_prior=0.0, pi_prior=0.0,
                 min_dist_easy_neg=10, conf_grad_q=False, project_bev_to_views=None):
        super().__init__()
        self.lambda_q = lambda_q
        self.lambda_s = lambda_s
        self.lambda_prior = lambda_prior
        self.pi_prior = pi_prior
        self.min_dist_easy_neg = int(min_dist_easy_neg)
        self.conf_grad_q = conf_grad_q
        # bound model method: BEV tensor [B, C, X, Y] -> list of per-view [B, C, h, w]
        self.project_bev_to_views = project_bev_to_views
        d = self.min_dist_easy_neg
        yy, xx = torch.meshgrid(torch.arange(-d, d + 1), torch.arange(-d, d + 1), indexing='ij')
        disk = ((xx ** 2 + yy ** 2) <= d ** 2).float()
        self.register_buffer('disk_kernel', disk[None, None], persistent=False)  # Euclidean radius d

    # same name / behaviour as GaussianMSE so trainer.test's visualisation keeps working
    def _traget_transform(self, x, target, kernel):
        target = F.adaptive_max_pool2d(target, x.shape[2:])
        with torch.no_grad():
            target = F.conv2d(target, kernel.float().to(target.device), padding=int((kernel.shape[-1] - 1) / 2))
        return target

    @staticmethod
    def bg_weighted_mean(values, soft_gt):
        """mean of `values` over the y_obs = 0 region, weighted by (1 - soft_gt) -- no threshold."""
        w = 1 - soft_gt
        return (values * w).sum() / w.sum().clamp_min(1e-6)

    def far_mask(self, map_gt):
        """[B, 1, X, Y] float: 1 where the cell is farther than min_dist_easy_neg from every label."""
        with torch.no_grad():
            near = F.conv2d(map_gt.float(), self.disk_kernel.to(map_gt.device), padding=self.min_dist_easy_neg) > 0
            return (~near).float()

    def evidence_loss(self, s_logits, imgs_gt, img_kernel, map_gt):
        """L_s over views. s_logits: list of [B, 1, h, w]; imgs_gt: list of [B, 2, H, W] (head, foot)."""
        far_views = self.project_bev_to_views(self.far_mask(map_gt), s_logits[0].shape[-2:])
        foot_kernel = img_kernel[1:2, 1:2]
        l_pos, l_neg = 0.0, 0.0
        for s, img_gt, far_v in zip(s_logits, imgs_gt, far_views):
            foot_gt = img_gt[:, 1:2].to(s.device)
            w_pos = self._traget_transform(s, foot_gt, foot_kernel)  # Gaussian around KEPT feet
            with torch.no_grad():
                m_neg = ((far_v.to(s.device) > 0.5) & (w_pos == 0)).float()
            bce_pos = F.binary_cross_entropy_with_logits(s, torch.ones_like(s), reduction='none')
            bce_neg = F.binary_cross_entropy_with_logits(s, torch.zeros_like(s), reduction='none')
            l_pos = l_pos + (w_pos * bce_pos).sum() / w_pos.sum().clamp_min(1.0)
            l_neg = l_neg + (m_neg * bce_neg).sum() / m_neg.sum().clamp_min(1.0)
        n = len(s_logits)
        return l_pos / n, l_neg / n

    def forward(self, map_res, map_gt, map_kernel, imgs_gt, img_kernel, mal_out, active=True):
        """
        Returns a dict of the separate terms plus 'total' = L_map + lambda_q L_q + lambda_s L_s
        (+ lambda_prior L_prior). The per-view head/foot loss of MVDet is added by the trainer.
        active=False (warm-up / test): L_map is the plain GaussianMSE, L_q and L_prior are 0;
        L_s is still returned so the evidence branch trains from epoch 1.
        """
        soft_gt = self._traget_transform(map_res, map_gt, map_kernel)
        q, c = mal_out['q'], mal_out['c']
        zero = map_res.new_zeros(())

        if active:
            q_used = q if self.conf_grad_q else q.detach()
            q_eff = q_used * (1 - soft_gt)
            l_map = ((1 - q_eff) * (map_res - soft_gt) ** 2 + q_eff * (map_res - 1) ** 2).mean()
            l_q = ((q - c.detach()) ** 2).mean()
            l_prior = (self.bg_weighted_mean(q, soft_gt) - self.pi_prior) ** 2 if self.lambda_prior > 0 else zero
        else:
            l_map = F.mse_loss(map_res, soft_gt)
            l_q, l_prior = zero, zero

        l_s_pos, l_s_neg = self.evidence_loss(mal_out['s_logits'], imgs_gt, img_kernel, map_gt)
        # the evidence branch may live on another GPU: bring its loss next to L_map before summing
        l_s_pos, l_s_neg = l_s_pos.to(map_res.device), l_s_neg.to(map_res.device)
        l_s = l_s_pos + l_s_neg

        total = l_map + self.lambda_q * l_q + self.lambda_s * l_s + self.lambda_prior * l_prior
        with torch.no_grad():
            stats = {
                'q_bg_mean': self.bg_weighted_mean(q, soft_gt).item(),
                'c_bg_mean': self.bg_weighted_mean(c, soft_gt).item(),
                'c_mean': c.mean().item(),
                'q_hist': torch.histc(q.float(), bins=10, min=0.0, max=1.0).cpu(),
            }
        return {'total': total, 'map': l_map, 'q': l_q, 's': l_s, 's_pos': l_s_pos, 's_neg': l_s_neg,
                'prior': l_prior, 'stats': stats}
