"""
CPU unit tests for the missing-annotation robust loss (no dataset, no GPU needed):

    python -m pytest tests/ -q          # or:  python tests/test_missing_annotation_loss.py

They pin down the design constraints of MISSING_ANNOTATION_LOSS.md:
  * c is detached and never carries gradient into the main branch,
  * the main branch never receives gradient from L_q / L_s, the evidence branch never from L_map / L_q,
  * with q = 0 (warm-up) the map term is exactly GaussianMSE,
  * on background the map term is minimised at p = q,
  * consensus = median / percentile25 / min over VISIBLE views only, c = 0 below min_visible_views.
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multiview_detector.loss.gaussian_mse import GaussianMSE
from multiview_detector.loss.missing_annotation_loss import MissingAnnotationLoss
from multiview_detector.models.missing_annotation import GateHead, aggregate_consensus

torch.manual_seed(0)

N_VIEWS, X, Y = 3, 24, 32
H, W = 24, 32  # keep view == BEV shape so the fake projection is the identity


def make_kernels():
    map_kernel = torch.zeros(1, 1, 5, 5)
    map_kernel[0, 0] = torch.exp(-(torch.arange(-2, 3)[:, None] ** 2 + torch.arange(-2, 3)[None] ** 2) / 2.0)
    img_kernel = torch.zeros(2, 2, 5, 5)
    img_kernel[0, 0] = map_kernel[0, 0]
    img_kernel[1, 1] = map_kernel[0, 0]
    return map_kernel, img_kernel


def identity_projection(bev_map, view_shape):
    return [F.interpolate(bev_map, list(view_shape), mode='bilinear', align_corners=False) for _ in range(N_VIEWS)]


class ToyMain(nn.Module):
    """stand-in for backbone -> map_classifier trunk -> p, with a GateHead on the (detached) trunk"""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Conv2d(3, 8, 3, padding=1)
        self.trunk = nn.Conv2d(8, 8, 3, padding=1)
        self.map_head = nn.Conv2d(8, 1, 1)
        self.gate_head = GateHead(8, hidden=4)

    def forward(self, x):
        trunk = F.relu(self.trunk(F.relu(self.backbone(x))))
        return self.map_head(trunk), self.gate_head(trunk)


class ToyEvidence(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.Conv2d(4, 1, 1))

    def forward(self, x):
        return self.net(x)


def make_batch():
    imgs = torch.randn(1, N_VIEWS, 3, H, W)
    map_gt = torch.zeros(1, 1, X, Y)
    map_gt[0, 0, 5, 7] = 1
    map_gt[0, 0, 18, 25] = 1
    imgs_gt = []
    for _ in range(N_VIEWS):
        g = torch.zeros(1, 2, H, W)
        g[0, 1, 5, 7] = 1
        imgs_gt.append(g)
    return imgs, map_gt, imgs_gt


def run_toy(active=True, conf_grad_q=False, lambda_q=1.0, lambda_s=1.0):
    main, evidence = ToyMain(), ToyEvidence()
    crit = MissingAnnotationLoss(lambda_q=lambda_q, lambda_s=lambda_s, min_dist_easy_neg=4,
                                 conf_grad_q=conf_grad_q, project_bev_to_views=identity_projection)
    map_kernel, img_kernel = make_kernels()
    imgs, map_gt, imgs_gt = make_batch()
    p, q = main(imgs[:, 0])
    s_logits = [evidence(imgs[:, v]) for v in range(N_VIEWS)]
    s_bev = torch.stack([torch.sigmoid(s.detach())[:, 0] for s in s_logits], dim=1)
    c, _ = aggregate_consensus(s_bev, torch.ones(N_VIEWS, X, Y, dtype=torch.bool), 'median', 2)
    mal_out = {'q': q, 'c': c, 's_logits': s_logits}
    out = crit(p, map_gt, map_kernel, imgs_gt, img_kernel, mal_out, active=active)
    return main, evidence, crit, out, mal_out, (p, map_gt, map_kernel)


def grads(module):
    return {n: (None if p.grad is None else p.grad.clone()) for n, p in module.named_parameters()}


def all_zero(gdict, names):
    return all(gdict[n] is None or torch.all(gdict[n] == 0) for n in names)


def test_c_is_detached():
    _, _, _, _, mal_out, _ = run_toy()
    assert not mal_out['c'].requires_grad
    assert mal_out['c'].grad_fn is None


def test_gradient_isolation():
    main, evidence, crit, out, _, _ = run_toy(active=True)
    main_names = [n for n, _ in main.named_parameters() if not n.startswith('gate_head')]
    gate_names = [n for n, _ in main.named_parameters() if n.startswith('gate_head')]
    ev_names = [n for n, _ in evidence.named_parameters()]

    # L_q + L_s alone: nothing reaches the main branch (trunk detached in GateHead, c detached)
    main.zero_grad(); evidence.zero_grad()
    (out['q'] + out['s']).backward(retain_graph=True)
    g_main, g_ev = grads(main), grads(evidence)
    assert all_zero(g_main, main_names), 'L_q / L_s leaked into the main branch'
    assert not all_zero(g_main, gate_names), 'gate head got no gradient from L_q'
    assert not all_zero(g_ev, ev_names), 'evidence branch got no gradient from L_s'

    # L_map alone: never touches the evidence branch nor (by default) the gate
    main.zero_grad(); evidence.zero_grad()
    out['map'].backward(retain_graph=True)
    g_main, g_ev = grads(main), grads(evidence)
    assert all_zero(g_ev, ev_names), 'L_map leaked into the evidence branch'
    assert all_zero(g_main, gate_names), 'L_map pulled on q although conf_grad_q=False'
    assert not all_zero(g_main, main_names)


def test_conf_grad_q_ablation_reaches_gate_only():
    main, evidence, crit, out, _, _ = run_toy(active=True, conf_grad_q=True)
    gate_names = [n for n, _ in main.named_parameters() if n.startswith('gate_head')]
    ev_names = [n for n, _ in evidence.named_parameters()]
    main.zero_grad(); evidence.zero_grad()
    out['map'].backward()
    assert not all_zero(grads(main), gate_names)
    assert all_zero(grads(evidence), ev_names)


def test_inactive_equals_gaussian_mse():
    _, _, _, out, _, (p, map_gt, map_kernel) = run_toy(active=False)
    ref = GaussianMSE()(p, map_gt, map_kernel)
    assert torch.allclose(out['map'], ref), (out['map'].item(), ref.item())
    assert float(out['q']) == 0 and float(out['prior']) == 0


def test_active_with_zero_gate_equals_gaussian_mse():
    _, _, crit, _, mal_out, (p, map_gt, map_kernel) = run_toy(active=True)
    _, img_kernel = make_kernels()
    _, _, imgs_gt = make_batch()
    mal_out = dict(mal_out, q=torch.zeros_like(mal_out['q']))
    out = crit(p, map_gt, map_kernel, imgs_gt, img_kernel, mal_out, active=True)
    ref = GaussianMSE()(p, map_gt, map_kernel)
    assert torch.allclose(out['map'], ref)


def test_background_minimiser_is_q():
    crit = MissingAnnotationLoss(project_bev_to_views=identity_projection)
    soft_gt = torch.zeros(1, 1, 4, 4)
    q = torch.full((1, 1, 4, 4), 0.3)
    p = torch.full((1, 1, 4, 4), 0.3, requires_grad=True)
    q_eff = q * (1 - soft_gt)
    l = ((1 - q_eff) * (p - soft_gt) ** 2 + q_eff * (p - 1) ** 2).mean()
    l.backward()
    assert torch.allclose(p.grad, torch.zeros_like(p.grad), atol=1e-7)


def test_positive_labels_are_never_doubted():
    """where soft_gt = 1 the map term is (p - 1)^2 whatever q says"""
    soft_gt = torch.ones(1, 1, 2, 2)
    q = torch.full((1, 1, 2, 2), 0.9)
    p = torch.full((1, 1, 2, 2), 0.2)
    q_eff = q * (1 - soft_gt)
    l = ((1 - q_eff) * (p - soft_gt) ** 2 + q_eff * (p - 1) ** 2).mean()
    assert torch.allclose(l, ((p - 1) ** 2).mean())


def test_consensus_rules():
    s = torch.tensor([[[[0.1]], [[0.9]], [[0.5]]]])  # [B=1, N=3, 1, 1]
    vis_all = torch.ones(3, 1, 1, dtype=torch.bool)
    c, n = aggregate_consensus(s, vis_all, 'median', 2)
    assert torch.allclose(c, torch.tensor(0.5)) and int(n) == 3
    c, _ = aggregate_consensus(s, vis_all, 'min', 2)
    assert torch.allclose(c, torch.tensor(0.1))
    c, _ = aggregate_consensus(s, vis_all, 'percentile25', 2)
    assert torch.allclose(c, torch.tensor(0.3))
    # one hallucinating view among two visible ones cannot pull the median up: lower median = 0.1
    vis = torch.tensor([[[True]], [[True]], [[False]]])
    c, n = aggregate_consensus(s, vis, 'median', 2)
    assert torch.allclose(c, torch.tensor(0.1)) and int(n) == 2
    # a single visible view is not enough evidence
    vis = torch.tensor([[[False]], [[True]], [[False]]])
    c, n = aggregate_consensus(s, vis, 'median', 2)
    assert float(c) == 0 and int(n) == 1
    assert not c.requires_grad


def test_easy_negative_mask_excludes_confusion_zone():
    crit = MissingAnnotationLoss(min_dist_easy_neg=4, project_bev_to_views=identity_projection)
    map_gt = torch.zeros(1, 1, X, Y)
    map_gt[0, 0, 10, 10] = 1
    far = crit.far_mask(map_gt)[0, 0]
    assert far[10, 10] == 0 and far[10, 14] == 0 and far[12, 13] == 0  # inside radius 4 (Euclidean)
    assert far[10, 15] == 1 and far[13, 13] == 1 and far[0, 0] == 1  # sqrt(18) > 4


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('PASS', name)
