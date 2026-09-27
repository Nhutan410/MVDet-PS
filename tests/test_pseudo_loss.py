"""
CPU unit tests for the pseudo-label loss (no dataset, no GPU needed):

    python -m pytest tests/test_pseudo_loss.py -q      # or:  python tests/test_pseudo_loss.py

They pin down the design of PSEUDO_LABEL.md:
  * without pseudo points the loss is exactly GaussianMSE,
  * gauss, r = 0, alpha * lambda = 1, no ignore == GaussianMSE with the pseudo points added to map_gt,
  * one pseudo person weighs as much as one kept person at P = 0 (every variant),
  * jitter only in train mode and always inside the r-disk,
  * maxval / mil follow the argmax of P inside the disk, point / maxval touch a single pixel,
  * ignore disk: no gradient on background near a pseudo point, kept labels untouched,
  * lambda = 0 removes every pseudo gradient.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multiview_detector.loss.gaussian_mse import GaussianMSE
from multiview_detector.loss.pseudo_gaussian_mse import PseudoGaussianMSE, disk_offsets

torch.manual_seed(0)
H, W = 30, 40


def make_kernel(size=4, sigma2=5.0):
    d = torch.arange(-size, size + 1).float()
    k = torch.exp(-(d[:, None] ** 2 + d[None] ** 2) / (2 * sigma2))
    return (k / k.max())[None, None]


KERNEL = make_kernel()


def gt_map(points):
    m = torch.zeros(1, 1, H, W)
    for r, c in points:
        m[0, 0, r, c] = 1
    return m


def pseudo_tensor(rows):
    return torch.tensor(rows, dtype=torch.float32).reshape(1, -1, 3)


def test_no_pseudo_equals_gaussian_mse():
    x = torch.rand(1, 1, H, W)
    gt = gt_map([(5, 5), (20, 30)])
    ref = GaussianMSE()(x, gt, KERNEL)
    for variant in ('gauss', 'point', 'maxval', 'mil'):
        crit = PseudoGaussianMSE(variant, r=3, r_ignore=4)
        loss, st = crit(x, gt, KERNEL, None)
        assert torch.allclose(loss, ref) and st is None
        loss, st = crit(x, gt, KERNEL, torch.zeros(1, 0, 3))
        assert torch.allclose(loss, ref), variant


def test_gauss_r0_equals_pseudo_as_gt():
    x = torch.rand(1, 1, H, W, requires_grad=True)
    gt = gt_map([(5, 5)])
    ps = pseudo_tensor([[15.4, 20.7, 1.0], [25.0, 8.2, 1.0]])  # floored to (15, 20), (25, 8)
    crit = PseudoGaussianMSE('gauss', r=0, r_ignore=0)
    crit.lam = 1.0
    loss, _ = crit(x, gt, KERNEL, ps)
    ref = GaussianMSE()(x, gt_map([(5, 5), (15, 20), (25, 8)]), KERNEL)
    assert torch.allclose(loss, ref, atol=1e-7), (loss, ref)
    g1, = torch.autograd.grad(loss, x)
    g2, = torch.autograd.grad(ref, x)
    assert torch.allclose(g1, g2, atol=1e-8)


def test_pseudo_person_weighs_like_gt_person_at_init():
    x = torch.zeros(1, 1, H, W)
    empty = gt_map([])
    gt_one = GaussianMSE()(x, gt_map([(15, 20)]), KERNEL)
    for variant in ('gauss', 'point', 'maxval', 'mil'):
        crit = PseudoGaussianMSE(variant, r=0, r_ignore=0)
        _, st = crit(x, empty, KERNEL, pseudo_tensor([[15, 20, 1.0]]))
        assert abs(st['l_ps'] - gt_one.item()) / gt_one.item() < 1e-5, (variant, st['l_ps'], gt_one.item())


def test_alpha_and_lambda_scale_pseudo_term():
    x = torch.rand(1, 1, H, W)
    empty = gt_map([])
    crit = PseudoGaussianMSE('point', r=0, r_ignore=0)
    _, st1 = crit(x, empty, KERNEL, pseudo_tensor([[15, 20, 1.0]]))
    _, st2 = crit(x, empty, KERNEL, pseudo_tensor([[15, 20, 0.5]]))
    assert abs(st2['l_ps'] - 0.5 * st1['l_ps']) < 1e-9
    x = torch.rand(1, 1, H, W, requires_grad=True)
    for variant in ('gauss', 'point', 'maxval', 'mil'):
        crit = PseudoGaussianMSE(variant, r=0, r_ignore=0)
        crit.lam = 0.0
        loss, _ = crit(x, empty, KERNEL, pseudo_tensor([[15, 20, 1.0]]))
        g, = torch.autograd.grad(loss, x)
        # lambda = 0: the pseudo pixels get no gradient at all (they are not background either)
        assert g[0, 0, 15, 20] == 0, variant


def test_jitter_train_only_and_inside_disk():
    x = torch.zeros(1, 1, H, W)
    empty = gt_map([])
    ps = pseudo_tensor([[15.0, 20.0, 1.0]])
    crit = PseudoGaussianMSE('point', r=3, r_ignore=0)
    crit.eval()
    for _ in range(20):
        c = crit._centres(x[0, 0], ps[0, :, :2], ps[0, :, :2].long())
        assert c.tolist() == [[15, 20]]
    crit.train()
    seen = set()
    for _ in range(400):
        c = crit._centres(x[0, 0], ps[0, :, :2], ps[0, :, :2].long())[0]
        dr, dc = c[0].item() - 15, c[1].item() - 20
        # continuous disk of radius r, floored: each coordinate moves by at most r (+1 from flooring)
        assert (dr + 0.5) ** 2 + (dc + 0.5) ** 2 <= (3 + 1) ** 2, (dr, dc)
        seen.add((dr, dc))
    assert len(seen) > 10


def test_maxval_and_mil_follow_argmax_in_disk():
    x = torch.full((1, 1, H, W), 0.1)
    x[0, 0, 17, 21] = 0.8  # inside r = 3 of (15, 20)
    x[0, 0, 15, 26] = 0.9  # outside the disk -> must be ignored
    x.requires_grad_(True)
    empty = gt_map([])
    ps = pseudo_tensor([[15.0, 20.0, 1.0]])
    crit = PseudoGaussianMSE('maxval', r=3, r_ignore=0)
    loss, st = crit(x, empty, KERNEL, ps)
    g, = torch.autograd.grad(loss, x)
    assert g[0, 0, 17, 21] < 0  # pushed up
    assert abs(st['p_at_ps'] - 0.8) < 1e-6
    mil = PseudoGaussianMSE('mil', r=3, r_ignore=0)
    c = mil._centres(x[0, 0].detach(), ps[0, :, :2], ps[0, :, :2].long())
    assert c.tolist() == [[17, 21]]


def test_ignore_disk():
    x = torch.rand(1, 1, H, W, requires_grad=True)
    gt = gt_map([(15, 23)])  # a kept person right next to the pseudo point
    ps = pseudo_tensor([[15.0, 10.0, 1.0]])
    crit = PseudoGaussianMSE('point', r=0, r_ignore=4)
    loss, st = crit(x, gt, KERNEL, ps)
    g, = torch.autograd.grad(loss, x)
    assert g[0, 0, 15, 13] == 0 and g[0, 0, 12, 10] == 0  # background within 4 cells of the pseudo point
    assert g[0, 0, 15, 15] != 0  # outside the disk: still background
    assert g[0, 0, 15, 23] != 0  # kept label: untouched
    assert g[0, 0, 15, 10] != 0  # the pseudo target itself
    n_disk = len(disk_offsets(4, 'cpu'))
    assert abs(st['ignored_frac'] - n_disk / (H * W)) < 1e-6


def test_point_touches_single_pixel():
    x = torch.rand(1, 1, H, W, requires_grad=True)
    empty = gt_map([])
    crit = PseudoGaussianMSE('point', r=0, r_ignore=0)
    crit.lam = 1.0
    _, st = crit(x, empty, KERNEL, pseudo_tensor([[15.0, 20.0, 1.0]]))
    expected = KERNEL.pow(2).sum() * (x[0, 0, 15, 20] - 1) ** 2 / (H * W)
    assert abs(st['l_ps'] - expected.item()) < 1e-7


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('PASS', name)
