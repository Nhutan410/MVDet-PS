import time
import json
import torch
import os
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from multiview_detector.evaluation.evaluate import evaluate
from multiview_detector.utils.nms import nms
from multiview_detector.utils.meters import AverageMeter
from multiview_detector.utils.image_utils import add_heatmap_to_image
from multiview_detector.loss.gaussian_mse import GaussianMSE
from multiview_detector.loss.pseudo_gaussian_mse import PseudoGaussianMSE


class BaseTrainer(object):
    def __init__(self):
        super(BaseTrainer, self).__init__()


class PerspectiveTrainer(BaseTrainer):
    PS_KEYS = ('l_gt', 'l_ps', 'l_view', 'n_ps', 'n_ign', 'p_at_ps', 'alpha_mean', 'ignored_frac')

    def __init__(self, model, criterion, logdir, denormalize, cls_thres=0.4, alpha=1.0, ps_schedule=(1.0, 0, 1)):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.cls_thres = cls_thres
        self.logdir = logdir
        self.denormalize = denormalize
        self.alpha = alpha
        # pseudo labels: the BEV map loss is PseudoGaussianMSE, the per-view head/foot loss stays GaussianMSE;
        # lambda_ps(epoch) = lam_max * min(1, (epoch - warmup) / ramp), 0 during the warm-up
        self.is_pseudo = isinstance(criterion, PseudoGaussianMSE)
        self.img_criterion = GaussianMSE() if self.is_pseudo else criterion
        self.ps_lam_max, self.ps_warmup, self.ps_ramp = ps_schedule

    def lambda_ps(self, epoch):
        """epoch is 1-based"""
        e = epoch - 1
        if e < self.ps_warmup:
            return 0.0
        return self.ps_lam_max * min(1.0, (e - self.ps_warmup + 1) / max(self.ps_ramp, 1))

    def _loss(self, map_res, imgs_res, map_gt, imgs_gt, dataset, pseudo=None):
        """original MVDet loss; with pseudo labels also returns the per-term stats (else None)"""
        img_loss = 0
        for img_res, img_gt in zip(imgs_res, imgs_gt):
            img_loss += self.img_criterion(img_res, img_gt.to(img_res.device), dataset.img_kernel)
        img_loss = img_loss / len(imgs_gt) * self.alpha
        if not self.is_pseudo:
            return self.criterion(map_res, map_gt.to(map_res.device), dataset.map_kernel) + img_loss, None
        map_loss, st = self.criterion(map_res, map_gt.to(map_res.device), dataset.map_kernel, pseudo)
        if st is not None:
            st['l_view'] = img_loss.item()
        return map_loss + img_loss, st

    def _ps_str(self, ps_s):
        if not self.is_pseudo or ps_s['l_gt'].count == 0:
            return ''
        return (', lam_ps: {:.2f}, L_gt: {:.5f}, L_ps: {:.5f}, L_view: {:.5f}, n_ps: {:.1f}, n_ign: {:.1f}, '
                'P@ps: {:.3f}, alpha: {:.3f}, ignored: {:.4f}'.format(self.criterion.lam,
                                                                      *(ps_s[k].avg for k in self.PS_KEYS)))

    def train(self, epoch, data_loader, optimizer, log_interval=100, cyclic_scheduler=None):
        self.model.train()
        losses = 0
        precision_s, recall_s = AverageMeter(), AverageMeter()
        ps_s = {k: AverageMeter() for k in self.PS_KEYS}
        if self.is_pseudo:
            self.criterion.train()  # jitter on
            self.criterion.lam = self.lambda_ps(epoch)
            print(f'[pseudo] epoch {epoch}: lambda_ps = {self.criterion.lam:.3f}')
        t0 = time.time()
        t_b = time.time()
        t_forward = 0
        t_backward = 0
        for batch_idx, batch in enumerate(data_loader):
            data, map_gt, imgs_gt = batch[:3]
            pseudo = batch[4] if len(batch) > 4 else None
            optimizer.zero_grad()
            map_res, imgs_res = self.model(data)
            t_f = time.time()
            t_forward += t_f - t_b
            loss, st = self._loss(map_res, imgs_res, map_gt, imgs_gt, data_loader.dataset, pseudo)
            loss.backward()
            optimizer.step()
            losses += loss.item()
            if st is not None:
                for k in self.PS_KEYS:
                    if k in st and st[k] == st[k]:  # skip NaN (frame without pseudo points)
                        ps_s[k].update(st[k])
            pred = (map_res > self.cls_thres).int().to(map_gt.device)
            true_positive = (pred.eq(map_gt) * pred.eq(1)).sum().item()
            false_positive = pred.sum().item() - true_positive
            false_negative = map_gt.sum().item() - true_positive
            precision = true_positive / (true_positive + false_positive + 1e-4)
            recall = true_positive / (true_positive + false_negative + 1e-4)
            precision_s.update(precision)
            recall_s.update(recall)

            t_b = time.time()
            t_backward += t_b - t_f

            if cyclic_scheduler is not None:
                if isinstance(cyclic_scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    cyclic_scheduler.step(epoch - 1 + batch_idx / len(data_loader))
                elif isinstance(cyclic_scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    cyclic_scheduler.step()
            if (batch_idx + 1) % log_interval == 0:
                # print(cyclic_scheduler.last_epoch, optimizer.param_groups[0]['lr'])
                t1 = time.time()
                t_epoch = t1 - t0
                print('Train Epoch: {}, Batch:{}, Loss: {:.6f}, '
                      'prec: {:.1f}%, recall: {:.1f}%, Time: {:.1f} (f{:.3f}+b{:.3f}), maxima: {:.3f}'.format(
                    epoch, (batch_idx + 1), losses / (batch_idx + 1), precision_s.avg * 100, recall_s.avg * 100,
                    t_epoch, t_forward / batch_idx, t_backward / batch_idx, map_res.max()) + self._ps_str(ps_s))
                pass

        t1 = time.time()
        t_epoch = t1 - t0
        print('Train Epoch: {}, Batch:{}, Loss: {:.6f}, '
              'Precision: {:.1f}%, Recall: {:.1f}%, Time: {:.3f}'.format(
            epoch, len(data_loader), losses / len(data_loader), precision_s.avg * 100, recall_s.avg * 100, t_epoch)
              + self._ps_str(ps_s))
        if self.is_pseudo:
            with open(os.path.join(self.logdir, 'pseudo_stats.jsonl'), 'a') as f:
                f.write(json.dumps({'epoch': epoch, 'lam_ps': self.criterion.lam, 'loss': losses / len(data_loader),
                                    'precision': precision_s.avg * 100, 'recall': recall_s.avg * 100,
                                    **{k: m.avg for k, m in ps_s.items()}}) + '\n')

        return losses / len(data_loader), precision_s.avg * 100

    def test(self, data_loader, res_fpath=None, gt_fpath=None, visualize=False):
        self.model.eval()
        losses = 0
        precision_s, recall_s = AverageMeter(), AverageMeter()
        all_res_list = []
        t0 = time.time()
        if res_fpath is not None:
            assert gt_fpath is not None
        if self.is_pseudo:
            self.criterion.eval()  # no jitter
        for batch_idx, batch in enumerate(data_loader):
            data, map_gt, imgs_gt, frame = batch[:4]
            with torch.no_grad():
                map_res, imgs_res = self.model(data)
            if res_fpath is not None:
                map_grid_res = map_res.detach().cpu().squeeze()
                v_s = map_grid_res[map_grid_res > self.cls_thres].unsqueeze(1)
                grid_ij = (map_grid_res > self.cls_thres).nonzero()
                if data_loader.dataset.base.indexing == 'xy':
                    grid_xy = grid_ij[:, [1, 0]]
                else:
                    grid_xy = grid_ij
                all_res_list.append(torch.cat([torch.ones_like(v_s) * frame, grid_xy.float() *
                                               data_loader.dataset.grid_reduce, v_s], dim=1))

            with torch.no_grad():
                loss, _ = self._loss(map_res, imgs_res, map_gt, imgs_gt, data_loader.dataset)
            losses += loss.item()
            pred = (map_res > self.cls_thres).int().to(map_gt.device)
            true_positive = (pred.eq(map_gt) * pred.eq(1)).sum().item()
            false_positive = pred.sum().item() - true_positive
            false_negative = map_gt.sum().item() - true_positive
            precision = true_positive / (true_positive + false_positive + 1e-4)
            recall = true_positive / (true_positive + false_negative + 1e-4)
            precision_s.update(precision)
            recall_s.update(recall)

        t1 = time.time()
        t_epoch = t1 - t0

        if visualize:
            fig = plt.figure()
            subplt0 = fig.add_subplot(211, title="output")
            subplt1 = fig.add_subplot(212, title="target")
            subplt0.imshow(map_res.cpu().detach().numpy().squeeze())
            subplt1.imshow(self.criterion._traget_transform(map_res, map_gt, data_loader.dataset.map_kernel)
                           .cpu().detach().numpy().squeeze())
            plt.savefig(os.path.join(self.logdir, 'map.jpg'))
            plt.close(fig)

            # visualizing the heatmap for per-view estimation
            heatmap0_head = imgs_res[0][0, 0].detach().cpu().numpy().squeeze()
            heatmap0_foot = imgs_res[0][0, 1].detach().cpu().numpy().squeeze()
            img0 = self.denormalize(data[0, 0]).cpu().numpy().squeeze().transpose([1, 2, 0])
            img0 = Image.fromarray((img0 * 255).astype('uint8'))
            head_cam_result = add_heatmap_to_image(heatmap0_head, img0)
            head_cam_result.save(os.path.join(self.logdir, 'cam1_head.jpg'))
            foot_cam_result = add_heatmap_to_image(heatmap0_foot, img0)
            foot_cam_result.save(os.path.join(self.logdir, 'cam1_foot.jpg'))

        moda = 0
        if res_fpath is not None:
            all_res_list = torch.cat(all_res_list, dim=0)
            np.savetxt(os.path.abspath(os.path.dirname(res_fpath)) + '/all_res.txt', all_res_list.numpy(), '%.8f')
            res_list = []
            for frame in np.unique(all_res_list[:, 0]):
                res = all_res_list[all_res_list[:, 0] == frame, :]
                positions, scores = res[:, 1:3], res[:, 3]
                ids, count = nms(positions, scores, 20, np.inf)
                res_list.append(torch.cat([torch.ones([count, 1]) * frame, positions[ids[:count], :]], dim=1))
            res_list = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
            np.savetxt(res_fpath, res_list, '%d')

            recall, precision, moda, modp = evaluate(os.path.abspath(res_fpath), os.path.abspath(gt_fpath),
                                                     data_loader.dataset.base.__name__)

            # If you want to use the unofiicial python evaluation tool for convenient purposes.
            # recall, precision, modp, moda = python_eval(os.path.abspath(res_fpath), os.path.abspath(gt_fpath),
            #                                             data_loader.dataset.base.__name__)

            print('moda: {:.1f}%, modp: {:.1f}%, precision: {:.1f}%, recall: {:.1f}%'.
                  format(moda, modp, precision, recall))

        print('Test, Loss: {:.6f}, Precision: {:.1f}%, Recall: {:.1f}, \tTime: {:.3f}'.format(
            losses / (len(data_loader) + 1), precision_s.avg * 100, recall_s.avg * 100, t_epoch))

        return losses / len(data_loader), precision_s.avg * 100, moda


class BBOXTrainer(BaseTrainer):
    def __init__(self, model, criterion, cls_thres):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.cls_thres = cls_thres

    def train(self, epoch, data_loader, optimizer, log_interval=100, cyclic_scheduler=None):
        self.model.train()
        losses = 0
        correct = 0
        miss = 0
        t0 = time.time()
        for batch_idx, (data, target, _) in enumerate(data_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = self.model(data)
            pred = torch.argmax(output, 1)
            correct += pred.eq(target).sum().item()
            miss += target.numel() - pred.eq(target).sum().item()
            loss = self.criterion(output, target)
            loss.backward()
            optimizer.step()
            losses += loss.item()
            if cyclic_scheduler is not None:
                if isinstance(cyclic_scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    cyclic_scheduler.step(epoch - 1 + batch_idx / len(data_loader))
                elif isinstance(cyclic_scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    cyclic_scheduler.step()
            if (batch_idx + 1) % log_interval == 0:
                # print(cyclic_scheduler.last_epoch, optimizer.param_groups[0]['lr'])
                t1 = time.time()
                t_epoch = t1 - t0
                print('Train Epoch: {}, Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
                    epoch, (batch_idx + 1), losses / (batch_idx + 1), 100. * correct / (correct + miss), t_epoch))

        t1 = time.time()
        t_epoch = t1 - t0
        print('Train Epoch: {}, Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
            epoch, len(data_loader), losses / len(data_loader), 100. * correct / (correct + miss), t_epoch))

        return losses / len(data_loader), correct / (correct + miss)

    def test(self, test_loader, log_interval=100, res_fpath=None):
        self.model.eval()
        losses = 0
        correct = 0
        miss = 0
        all_res_list = []
        t0 = time.time()
        for batch_idx, (data, target, (frame, pid, grid_x, grid_y)) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(data)
                output = F.softmax(output, dim=1)
            pred = torch.argmax(output, 1)
            correct += pred.eq(target).sum().item()
            miss += target.numel() - pred.eq(target).sum().item()
            loss = self.criterion(output, target)
            losses += loss.item()
            if res_fpath is not None:
                indices = output[:, 1] > self.cls_thres
                all_res_list.append(torch.stack([frame[indices].float(), grid_x[indices].float(),
                                                 grid_y[indices].float(), output[indices, 1].cpu()], dim=1))
            if (batch_idx + 1) % log_interval == 0:
                # print(cyclic_scheduler.last_epoch, optimizer.param_groups[0]['lr'])
                t1 = time.time()
                t_epoch = t1 - t0
                print('Test Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
                    (batch_idx + 1), losses / (batch_idx + 1), 100. * correct / (correct + miss), t_epoch))

        t1 = time.time()
        t_epoch = t1 - t0
        print('Test, Batch:{}, Loss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
            len(test_loader), losses / (len(test_loader) + 1), 100. * correct / (correct + miss), t_epoch))

        if res_fpath is not None:
            all_res_list = torch.cat(all_res_list, dim=0)
            np.savetxt(os.path.dirname(res_fpath) + '/all_res.txt', all_res_list.numpy(), '%.8f')
            res_list = []
            for frame in np.unique(all_res_list[:, 0]):
                res = all_res_list[all_res_list[:, 0] == frame, :]
                positions, scores = res[:, 1:3], res[:, 3]
                ids, count = nms(positions, scores, )
                res_list.append(torch.cat([torch.ones([count, 1]) * frame, positions[ids[:count], :]], dim=1))
            res_list = torch.cat(res_list, dim=0).numpy()
            np.savetxt(res_fpath, res_list, '%d')

        return losses / len(test_loader), correct / (correct + miss)
