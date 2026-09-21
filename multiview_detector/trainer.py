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
from multiview_detector.loss.missing_annotation_loss import MissingAnnotationLoss


class BaseTrainer(object):
    def __init__(self):
        super(BaseTrainer, self).__init__()


class PerspectiveTrainer(BaseTrainer):
    def __init__(self, model, criterion, logdir, denormalize, cls_thres=0.4, alpha=1.0,
                 mal_warmup_epochs=0, aux_optimizer=None, clip_grad_norm=0.0):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.cls_thres = cls_thres
        self.logdir = logdir
        self.denormalize = denormalize
        self.alpha = alpha
        # missing-annotation loss: L_conf / L_q switch on only after mal_warmup_epochs (train p alone
        # first); evidence branch s_v + gate head q have their own optimizer, s_v trains from epoch 1
        self.is_mal = isinstance(criterion, MissingAnnotationLoss)
        self.mal_warmup_epochs = mal_warmup_epochs
        self.aux_optimizer = aux_optimizer
        # 0 = off (original MVDet). Applied to the main-branch parameters only, after the NaN guard.
        self.clip_grad_norm = clip_grad_norm
        self.img_criterion = GaussianMSE()  # per-view head/foot loss is always the original one

    def _forward(self, data):
        # model returns (map_res, imgs_res) or, with the missing-annotation branches, (map_res, imgs_res, mal_out)
        out = self.model(data)
        map_res, imgs_res = out[0], out[1]
        mal_out = out[2] if len(out) > 2 else None
        return map_res, imgs_res, mal_out

    def _loss(self, map_res, imgs_res, mal_out, map_gt, imgs_gt, dataset, active=False):
        """Returns (total_loss, components dict or None)."""
        img_loss = 0
        for img_res, img_gt in zip(imgs_res, imgs_gt):
            img_loss += self.img_criterion(img_res, img_gt.to(img_res.device), dataset.img_kernel)
        img_loss = img_loss / len(imgs_gt) * self.alpha
        if not self.is_mal:
            return self.criterion(map_res, map_gt.to(map_res.device), dataset.map_kernel) + img_loss, None
        out = self.criterion(map_res, map_gt.to(map_res.device), dataset.map_kernel, imgs_gt, dataset.img_kernel,
                             mal_out, active=active)
        return out['total'] + img_loss, out

    def _write_mal_stats(self, record):
        with open(os.path.join(self.logdir, 'mal_stats.jsonl'), 'a') as f:
            f.write(json.dumps(record) + '\n')

    def train(self, epoch, data_loader, optimizer, log_interval=100, cyclic_scheduler=None):
        self.model.train()
        losses = 0
        precision_s, recall_s = AverageMeter(), AverageMeter()
        active = self.is_mal and epoch > self.mal_warmup_epochs
        comp_s = {k: AverageMeter() for k in ['map', 'q', 's', 's_pos', 's_neg', 'prior', 'q_bg_mean', 'c_bg_mean', 'c_mean']}
        q_hist = torch.zeros(10)
        n_skipped = 0
        if self.is_mal:
            print(f'[mal] epoch {epoch}: L_conf/L_q {"ACTIVE" if active else "OFF (warm-up, plain GaussianMSE)"}; '
                  f'L_s on')
        t0 = time.time()
        t_b = time.time()
        t_forward = 0
        t_backward = 0
        for batch_idx, (data, map_gt, imgs_gt, _) in enumerate(data_loader):
            optimizer.zero_grad()
            if self.aux_optimizer is not None:
                self.aux_optimizer.zero_grad()
            map_res, imgs_res, mal_out = self._forward(data)
            t_f = time.time()
            t_forward += t_f - t_b
            loss, comp = self._loss(map_res, imgs_res, mal_out, map_gt, imgs_gt, data_loader.dataset, active)
            loss.backward()
            # NaN guard: a single non-finite loss / gradient would poison the weights for the rest of
            # the run, so skip this step (both optimizers) and report it instead of silently stepping.
            main_params = [p for g in optimizer.param_groups for p in g['params']]
            grad_norm = torch.nn.utils.clip_grad_norm_(
                main_params, self.clip_grad_norm if self.clip_grad_norm > 0 else float('inf'))
            if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
                n_skipped += 1
                if n_skipped <= 5 or n_skipped % 50 == 0:
                    print(f'[nan-guard] epoch {epoch} batch {batch_idx + 1}: loss={loss.item():.4g}, '
                          f'grad_norm={grad_norm.item():.4g}, maxima={map_res.max().item():.4g}'
                          + (f', L_map={comp["map"].item():.4g}, L_q={comp["q"].item():.4g}, '
                             f'L_s={comp["s"].item():.4g}' if comp is not None else '')
                          + ' -> step skipped')
                optimizer.zero_grad()
                if self.aux_optimizer is not None:
                    self.aux_optimizer.zero_grad()
                t_b = time.time()
                continue
            optimizer.step()
            if self.aux_optimizer is not None:
                self.aux_optimizer.step()
            losses += loss.item()
            if comp is not None:
                for k in ['map', 'q', 's', 's_pos', 's_neg', 'prior']:
                    comp_s[k].update(comp[k].detach().item())
                for k in ['q_bg_mean', 'c_bg_mean', 'c_mean']:
                    comp_s[k].update(comp['stats'][k])
                q_hist += comp['stats']['q_hist']
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
                    t_epoch, t_forward / max(batch_idx, 1), t_backward / max(batch_idx, 1), map_res.max()) + self._mal_str(comp_s))
                pass

        t1 = time.time()
        t_epoch = t1 - t0
        print('Train Epoch: {}, Batch:{}, Loss: {:.6f}, '
              'Precision: {:.1f}%, Recall: {:.1f}%, Time: {:.3f}'.format(
            epoch, len(data_loader), losses / len(data_loader), precision_s.avg * 100, recall_s.avg * 100, t_epoch)
              + self._mal_str(comp_s))
        if n_skipped:
            print(f'[nan-guard] epoch {epoch}: {n_skipped}/{len(data_loader)} steps skipped because of '
                  f'non-finite loss/gradients -- if this is most of the epoch the weights are already NaN')
        if torch.cuda.is_available():
            print('GPU peak memory this epoch: ' + ', '.join(
                f'cuda:{i} {torch.cuda.max_memory_allocated(i) / 2 ** 30:.2f} GB' for i in range(torch.cuda.device_count())))
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)
        if self.is_mal:
            hist = (q_hist / q_hist.sum().clamp_min(1)).tolist()
            print('[mal] epoch {} q histogram (10 bins on [0,1], fraction of BEV cells): {}'.format(
                epoch, ' '.join(f'{h:.3f}' for h in hist)))
            self._write_mal_stats({'epoch': epoch, 'active': active, 'loss': losses / len(data_loader),
                                   **{k: m.avg for k, m in comp_s.items()}, 'q_hist': hist})

        return losses / len(data_loader), precision_s.avg * 100

    def _mal_str(self, comp_s):
        if not self.is_mal or comp_s['map'].count == 0:
            return ''
        return (', L_map: {:.5f}, L_q: {:.5f}, L_s: {:.4f} (pos {:.4f} / neg {:.4f}), '
                'q_bg: {:.4f}, c_bg: {:.4f}, c: {:.4f}'.format(
                    comp_s['map'].avg, comp_s['q'].avg, comp_s['s'].avg, comp_s['s_pos'].avg, comp_s['s_neg'].avg,
                    comp_s['q_bg_mean'].avg, comp_s['c_bg_mean'].avg, comp_s['c_mean'].avg))

    def test(self, data_loader, res_fpath=None, gt_fpath=None, visualize=False):
        self.model.eval()
        losses = 0
        precision_s, recall_s = AverageMeter(), AverageMeter()
        all_res_list = []
        t0 = time.time()
        if res_fpath is not None:
            assert gt_fpath is not None
        for batch_idx, (data, map_gt, imgs_gt, frame) in enumerate(data_loader):
            with torch.no_grad():
                map_res, imgs_res, mal_out = self._forward(data)
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
                # test annotations are complete: report the plain GaussianMSE map loss (active=False)
                # so the number stays comparable with the baseline
                loss, _ = self._loss(map_res, imgs_res, mal_out, map_gt, imgs_gt, data_loader.dataset, active=False)
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
            n_rows = 4 if mal_out is not None else 2
            fig = plt.figure(figsize=(6, 2.5 * n_rows))
            subplt0 = fig.add_subplot(n_rows, 1, 1, title="output")
            subplt1 = fig.add_subplot(n_rows, 1, 2, title="target")
            subplt0.imshow(map_res.cpu().detach().numpy().squeeze())
            subplt1.imshow(self.criterion._traget_transform(map_res, map_gt.to(map_res.device),
                                                            data_loader.dataset.map_kernel)
                           .cpu().detach().numpy().squeeze())
            if mal_out is not None:
                subplt2 = fig.add_subplot(n_rows, 1, 3, title="gate q")
                subplt3 = fig.add_subplot(n_rows, 1, 4, title="multi-view evidence c")
                subplt2.imshow(mal_out['q'][0, 0].cpu().numpy(), vmin=0, vmax=1)
                subplt3.imshow(mal_out['c'][0, 0].cpu().numpy(), vmin=0, vmax=1)
            plt.tight_layout()
            plt.savefig(os.path.join(self.logdir, 'map.jpg'))
            plt.close(fig)
            if mal_out is not None:
                # per-view evidence s_v of camera 1 on top of the image
                s0 = torch.sigmoid(mal_out['s_logits'][0][0, 0]).detach().cpu().numpy()
                img0 = self.denormalize(data[0, 0]).cpu().numpy().squeeze().transpose([1, 2, 0])
                img0 = Image.fromarray((img0 * 255).astype('uint8'))
                add_heatmap_to_image(s0, img0).save(os.path.join(self.logdir, 'cam1_evidence.jpg'))

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
