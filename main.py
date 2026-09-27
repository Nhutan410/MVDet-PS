import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import sys
import shutil
import datetime
import tqdm
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as T
from multiview_detector.datasets import *
from multiview_detector.loss.gaussian_mse import GaussianMSE
from multiview_detector.loss.confuse_gaussian_mse import ConfuseGaussianMSE
from multiview_detector.loss.missing_annotation_loss import MissingAnnotationLoss
from multiview_detector.loss.pseudo_gaussian_mse import PseudoGaussianMSE, VARIANTS as PS_VARIANTS
from multiview_detector.models.persp_trans_detector import PerspTransDetector
from multiview_detector.models.image_proj_variant import ImageProjVariant
from multiview_detector.models.res_proj_variant import ResProjVariant
from multiview_detector.models.no_joint_conv_variant import NoJointConvVariant
from multiview_detector.utils.logger import Logger
from multiview_detector.utils.draw_curve import draw_curve
from multiview_detector.utils.image_utils import img_color_denormalize
from multiview_detector.trainer import PerspectiveTrainer


def build_mal_cfg(args):
    """Model-side config of the missing-annotation branches (None -> original MVDet)."""
    if args.loss != 'mal':
        return None
    return {'s_v_backbone': args.mal_ev_backbone, 'consensus_agg': args.mal_consensus,
            'min_visible_views': args.mal_min_visible_views, 'dino_name': args.mal_dino_name,
            'dino_input': tuple(args.mal_dino_input), 'ev_device': args.mal_ev_device}


def ps_r_ignore(args):
    return 1.5 * args.ps_r if args.ps_r_ignore < 0 else args.ps_r_ignore


def pseudo_cfg(args):
    return {'alpha': args.ps_alpha, 'alpha_const': args.ps_alpha_const, 'views_k': args.ps_views_k,
            'min_score': args.ps_min_score, 'min_views': args.ps_min_views}


def pseudo_tag(args):
    tag = f'pseudo_{args.ps_variant}_r{args.ps_r:g}_ri{ps_r_ignore(args):g}_a{args.ps_alpha}'
    if args.ps_alpha == 'const':
        tag += f'{args.ps_alpha_const:g}'
    tag += f'_l{args.ps_lambda:g}_w{args.ps_warmup}r{args.ps_ramp}'
    if args.ps_min_score > 0:
        tag += f'_ms{args.ps_min_score:g}'
    if args.ps_min_views > 1:
        tag += f'_mv{args.ps_min_views}'
    return tag + '_' + os.path.basename(os.path.normpath(args.pseudo_dir))


def build_criterion(args, model=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # cpu only for local smoke tests
    if args.loss == 'pseudo':
        # kept labels as in GaussianMSE + pseudo labels from a 2D detector -- see PSEUDO_LABEL.md
        return PseudoGaussianMSE(variant=args.ps_variant, r=args.ps_r, r_ignore=ps_r_ignore(args)).to(device)
    if args.loss == 'mal':
        # missing-annotation robust loss -- see MISSING_ANNOTATION_LOSS.md and
        # multiview_detector/loss/missing_annotation_loss.py. Needs the model for the BEV->view warp.
        return MissingAnnotationLoss(
            lambda_q=args.mal_lambda_q, lambda_s=args.mal_lambda_s,
            lambda_prior=args.mal_lambda_prior, pi_prior=args.mal_pi_prior,
            min_dist_easy_neg=args.mal_min_dist_easy_neg, conf_grad_q=args.mal_conf_grad_q,
            project_bev_to_views=model.project_bev_to_views,
        ).to(device)
    if args.loss == 'confuse_gaussian':
        # keeps the Gaussian soft target but uses no pos_thr gate --
        # see multiview_detector/loss/confuse_gaussian_mse.py docstring for how it protects
        # known/kept people continuously via (1 - soft_gt) instead of a hard threshold.
        return ConfuseGaussianMSE(
            confuse_pred_thr=args.brl_confuse_thr,
            beta=args.brl_beta,
            mirror=not args.brl_no_mirror,
        ).to(device)
    return GaussianMSE().to(device)


def main(args):
    # seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.benchmark = True

    # dataset
    normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_trans = T.Compose([T.Resize([720, 1280]), T.ToTensor(), normalize, ])
    if 'wildtrack' in args.dataset:
        data_path = os.path.expanduser('~/Data/Wildtrack')
        base = Wildtrack(data_path)
    elif 'multiviewx' in args.dataset:
        data_path = os.path.expanduser('~/Data/MultiviewX')
        base = MultiviewX(data_path)
    else:
        raise Exception('must choose from [wildtrack, multiviewx]')
    if args.loss == 'pseudo':
        if args.pseudo_dir is None:
            raise Exception('--loss pseudo needs --pseudo_dir (tools/pseudo/build_pseudo.py output)')
        if args.batch_size != 1:
            raise Exception('--loss pseudo: pseudo points differ in number per frame, use --batch_size 1')
        train_set = frameDataset(base, train=True, transform=train_trans, grid_reduce=4,
                                 pseudo_dir=os.path.expanduser(args.pseudo_dir), pseudo_cfg=pseudo_cfg(args))
    else:
        train_set = frameDataset(base, train=True, transform=train_trans, grid_reduce=4)
    test_set = frameDataset(base, train=False, transform=train_trans, grid_reduce=4)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.num_workers, pin_memory=True)

    # model
    if args.loss == 'mal' and args.variant != 'default':
        raise Exception('--loss mal needs the gate/evidence branches, only implemented in the default variant')
    if args.variant == 'default':
        model = PerspTransDetector(train_set, args.arch, mal_cfg=build_mal_cfg(args))
    elif args.variant == 'img_proj':
        model = ImageProjVariant(train_set, args.arch)
    elif args.variant == 'res_proj':
        model = ResProjVariant(train_set, args.arch)
    elif args.variant == 'no_joint_conv':
        model = NoJointConvVariant(train_set, args.arch)
    else:
        raise Exception('no support for this variant')

    main_params = model.main_parameters() if hasattr(model, 'main_parameters') else model.parameters()
    optimizer = optim.SGD(main_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader),
                                                    epochs=args.epochs)
    # evidence branch s_v + gate head q have their own optimizer: no parameter, no gradient shared with p
    aux_optimizer = None
    if args.loss == 'mal':
        aux_params = model.aux_parameters()
        print(f'[mal] aux optimizer (evidence {args.mal_ev_backbone} + gate head): '
              f'{sum(p.numel() for p in aux_params)} params, Adam lr={args.mal_aux_lr}; main SGD excludes them')
        aux_optimizer = optim.Adam(aux_params, lr=args.mal_aux_lr)

    # loss
    criterion = build_criterion(args, model)

    # logging
    if args.loss == 'confuse_gaussian':
        loss_tag = f'{args.loss}_b{args.brl_beta}_c{args.brl_confuse_thr}'
        if args.brl_no_mirror:
            loss_tag += '_nomirror'
    elif args.loss == 'mal':
        loss_tag = (f'mal_{args.mal_ev_backbone}_{args.mal_consensus}_w{args.mal_warmup_epochs}'
                    f'_d{args.mal_min_dist_easy_neg}_lq{args.mal_lambda_q}_ls{args.mal_lambda_s}')
        if args.mal_lambda_prior > 0:
            loss_tag += f'_lp{args.mal_lambda_prior}_pi{args.mal_pi_prior}'
        if args.mal_conf_grad_q:
            loss_tag += '_gradq'
    elif args.loss == 'pseudo':
        loss_tag = pseudo_tag(args)
    else:
        loss_tag = 'mse'
    logdir = f'logs/{args.dataset}_frame/{loss_tag}/{args.variant}/' + datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S') \
        if not args.resume else f'logs/{args.dataset}_frame/{loss_tag}/{args.variant}/{args.resume}'
    if args.resume is None:
        os.makedirs(logdir, exist_ok=True)
        shutil.copytree('./multiview_detector', logdir + '/scripts/multiview_detector', dirs_exist_ok=True)
        for script in os.listdir('.'):
            if script.split('.')[-1] == 'py':
                dst_file = os.path.join(logdir, 'scripts', os.path.basename(script))
                shutil.copyfile(script, dst_file)
        sys.stdout = Logger(os.path.join(logdir, 'log.txt'), )
    print('Settings:')
    print(vars(args))

    # draw curve
    x_epoch = []
    train_loss_s = []
    train_prec_s = []
    test_loss_s = []
    test_prec_s = []
    test_moda_s = []

    trainer = PerspectiveTrainer(model, criterion, logdir, denormalize, args.cls_thres, args.alpha,
                                 mal_warmup_epochs=args.mal_warmup_epochs, aux_optimizer=aux_optimizer,
                                 clip_grad_norm=args.clip_grad_norm,
                                 ps_schedule=(args.ps_lambda, args.ps_warmup, args.ps_ramp))

    # learn
    if args.resume is None:
        print('Testing...')
        trainer.test(test_loader, os.path.join(logdir, 'test.txt'), train_set.gt_fpath, True)

        for epoch in tqdm.tqdm(range(1, args.epochs + 1)):
            print('Training...')
            train_loss, train_prec = trainer.train(epoch, train_loader, optimizer, args.log_interval, scheduler)
            print('Testing...')
            test_loss, test_prec, moda = trainer.test(test_loader, os.path.join(logdir, 'test.txt'),
                                                      train_set.gt_fpath, True)

            x_epoch.append(epoch)
            train_loss_s.append(train_loss)
            train_prec_s.append(train_prec)
            test_loss_s.append(test_loss)
            test_prec_s.append(test_prec)
            test_moda_s.append(moda)
            draw_curve(os.path.join(logdir, 'learning_curve.jpg'), x_epoch, train_loss_s, train_prec_s,
                       test_loss_s, test_prec_s, test_moda_s)
            # save
            torch.save(model.state_dict(), os.path.join(logdir, 'MultiviewDetector.pth'))
    else:
        resume_dir = f'logs/{args.dataset}_frame/{loss_tag}/{args.variant}/' + args.resume
        resume_fname = resume_dir + '/MultiviewDetector.pth'
        model.load_state_dict(torch.load(resume_fname), strict=(args.loss != 'mal'))
        model.eval()
    print('Test loaded model...')
    trainer.test(test_loader, os.path.join(logdir, 'test.txt'), train_set.gt_fpath, True)


if __name__ == '__main__':
    # settings
    parser = argparse.ArgumentParser(description='Multiview detector')
    parser.add_argument('--reID', action='store_true')
    parser.add_argument('--cls_thres', type=float, default=0.4)
    parser.add_argument('--alpha', type=float, default=1.0, help='ratio for per view loss')
    parser.add_argument('--variant', type=str, default='default',
                        choices=['default', 'img_proj', 'res_proj', 'no_joint_conv'])
    parser.add_argument('--arch', type=str, default='resnet18', choices=['vgg11', 'resnet18'])
    parser.add_argument('-d', '--dataset', type=str, default='wildtrack', choices=['wildtrack', 'multiviewx'])
    parser.add_argument('-j', '--num_workers', type=int, default=4)
    parser.add_argument('-b', '--batch_size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: None)')
    parser.add_argument('--clip_grad_norm', type=float, default=0.0,
                        help='max grad norm of the main-branch parameters; 0 = off (original MVDet). '
                             'Non-finite loss/gradient steps are always skipped and reported.')

    # Confuse-region heatmap loss, adapted from the BRL idea in
    # duclld1709/multiview-pedestrian-detection (Background Recalibration Loss) -- keeps the
    # Gaussian soft target (identical to --loss mse) but uses no pos_thr gate to decide which
    # pixels are protected: a confuse-candidate pixel (pred >= c) is blended continuously by
    # (1 - soft_gt) instead, so pixels near a KNOWN/KEPT person are auto-protected without any
    # extra threshold (see confuse_gaussian_mse.py docstring for the full derivation).
    parser.add_argument('--loss', type=str, default='mse', choices=['confuse_gaussian', 'mal', 'mse', 'pseudo'],
                        help='confuse_gaussian = Gaussian target, no pos_thr gate, continuous '
                             '(1 - soft_gt) blend for confuse pixels; mal = missing-annotation robust loss '
                             '(learned gate q anchored to independent multi-view evidence c, see '
                             'MISSING_ANNOTATION_LOSS.md); mse = original GaussianMSE; pseudo = GaussianMSE on '
                             'the kept labels + pseudo labels from a 2D detector (PSEUDO_LABEL.md)')
    parser.add_argument('--brl_confuse_thr', type=float, default=0.3,
                        help='pred threshold on background to mark confuse (possible missing GT)')
    parser.add_argument('--brl_beta', type=float, default=0.1,
                        help='weight / strength of confuse term')
    parser.add_argument('--brl_no_mirror', action='store_true',
                        help='if set, down-weight bg MSE on confuse instead of mirroring toward 1')
    # Missing-annotation robust loss (--loss mal). Every term can be switched off independently for
    # ablations: lambda_q=0 leaves q at its init (~0.02, i.e. nearly the baseline), lambda_s=0 stops
    # the evidence branch from learning (c stays at its init), lambda_prior=0 disables L_prior.
    parser.add_argument('--mal_warmup_epochs', type=int, default=3,
                        help='epochs of plain GaussianMSE (q fixed at 0) before L_conf / L_q switch on')
    parser.add_argument('--mal_min_dist_easy_neg', type=int, default=10,
                        help='min distance (reduced BEV cells; Wildtrack: 1 cell = 10 cm) from every kept '
                             'annotation for a pixel to count as easy negative for s_v; closer unlabeled '
                             'cells are the confusion zone and are ignored by L_s')
    parser.add_argument('--mal_consensus', type=str, default='median', choices=['median', 'percentile25', 'min'],
                        help='how the projected per-view evidence is aggregated into c(x, y) (never mean)')
    parser.add_argument('--mal_min_visible_views', type=int, default=2,
                        help='cells seen by fewer cameras get c = 0 (no single-view inference)')
    parser.add_argument('--mal_lambda_q', type=float, default=1.0, help='weight of L_q = (q - stopgrad(c))^2')
    parser.add_argument('--mal_lambda_s', type=float, default=1.0, help='weight of the evidence loss L_s')
    parser.add_argument('--mal_lambda_prior', type=float, default=0.0,
                        help='weight of L_prior = (mean_bg(q) - pi)^2; 0 = off (L_q is the main anchor)')
    parser.add_argument('--mal_pi_prior', type=float, default=0.0, help='estimated missing-label rate for L_prior')
    parser.add_argument('--mal_ev_backbone', type=str, default='frozen_dinov2',
                        choices=['frozen_dinov2', 'separate_trainable'],
                        help='backbone of the evidence branch s_v (never shared with the main backbone)')
    parser.add_argument('--mal_aux_lr', type=float, default=1e-3, help='Adam lr of the evidence branch + gate head')
    parser.add_argument('--mal_ev_device', type=str, default='auto',
                        help='device of the evidence branch: auto = same GPU as base_pt1 (cuda:1 when 2 GPUs are '
                             'visible, else cuda:0), or an explicit cuda:N')
    parser.add_argument('--mal_dino_name', type=str, default='dinov2_vits14',
                        help='torch.hub facebookresearch/dinov2 model for --mal_ev_backbone frozen_dinov2')
    parser.add_argument('--mal_dino_input', type=int, nargs=2, default=[504, 896],
                        help='H W the views are resized to before DINOv2 (multiples of 14)')
    # Pseudo labels (--loss pseudo), PSEUDO_LABEL.md. Distances are in OUTPUT map cells
    # (grid_reduce 4: 1 cell = 10 cm on Wildtrack / MultiviewX).
    parser.add_argument('--pseudo_dir', type=str, default=None,
                        help='per-frame pseudo-label json dir from tools/pseudo/build_pseudo.py or make_oracle.py')
    parser.add_argument('--ps_variant', type=str, default='gauss', choices=PS_VARIANTS,
                        help='gauss = Gaussian at a random point of the r-disk (B); point = one pixel at a random '
                             'point pushed to 1 (A, r=0: basic alpha*l(P,1)); maxval = max of P in the disk pushed '
                             'to 1 (C); mil = Gaussian at the argmax of P in the disk')
    parser.add_argument('--ps_r', type=float, default=0.0, help='location-uncertainty radius (output cells)')
    parser.add_argument('--ps_r_ignore', type=float, default=-1,
                        help='background pixels within this radius of a pseudo point get weight 0; -1 = 1.5 * ps_r')
    parser.add_argument('--ps_alpha', type=str, default='const', choices=['const', 'score', 'views'],
                        help='confidence alpha: constant, detector score, or score * min(1, n_views / min(k, n_visible))')
    parser.add_argument('--ps_alpha_const', type=float, default=0.5)
    parser.add_argument('--ps_views_k', type=int, default=3)
    parser.add_argument('--ps_min_score', type=float, default=0.0, help='drop pseudo points with a lower score')
    parser.add_argument('--ps_min_views', type=int, default=1, help='drop pseudo points seen by fewer cameras')
    parser.add_argument('--ps_lambda', type=float, default=1.0, help='lambda_max of the pseudo term')
    parser.add_argument('--ps_warmup', type=int, default=0, help='epochs with lambda_ps = 0')
    parser.add_argument('--ps_ramp', type=int, default=1, help='epochs of linear ramp to lambda_max after warm-up')
    parser.add_argument('--mal_conf_grad_q', action='store_true',
                        help='ABLATION ONLY: let L_conf backprop into q (default: q detached in L_conf, '
                             'learned only from c via L_q -- the spec forbids self-judging)')
    args = parser.parse_args()

    main(args)
