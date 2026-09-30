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
from multiview_detector.loss.pseudo_gaussian_mse import PseudoGaussianMSE, VARIANTS as PS_VARIANTS
from multiview_detector.models.persp_trans_detector import PerspTransDetector
from multiview_detector.models.image_proj_variant import ImageProjVariant
from multiview_detector.models.res_proj_variant import ResProjVariant
from multiview_detector.models.no_joint_conv_variant import NoJointConvVariant
from multiview_detector.utils.logger import Logger
from multiview_detector.utils.draw_curve import draw_curve
from multiview_detector.utils.image_utils import img_color_denormalize
from multiview_detector.trainer import PerspectiveTrainer


def ps_r_ignore(args):
    return 1.5 * args.ps_r if args.ps_r_ignore < 0 else args.ps_r_ignore


def pseudo_cfg(args):
    return {'alpha': args.ps_alpha, 'alpha_const': args.ps_alpha_const, 'views_k': args.ps_views_k,
            'min_score': args.ps_min_score, 'min_views': args.ps_min_views,
            'min_view_ratio': args.ps_min_view_ratio, 'ignore_min_views': args.ps_ignore_min_views,
            'border': args.ps_border * 4}  # output cells -> full-res grid cells (grid_reduce 4)


def pseudo_tag(args):
    """log sub-directory name, one per pseudo-label configuration"""
    tag = f'pseudo_{args.ps_variant}_r{args.ps_r:g}_ri{ps_r_ignore(args):g}_a{args.ps_alpha}'
    if args.ps_alpha == 'const':
        tag += f'{args.ps_alpha_const:g}'
    tag += f'_l{args.ps_lambda:g}_w{args.ps_warmup}r{args.ps_ramp}'
    if args.ps_min_score > 0:
        tag += f'_ms{args.ps_min_score:g}'
    if args.ps_min_views > 1:
        tag += f'_mv{args.ps_min_views}'
    if args.ps_min_view_ratio > 0:
        tag += f'_vr{args.ps_min_view_ratio:g}'
    if args.ps_ignore_min_views > 0:
        tag += f'_ig{args.ps_ignore_min_views}r{args.ps_ignore_r:g}'
    if args.ps_border > 0:
        tag += f'_b{args.ps_border:g}'
    if args.brl_beta > 0:
        tag += f'_brl{args.brl_beta:g}c{args.brl_conf_thr:g}p{args.brl_pos_thr:g}'
        if args.brl_border > 0:
            tag += f'bd{args.brl_border:g}'
    if args.view_ignore_dets:
        tag += '_vi' + (f'{args.view_ignore_score:g}' if args.view_ignore_score != 0.5 else '')
    return tag + '_' + os.path.basename(os.path.normpath(args.pseudo_dir))


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
                                 pseudo_dir=os.path.expanduser(args.pseudo_dir), pseudo_cfg=pseudo_cfg(args),
                                 view_ignore_dets=os.path.expanduser(args.view_ignore_dets) if args.view_ignore_dets
                                 else None, view_ignore_score=args.view_ignore_score)
    else:
        if args.view_ignore_dets or args.brl_beta > 0:
            raise Exception('--view_ignore_dets / --brl_beta are options of --loss pseudo')
        train_set = frameDataset(base, train=True, transform=train_trans, grid_reduce=4)
    test_set = frameDataset(base, train=False, transform=train_trans, grid_reduce=4)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.num_workers, pin_memory=True)

    # model
    if args.variant == 'default':
        model = PerspTransDetector(train_set, args.arch)
    elif args.variant == 'img_proj':
        model = ImageProjVariant(train_set, args.arch)
    elif args.variant == 'res_proj':
        model = ResProjVariant(train_set, args.arch)
    elif args.variant == 'no_joint_conv':
        model = NoJointConvVariant(train_set, args.arch)
    else:
        raise Exception('no support for this variant')

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader),
                                                    epochs=args.epochs)

    # loss
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.loss == 'pseudo':
        # kept labels as in GaussianMSE + pseudo labels from a 2D detector
        criterion = PseudoGaussianMSE(variant=args.ps_variant, r=args.ps_r, r_ignore=ps_r_ignore(args),
                                      r_ignore_only=args.ps_ignore_r, brl_beta=args.brl_beta,
                                      brl_conf_thr=args.brl_conf_thr, brl_pos_thr=args.brl_pos_thr,
                                      brl_border=args.brl_border).to(device)
        loss_tag = pseudo_tag(args)
    else:
        criterion = GaussianMSE().to(device)
        loss_tag = 'mse'

    # logging
    logdir = f'logs/{args.dataset}_frame/{loss_tag}/{args.variant}/' + \
        datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S') \
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
        model.load_state_dict(torch.load(resume_fname))
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

    # Pseudo labels (--loss pseudo). Distances are in OUTPUT map cells
    # (grid_reduce 4: 1 cell = 10 cm on Wildtrack / MultiviewX).
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'pseudo'],
                        help='mse = original GaussianMSE; pseudo = GaussianMSE on the kept labels + pseudo labels')
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
    parser.add_argument('--ps_min_score', type=float, default=0.0, help='positive only if score >= this')
    parser.add_argument('--ps_min_views', type=int, default=1, help='positive only if seen by >= this many cameras')
    parser.add_argument('--ps_min_view_ratio', type=float, default=0.0,
                        help='positive only if n_views / n_visible >= this (cameras that detect it / that can see it)')
    parser.add_argument('--ps_ignore_min_views', type=int, default=0,
                        help='3-tier pseudo: points failing the positive filters but with n_views >= this become '
                             'IGNORE regions (neither positive nor background); 0 = off (they are background)')
    parser.add_argument('--ps_ignore_r', type=float, default=10.0,
                        help='radius (output cells) of the weight-0 disk around an ignore-only point')
    parser.add_argument('--ps_border', type=float, default=0.0,
                        help='drop pseudo points closer than this (output cells, 10 = 1 m) to the edge of the '
                             'annotated area -- they stay background (most projection ghosts are there)')
    parser.add_argument('--ps_lambda', type=float, default=1.0, help='lambda_max of the pseudo term')
    # background recalibration (MSE form of BRLFocalLoss_v2), on top of the pseudo loss; off when brl_beta = 0
    parser.add_argument('--brl_beta', type=float, default=0.0,
                        help='weight of beta * (P - 1)^2 on confuse pixels; 0 = off')
    parser.add_argument('--brl_conf_thr', type=float, default=0.3,
                        help='background pixel with prediction >= this (detached) is confuse')
    parser.add_argument('--brl_pos_thr', type=float, default=0.1,
                        help='pixels with soft GT (kept labels) >= this are never confuse')
    parser.add_argument('--brl_border', type=float, default=0.0,
                        help='no confuse pixels within this many output cells of the map edge (10 = 1 m); 0 = off')
    # per-view head/foot loss: ignore 2D detector boxes (unlabelled people are not taught as "no person")
    parser.add_argument('--view_ignore_dets', type=str, default=None,
                        help='detect_2d.py json; background pixels inside its boxes get weight 0 in the per-view loss')
    parser.add_argument('--view_ignore_score', type=float, default=0.5)
    parser.add_argument('--ps_warmup', type=int, default=0, help='epochs with lambda_ps = 0')
    parser.add_argument('--ps_ramp', type=int, default=1, help='epochs of linear ramp to lambda_max after warm-up')
    args = parser.parse_args()

    main(args)
