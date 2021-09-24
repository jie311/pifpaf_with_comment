"""Train a pifpaf network."""

import argparse
import datetime
import socket

import torch

# 这里将相对运用的.改成了绝对引用
from openpifpaf import datasets, encoder, optimize, transforms  # ,logs
from openpifpaf.network import losses, nets, Trainer
from openpifpaf import __version__ as VERSION
import warnings

warnings.filterwarnings("ignore")


def default_output_file(args):
    out = 'outputs/{}-{}'.format(args.basenet, '-'.join(args.headnets))
    if args.square_edge != 321:
        out += '-edge{}'.format(args.square_edge)
    if args.regression_loss != 'laplace':
        out += '-{}'.format(args.regression_loss)
    if args.r_smooth != 0.0:
        out += '-rsmooth{}'.format(args.r_smooth)
    if args.dilation:
        out += '-dilation{}'.format(args.dilation)
    if args.dilation_end:
        out += '-dilationend{}'.format(args.dilation_end)

    now = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
    out += '-{}.pkl'.format(now)

    return out


def cli():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # logs.cli(parser)
    nets.cli(parser)
    losses.cli(parser)
    encoder.cli(parser)
    optimize.cli(parser)
    datasets.train_cli(parser)

    parser.add_argument('-o', '--output', default=None,
                        help='output file')
    parser.add_argument('--stride-apply', default=1, type=int,
                        help='apply and reset gradients every n batches')
    parser.add_argument('--epochs', default=75, type=int,
                        help='number of epochs to train')
    parser.add_argument('--freeze-base', default=0, type=int,
                        help='number of epochs to train with frozen base')
    parser.add_argument('--pre-lr', type=float, default=1e-4,
                        help='pre learning rate')
    parser.add_argument('--update-batchnorm-runningstatistics',
                        default=False, action='store_true',
                        help='update batch norm running statistics')
    parser.add_argument('--square-edge', default=401, type=int,
                        help='square edge of input images')
    # 前置因子 for head losses
    parser.add_argument('--lambdas', default=[30.0, 2.0, 2.0, 50.0, 3.0, 3.0],
                        type=float, nargs='+',
                        help='prefactor for head losses')
    parser.add_argument('--ema', default=1e-3, type=float,
                        help='ema decay constant')
    parser.add_argument('--debug-without-plots', default=False, action='store_true',
                        help='enable debug but dont plot')
    parser.add_argument('--profile', default=None,
                        help='enables profiling. specify path for chrome tracing file')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    args = parser.parse_args()

    if args.output is None:
        args.output = default_output_file(args)

    # if args.debug and 'skeleton' not in args.headnets:
    #     raise Exception('add "skeleton" as last headnet to see debug output')

    if args.debug_without_plots:
        args.debug = True

    # add args.device
    args.device = torch.device('cpu')
    args.pin_memory = False
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True

    return args


def main():
    args = cli()
    # logs.configure(args)
    net_cpu, start_epoch = nets.factory_from_args(args)

    for head in net_cpu.head_nets:
        head.apply_class_sigmoid = False

    net = net_cpu.to(device=args.device)
    if not args.disable_cuda and torch.cuda.device_count() > 1:
        print('Using multiple GPUs: {}'.format(torch.cuda.device_count()))
        net = torch.nn.DataParallel(net)

    optimizer, lr_scheduler = optimize.factory(args, net.parameters())
    loss_list = losses.factory_from_args(args)
    target_transforms = encoder.factory(args, net_cpu.io_scales())
    preprocess = transforms.Compose([
        transforms.HFlip(0.5),  # 水平翻转
        transforms.RescaleRelative(),  # 进行缩放,缩放的比例在[0.5,1]之间随机进行
        # 下面的crop作用对象是大图片，centerpad的对象是小图片
        transforms.Crop(401),  # 将图片随机裁剪成401x401大小,比这尺寸小的图片不做处理
        transforms.CenterPad(401),  # centerpad -> 将图片放在中心，四周分别padding，将图片变成401x401
    ])
    # 加载训练集，验证集，以及预训练集
    train_loader, val_loader, pre_train_loader = datasets.train_factory(
        args, preprocess, target_transforms)

    # 可视化，先略过了
    encoder_visualizer = None
    if not args.debug_without_plots:
        encoder_visualizer = encoder.Visualizer(args.headnets, net_cpu.io_scales())

    # 是否冻结 base net
    if args.freeze_base:
        # freeze base net parameters
        frozen_params = set()
        for n, p in net.named_parameters():
            # Freeze only base_net parameters.
            # Parameter names in DataParallel models start with 'module.'.
            if not n.startswith('module.base_net.') and not n.startswith('base_net.'):
                print('not freezing', n)
                continue
            print('freezing', n)
            if p.requires_grad is False:
                continue
            p.requires_grad = False
            frozen_params.add(p)
        print('froze {} parameters'.format(len(frozen_params)))

        # training
        foptimizer = torch.optim.SGD(
            (p for p in net.parameters() if p.requires_grad),
            lr=args.pre_lr, momentum=0.9, weight_decay=0.0, nesterov=True)
        ftrainer = Trainer(net, loss_list, foptimizer, args.output, args.lambdas,
                           device=args.device, fix_batch_norm=True,
                           encoder_visualizer=encoder_visualizer)
        for i in range(-args.freeze_base, 0):
            ftrainer.train(pre_train_loader, i)  # 预训练

        # unfreeze 解冻
        for p in frozen_params:
            p.requires_grad = True

    trainer = Trainer(
        # todo checkpoint，查看网络结构
        net, loss_list, optimizer, args.output,
        lr_scheduler=lr_scheduler,
        device=args.device,
        fix_batch_norm=not args.update_batchnorm_runningstatistics,
        lambdas=args.lambdas,
        stride_apply=args.stride_apply,
        ema_decay=args.ema,
        encoder_visualizer=encoder_visualizer,
        train_profile=args.profile,
        model_meta_data={
            'args': vars(args),
            'version': VERSION,
            'hostname': socket.gethostname(),
        },
    )
    trainer.loop(train_loader, val_loader, args.epochs, start_epoch=start_epoch)


if __name__ == '__main__':
    # python -m openpifpaf.train --lr=1e-3 --momentum=0.95 --epochs=75 --lr-decay 60 70 --batch-size=8 --basenet=resnet50block5 --head-quad=1 --headnets pif paf --square-edge=401 --regression-loss=laplace --lambdas 30 2 2 50 3 3  --freeze-base=1
    # python -m openpifpaf.webcam --scale 0.1 --source=0
    main()
