"""Head networks."""

from abc import ABCMeta, abstractstaticmethod
import logging
import re

import torch

LOG = logging.getLogger(__name__)


class Head(metaclass=ABCMeta):
    @abstractstaticmethod
    def match(head_name):  # pylint: disable=unused-argument
        return False

    @classmethod
    def cli(cls, parser):
        """Add decoder specific command line arguments to the parser."""

    @classmethod
    def apply_args(cls, args):
        """Read command line arguments args to set class properties."""


class CompositeField(Head, torch.nn.Module):
    default_dropout_p = 0.0
    default_quad = 0
    default_kernel_size = 1
    default_padding = 0
    default_dilation = 1

    # 传过来的参数只有head_name([pif,paf]) 和 in_features,其余都是默认的
    def __init__(self, head_name, in_features, *,
                 n_fields=None,
                 n_confidences=1, n_vectors=None, n_scales=None,
                 kernel_size=None, padding=None, dilation=None):
        super(CompositeField, self).__init__()

        n_fields = n_fields or self.determine_nfields(head_name)  # pif 返回 17, paf 返回 19
        n_vectors = n_vectors or self.determine_nvectors(head_name)  # pif 返回 1, paf 返回 2
        n_scales = n_scales or self.determine_nscales(head_name)  # pif 返回 1, paf 返回 0
        LOG.debug('%s loss: fields = %d, confidences = %d, vectors = %d, scales = %d',
                  head_name, n_fields, n_confidences, n_vectors, n_scales)

        if kernel_size is None:
            kernel_size = {'wpaf': 3}.get(head_name, self.default_kernel_size)  # default 为 1
        if padding is None:
            padding = {'wpaf': 5}.get(head_name, self.default_padding)  # default 为 0
        if dilation is None:
            dilation = {'wpaf': 5}.get(head_name, self.default_dilation)  # default 为 1
        LOG.debug('%s loss: kernel = %d, padding = %d, dilation = %d',
                  head_name, kernel_size, padding, dilation)

        self.shortname = head_name  # 重新命个名,成为类变量
        self.apply_class_sigmoid = True
        self.dilation = dilation

        self.dropout = torch.nn.Dropout2d(p=self.default_dropout_p)
        self._quad = self.default_quad  # 默认为0

        # classification 分类
        out_features = n_fields * (4 ** self._quad)  # 注意这个out_features,后面会多次用到
        self.class_convs = torch.nn.ModuleList([
            torch.nn.Conv2d(in_features, out_features,
                            kernel_size, padding=padding, dilation=dilation)
            for _ in range(n_confidences)  # 这里的n_confidences 可以决定Conv2D的个数,默认为1
        ])

        # regression 回归
        self.reg_convs = torch.nn.ModuleList([
            torch.nn.Conv2d(in_features, 2 * out_features,
                            kernel_size, padding=padding, dilation=dilation)
            for _ in range(n_vectors)  # 同上
        ])
        self.reg_spreads = torch.nn.ModuleList([
            torch.nn.Conv2d(in_features, out_features,
                            kernel_size, padding=padding, dilation=dilation)
            for _ in self.reg_convs  # 同上
        ])

        # scale
        self.scale_convs = torch.nn.ModuleList([
            torch.nn.Conv2d(in_features, out_features,
                            kernel_size, padding=padding, dilation=dilation)
            for _ in range(n_scales)
        ])

        # dequad
        # 将一个H × W的低分辨率输入图像，通过Sub-pixel操作将其变为rH x rW的高分辨率图像。
        # 此处将图片放大为原来的两倍
        self.dequad_op = torch.nn.PixelShuffle(2)

    @staticmethod
    # 决定有多少 n_field
    def determine_nfields(head_name):
        m = re.match('p[ia]f([0-9]+)$', head_name)
        if m is not None:
            return int(m.group(1))
        # head_name 在上述字典中的话返回对应的value值,否则返回17
        # 默认 paf 返回 19, pif 返回 17
        return {
            'paf': 19,
            'pafb': 19,
            'pafsb': 19,
            'pafs19': 19,
            'wpaf': 19,
        }.get(head_name, 17)

    @staticmethod
    def determine_nvectors(head_name):
        # pif 返回 1, paf 返回 2
        if 'pif' in head_name:
            return 1
        if 'paf' in head_name:
            return 2
        return 0

    @staticmethod
    def determine_nscales(head_name):
        if 'pif' in head_name:
            return 1
        if 'paf' in head_name:
            return 0
        return 0

    @staticmethod
    def match(head_name):
        return head_name in (
            'pif',
            'paf',
            'pafs',
            'wpaf',
            'pafb',
            'pafs19',
            'pafsb',
        ) or re.match('p[ia]f([0-9]+)$', head_name) is not None

    @classmethod
    def cli(cls, parser):
        group = parser.add_argument_group('head')
        group.add_argument('--head-dropout', default=cls.default_dropout_p, type=float,
                           help='zeroing probability of feature in head input')
        group.add_argument('--head-quad', default=cls.default_quad, type=int,
                           help='number of times to apply quad (subpixel conv) to heads')
        group.add_argument('--head-kernel-size', default=cls.default_kernel_size, type=int)
        group.add_argument('--head-padding', default=cls.default_padding, type=int)
        group.add_argument('--head-dilation', default=cls.default_dilation, type=int)

    @classmethod
    def apply_args(cls, args):
        cls.default_dropout_p = args.head_dropout
        cls.default_quad = args.head_quad
        cls.default_kernel_size = args.head_kernel_size
        cls.default_padding = args.head_padding
        cls.default_dilation = args.head_dilation

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.dropout(x)

        # classification
        classes_x = [class_conv(x) for class_conv in self.class_convs]
        if self.apply_class_sigmoid:
            classes_x = [torch.sigmoid(class_x) for class_x in classes_x]

        # regressions
        regs_x = [reg_conv(x) * self.dilation for reg_conv in self.reg_convs]
        regs_x_spread = [reg_spread(x) for reg_spread in self.reg_spreads]

        # scale
        scales_x = [scale_conv(x) for scale_conv in self.scale_convs]
        scales_x = [torch.nn.functional.relu(scale_x) for scale_x in scales_x]

        for _ in range(self._quad):
            classes_x = [self.dequad_op(class_x)[:, :, :-1, :-1]
                         for class_x in classes_x]
            regs_x = [self.dequad_op(reg_x)[:, :, :-1, :-1]
                      for reg_x in regs_x]
            regs_x_spread = [self.dequad_op(reg_x_spread)[:, :, :-1, :-1]
                             for reg_x_spread in regs_x_spread]
            scales_x = [self.dequad_op(scale_x)[:, :, :-1, :-1]
                        for scale_x in scales_x]

        regs_x = [
            reg_x.reshape(reg_x.shape[0],
                          reg_x.shape[1] // 2,
                          2,
                          reg_x.shape[2],
                          reg_x.shape[3])
            for reg_x in regs_x
        ]

        return classes_x + regs_x + regs_x_spread + scales_x
