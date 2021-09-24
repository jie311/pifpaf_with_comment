"""Decoder for pif fields."""

from collections import defaultdict
import logging
import re
import time

import numpy as np

from .annotation import AnnotationWithoutSkeleton
from .decoder import Decoder
from .utils import index_field, scalar_square_add_single, normalize_pif

# pylint: disable=import-error
from openpifpaf.functional import (scalar_square_add_constant, scalar_square_add_gauss)


class Pif(Decoder):
    default_pif_fixed_scale = None

    def __init__(self, stride, seed_threshold,
                 head_index=None,
                 profile=None,
                 debug_visualizer=None,
                 **kwargs):
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.debug('unused arguments %s', kwargs)

        self.stride = stride
        self.hr_scale = self.stride
        self.head_index = head_index or 0
        self.profile = profile
        self.seed_threshold = seed_threshold
        self.debug_visualizer = debug_visualizer
        self.pif_fixed_scale = self.default_pif_fixed_scale

        self.pif_nn = 16

    @staticmethod
    def match(head_names):
        return head_names in (
            ('pif',),
        ) or (
                       len(head_names) == 1 and
                       re.match('pif([0-9]+)$', head_names[0]) is not None
               )

    @classmethod
    def apply_args(cls, args):
        cls.default_pif_fixed_scale = args.pif_fixed_scale

    def __call__(self, fields):
        start = time.time()
        if self.profile is not None:
            self.profile.enable()

        pif = fields[self.head_index]
        if self.debug_visualizer:
            self.debug_visualizer.pif_raw(pif, self.stride)
        pif = normalize_pif(*pif, fixed_scale=self.pif_fixed_scale)

        gen = PifGenerator(
            pif,
            stride=self.stride,
            seed_threshold=self.seed_threshold,
            pif_nn=self.pif_nn,
            debug_visualizer=self.debug_visualizer,
        )

        annotations = gen.annotations()

        print('annotations', len(annotations), time.time() - start)
        if self.profile is not None:
            self.profile.disable()
        return annotations


class PifGenerator(object):
    def __init__(self, pif_field, *,
                 stride,
                 seed_threshold,
                 pif_nn,
                 debug_visualizer=None):
        self.pif = pif_field

        self.stride = stride
        self.seed_threshold = seed_threshold  # default: 0.2
        self.pif_nn = pif_nn
        self.debug_visualizer = debug_visualizer
        self.timers = defaultdict(float)

        # pif init
        # 初始化,这里的hr是high resolution 的缩写
        self._pifhr, self._pifhr_scales = self._target_intensities()
        if self.debug_visualizer:
            self.debug_visualizer.pifhr(self._pifhr)

    def _target_intensities(self, v_th=0.1):
        start = time.time()
        # pif的维度可能是之前在utils中那个,(17, 4, h, w),分别为[confidence, x, y, scale]
        targets = np.zeros((self.pif.shape[0],
                            int(self.pif.shape[2] * self.stride),
                            int(self.pif.shape[3] * self.stride)))
        scales = np.zeros(targets.shape)  # (17, h * stride, w * stride)
        ns = np.zeros(targets.shape)  # (17, h * stride, w * stride)
        for t, p, scale, n in zip(targets, self.pif, scales, ns):
            # t->(h*stride, w*stride), p->(4, h*str, w*str), scale->(h*str, w*str),  n->(h*str, w*str)
            # targets,                 self.pif,             scales,                 ns
            v, x, y, s = p[:, p[0] > v_th]  # 取完indice后的shape是(4, num), num是>v_th的数目
            # v, x, y, s的shape均为(num,)
            x = x * self.stride  # 加细后的x, (num,)
            y = y * self.stride  # 加细后的y
            s = s * self.stride  # scale
            # (double[:, :] field, x_np, y_np, sigma_np, v_np, double truncate=2.0)
            # t是target,x,y是坐标,s是scale
            # todo self.pif_nn这里是16,原因可能是算的是17个关节在这个位置上的累加,所以除16?
            # 以x,y为中心,半径为0.5 * s 的方形区域内的点置信度加上一个高斯核,以达到high resolution的目的
            scalar_square_add_gauss(t, x, y, s, v / self.pif_nn, truncate=0.5)
            # (double[:, :] field, x_np, y_np, width_np, double[:] v)
            # 以x,y为中心,半径为 s 的方形区域内的点scale加上s * v
            scalar_square_add_constant(scale, x, y, s, s * v)
            # 以x,y为中心,半径为 s 的方形区域内的点n加上v
            scalar_square_add_constant(n, x, y, s, v)

        targets = np.minimum(1.0, targets)

        m = ns > 0
        scales[m] = scales[m] / ns[m]  # 相除,类似加权平均?
        print('target_intensities', time.time() - start)
        # (17, h * stride, w * stride)
        return targets, scales

    def annotations(self):
        start = time.time()

        seeds = self._pifhr_seeds()
        annotations = []
        for v, f, x, y in seeds:
            ann = AnnotationWithoutSkeleton(f, (x, y, v), self._pifhr_scales.shape[0])
            ann.fill_joint_scales(self._pifhr_scales, self.stride)
            annotations.append(ann)

        print('keypoint sets', len(annotations), time.time() - start)
        return annotations

    # 产生seed,按照confidence从大往小选,其中为了避免选到同一关节的两个点采用了一定的距离阈值
    def _pifhr_seeds(self):
        start = time.time()
        seeds = []
        for field_i, (f, s) in enumerate(zip(self._pifhr, self._pifhr_scales)):
            # f:(h * stride, w * stride)    s:(h * stride, w * stride)
            # 获得一个(2, h * stride, w * stride)大小的索引矩阵，y在第一维, x在第二维
            index_fields = index_field(f.shape)
            # 将索引矩阵与 f 拼起来,这里的 np.expand_dims 类似 torch.unsqueeze
            # 这里candidates的shape为(3, h * stride, w * stride)
            candidates = np.concatenate((index_fields, np.expand_dims(f, 0)), 0)

            mask = f > self.seed_threshold  # 大于阈值,default为0.2
            # numpy.moveaxis(a, source, destination)将a的source维移到destination上
            # 例如 x = np.zeros((3, 4, 5)) 经过 np.moveaxis(x, 0, -1).shape 为 (4, 5, 3)
            # 下面操作完后维数不确定,具体视mask产生的true的个数而定,设为n
            # 得到一个(n, 3) 维度的数组,3列中第1列是y坐标,第2列是x坐标,
            # 第3列是对应的满足 f>self.seed_threshold 的 f 值
            candidates = np.moveaxis(candidates[:, mask], 0, -1)

            occupied = np.zeros(s.shape)  # (h * stride, w * stride)
            # 按confidence值排序
            for c in sorted(candidates, key=lambda c: c[2], reverse=True):
                i, j = int(c[0]), int(c[1])  # x与y坐标
                if occupied[j, i]:
                    continue

                width = max(4, s[j, i])
                # 为以(c[0], c[1])为中心，宽度为width/2范围内所有值+1
                scalar_square_add_single(occupied, c[0], c[1], width / 2.0, 1.0)
                # c[2]是confidence, field_i是关节index, c[0]是x, c[1]是y
                seeds.append((c[2], field_i, c[0] / self.stride, c[1] / self.stride))

            if self.debug_visualizer:
                if field_i in self.debug_visualizer.pif_indices:
                    print('occupied seed, field {}'.format(field_i))
                    self.debug_visualizer.occupied(occupied)

        seeds = list(sorted(seeds, reverse=True))
        if len(seeds) > 500:
            if seeds[500][0] > 0.1:
                seeds = [s for s in seeds if s[0] > 0.1]
            else:
                seeds = seeds[:500]

        if self.debug_visualizer:
            self.debug_visualizer.seeds(seeds, self.stride)

        print('seeds', len(seeds), time.time() - start)
        return seeds
