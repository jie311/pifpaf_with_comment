import logging
import numpy as np
import scipy
import torch

from ..data import COCO_PERSON_SKELETON, DENSER_COCO_PERSON_SKELETON, KINEMATIC_TREE_SKELETON
from .annrescaler import AnnRescaler
from .encoder import Encoder
from .utils import create_sink, mask_valid_area


class Paf(Encoder):
    default_min_size = 3
    default_fixed_size = False
    default_aspect_ratio = 0.0

    def __init__(self, head_name, stride, *, skeleton=None, n_keypoints=17, **kwargs):
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.debug('unused arguments in %s: %s', head_name, kwargs)

        # skeleton 就是一个关节点之间如何连接的数组
        if skeleton is None:
            if head_name in ('paf', 'paf19', 'pafs', 'wpaf', 'pafb'):
                skeleton = COCO_PERSON_SKELETON  # 总共有19个连接
            elif head_name in ('paf16',):
                skeleton = KINEMATIC_TREE_SKELETON
            elif head_name in ('paf44',):
                skeleton = DENSER_COCO_PERSON_SKELETON
            else:
                raise Exception('unknown skeleton type of head')

        self.stride = stride
        self.n_keypoints = n_keypoints  # 17
        self.skeleton = skeleton  # skeleton 就是一个关节点之间如何连接的数组

        self.min_size = self.default_min_size  # min side length of the PAF field, default 是 3
        self.fixed_size = self.default_fixed_size  # fixed paf size,default 是 False
        self.aspect_ratio = self.default_aspect_ratio  # paf width relative to its length, default 是 0.0

        if self.fixed_size:
            assert self.aspect_ratio == 0.0

    @staticmethod
    def match(head_name):
        return head_name in (
            'paf',
            'paf19',
            'paf16',
            'paf44',
            'pafs',
            'wpaf',
            'pafb',
        )

    @classmethod
    def cli(cls, parser):
        group = parser.add_argument_group('paf encoder')
        group.add_argument('--paf-min-size', default=cls.default_min_size, type=int,
                           help='min side length of the PAF field')
        group.add_argument('--paf-fixed-size', default=cls.default_fixed_size, action='store_true',
                           help='fixed paf size')
        group.add_argument('--paf-aspect-ratio', default=cls.default_aspect_ratio, type=float,
                           help='paf width relative to its length')

    @classmethod
    def apply_args(cls, args):
        cls.default_min_size = args.paf_min_size
        cls.default_fixed_size = args.paf_fixed_size
        cls.default_aspect_ratio = args.paf_aspect_ratio

    def __call__(self, anns, width_height_original):
        rescaler = AnnRescaler(self.stride, self.n_keypoints)
        # 返回缩放后的keypoints_sets，valid area 指的是有效的区域,将来需要学习的区域
        # bg_mask 是为有bounding box,但里没有任何一个可见的关节点的人设置的一个东西
        # bg_mask 的用途是用作背景层的处理.正常情况下背景只是在关节点处为0,
        # 但是如果一个人光有标注框但是没有keypoints信息的话那么这个人整个标注框都是0
        keypoint_sets, bg_mask, valid_area = rescaler(anns, width_height_original)
        self.log.debug('valid area: %s, paf min size = %d', valid_area, self.min_size)

        f = PafGenerator(self.min_size, self.skeleton,
                         fixed_size=self.fixed_size, aspect_ratio=self.aspect_ratio)
        f.init_fields(bg_mask)
        f.fill(keypoint_sets)
        return f.fields(valid_area)


class PafGenerator(object):
    def __init__(self, min_size, skeleton, *,
                 v_threshold=0, padding=10, fixed_size=False, aspect_ratio=0.0):
        self.min_size = min_size  # 默认是3
        self.skeleton = skeleton  # 骨架连接信息的二维array
        self.v_threshold = v_threshold
        self.padding = padding  # padding 是 10
        self.fixed_size = fixed_size
        self.aspect_ratio = aspect_ratio

        self.intensities = None
        self.fields_reg1 = None
        self.fields_reg2 = None
        self.fields_scale = None
        self.fields_reg_l = None

    def init_fields(self, bg_mask):
        n_fields = len(self.skeleton)  # 19, 总共19个连接
        field_w = bg_mask.shape[1] + 2 * self.padding  # w + 2 * padding
        field_h = bg_mask.shape[0] + 2 * self.padding  # h + 2 * padding
        # 额外加上一个
        # 注意这里的intensities的维数是(20, h, w),与PIF中(18, h, w)所不同
        self.intensities = np.zeros((n_fields + 1, field_h, field_w), dtype=np.float32)
        # 公式中的 x1, y1
        self.fields_reg1 = np.zeros((n_fields, 2, field_h, field_w), dtype=np.float32)
        # 公式中的 x2, y2
        self.fields_reg2 = np.zeros((n_fields, 2, field_h, field_w), dtype=np.float32)
        # 也是用来标识人的大小的便令
        self.fields_scale = np.zeros((n_fields, field_h, field_w), dtype=np.float32)
        # todo 猜测这个变量同时涵盖了b1与b2
        self.fields_reg_l = np.full((n_fields, field_h, field_w), np.inf, dtype=np.float32)

        # set background
        self.intensities[-1] = 1.0
        self.intensities[-1, self.padding:-self.padding, self.padding:-self.padding] = bg_mask
        self.intensities[-1] = scipy.ndimage.binary_erosion(self.intensities[-1],
                                                            iterations=int(self.min_size / 2.0) + 1,
                                                            border_value=1)

    def fill(self, keypoint_sets):
        for keypoints in keypoint_sets:
            # keypoints的维数是(17, 3)
            self.fill_keypoints(keypoints)

    def fill_keypoints(self, keypoints):
        visible = keypoints[:, 2] > 0
        if not np.any(visible):
            return

        area = (
                (np.max(keypoints[visible, 0]) - np.min(keypoints[visible, 0])) *
                (np.max(keypoints[visible, 1]) - np.min(keypoints[visible, 1]))
        )
        scale = np.sqrt(area)
        # 这步之前的操作与PIF中的相同

        # i是索引(0-18),joint1i是开始的关节,joint2i是需要连接的关节,两者都是数字表示
        for i, (joint1i, joint2i) in enumerate(self.skeleton):
            # keypoints的维数是(17, 3)
            joint1 = keypoints[joint1i - 1]  # 开始关节的(x, y, v)
            joint2 = keypoints[joint2i - 1]  # 要连接的关节的(x, y, v)
            if joint1[2] <= self.v_threshold or joint2[2] <= self.v_threshold:
                # 这个地方基本都满足,直接看下面就行
                continue
            self.fill_association(i, joint1, joint2, scale)

    def fill_association(self, i, joint1, joint2, scale):
        # offset between joints  两个关节位置相减得到的向量
        offset = joint2[:2] - joint1[:2]
        offset_d = np.linalg.norm(offset)  # 求得关节间臂的长度

        # dynamically create s
        # min_size 默认为3
        s = max(self.min_size, int(offset_d * self.aspect_ratio))
        sink = create_sink(s)  # 创建偏移矩阵 shape为(2, s, s)
        s_offset = (s - 1.0) / 2.0  # 和前面一样的,这里的s相当于之前的side_length

        # pixel coordinates of top-left joint pixel
        joint1ij = np.round(joint1[:2] - s_offset)
        joint2ij = np.round(joint2[:2] - s_offset)
        offsetij = joint2ij - joint1ij  # 可能比offset更准确一些,其实差不多

        # set fields
        # 动态地设置中间点的个数
        num = max(2, int(np.ceil(offset_d)))
        # np.spacing(1) = 2.2204e-16
        fmargin = min(0.4, (s_offset + 1) / (offset_d + np.spacing(1)))
        # fmargin = 0.0
        frange = np.linspace(fmargin, 1.0 - fmargin, num=num)
        if self.fixed_size:  # fixed_size default 是 False
            frange = [0.5]
        for f in frange:
            fij = np.round(joint1ij + f * offsetij) + self.padding
            # 以fij为起点，长和宽为s的区域
            fminx, fminy = int(fij[0]), int(fij[1])   # 这里的fminx是加了padding后的坐标
            fmaxx, fmaxy = fminx + s, fminy + s
            if fminx < 0 or fmaxx > self.intensities.shape[2] or \
                    fminy < 0 or fmaxy > self.intensities.shape[1]:
                continue
            fxy = (fij - self.padding) + s_offset  # 对应的真实的xy坐标,≈joint1[:2] + f * offsetij

            # precise floating point offset of sinks
            joint1_offset = (joint1[:2] - fxy).reshape(2, 1, 1)  # 与 joint1的偏移,≈ f * offsetij
            joint2_offset = (joint2[:2] - fxy).reshape(2, 1, 1)  # 与 joint2的偏移

            # update intensity
            self.intensities[i, fminy:fmaxy, fminx:fmaxx] = 1.0

            # update background
            self.intensities[-1, fminy:fmaxy, fminx:fmaxx] = 0.0

            # update regressions
            # sink 是从大到小的！这样就与论文中给出的数据相同了，向量相加，这里的sink创建时的side_length 为 s
            # 也就是以fxy为中心,大小为s的一个范围内的所有点都被标记了
            # fields_reg1中保存了这个范围内的点到起始关节点的向量,同理fields_reg2中保存了到终止关节点的向量
            # sink_l中保存的是距离 起始/终止 关节点中近的那个的距离
            sink1 = sink + joint1_offset
            sink2 = sink + joint2_offset
            # 这步蕴含了谁近就是谁的思想(但是仅限于开始的关节点与结束的关节点之间)
            sink_l = np.minimum(np.linalg.norm(sink1, axis=0),
                                np.linalg.norm(sink2, axis=0))
            mask = sink_l < self.fields_reg_l[i, fminy:fmaxy, fminx:fmaxx]
            # reg1放的是点到joint1的向量
            self.fields_reg1[i, :, fminy:fmaxy, fminx:fmaxx][:, mask] = \
                sink1[:, mask]
            # reg2放的是点到joint2的向量
            self.fields_reg2[i, :, fminy:fmaxy, fminx:fmaxx][:, mask] = \
                sink2[:, mask]
            # fields_reg_l放的是到最近关节点的距离
            self.fields_reg_l[i, fminy:fmaxy, fminx:fmaxx][mask] = sink_l[mask]

            # update scale
            self.fields_scale[i, fminy:fmaxy, fminx:fmaxx][mask] = scale

    def fields(self, valid_area):
        intensities = self.intensities[:, self.padding:-self.padding, self.padding:-self.padding]
        fields_reg1 = self.fields_reg1[:, :, self.padding:-self.padding, self.padding:-self.padding]
        fields_reg2 = self.fields_reg2[:, :, self.padding:-self.padding, self.padding:-self.padding]
        fields_scale = self.fields_scale[:, self.padding:-self.padding, self.padding:-self.padding]

        intensities = mask_valid_area(intensities, valid_area)

        return (
            torch.from_numpy(intensities),  # 论文中的c,(20, h, w)
            torch.from_numpy(fields_reg1),  # 论文中的x1,y1,(19, 2, h, w)
            torch.from_numpy(fields_reg2),  # 论文中的x2,y2,(19, 2, h, w)
            torch.from_numpy(fields_scale),  # scale只与keypoints的面积有关,所有的关节之间都一样
            # 注意这里同样没有返回 fields_reg_l
        )
