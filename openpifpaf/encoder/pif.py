import logging
import re

import numpy as np
import scipy.ndimage
import torch

from .annrescaler import AnnRescaler
from .encoder import Encoder
from .utils import create_sink, mask_valid_area


# keypoint_sets第1维可能是所有人的个数,第2维可能是17,第3维可能是(x,y,v)
class Pif(Encoder):
    default_side_length = 4

    def __init__(self, head_name, stride, *, n_keypoints=None, **kwargs):
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.debug('unused arguments in %s: %s', head_name, kwargs)

        self.stride = stride
        if n_keypoints is None:
            m = re.match('pif([0-9]+)$', head_name)
            if m is not None:
                n_keypoints = int(m.group(1))
                self.log.debug('using %d keypoints for pif', n_keypoints)
            else:
                n_keypoints = 17
        self.n_keypoints = n_keypoints
        self.side_length = self.default_side_length

    @staticmethod
    def match(head_name):
        return head_name in (
            'pif',
            'ppif',
            'pifb',
            'pifs',
        ) or re.match('pif([0-9]+)$', head_name) is not None

    @classmethod
    def cli(cls, parser):
        group = parser.add_argument_group('pif encoder')
        group.add_argument('--pif-side-length', default=cls.default_side_length, type=int,
                           help='side length of the PIF field')

    @classmethod
    def apply_args(cls, args):
        cls.default_side_length = args.pif_side_length

    def __call__(self, anns, width_height_original):
        rescaler = AnnRescaler(self.stride, self.n_keypoints)
        # 返回缩放后的keypoints_sets，valid area 指的是有效的区域,将来需要学习的区域
        # bg_mask 是为有bounding box,但里没有任何一个可见的关节点的人设置的一个东西
        # bg_mask 的用途是用作背景层的处理.正常情况下背景只是在关节点处为0,
        # 但是如果一个人光有标注框但是没有keypoints信息的话那么这个人整个标注框都是0
        keypoint_sets, bg_mask, valid_area = rescaler(anns, width_height_original)
        self.log.debug('valid area: %s, pif side length = %d', valid_area, self.side_length)
        # keypoint_sets第1维可能是所有人的个数,第2维可能是17,第3维可能是(x,y,v)
        n_fields = keypoint_sets.shape[1]  # 可能是 17
        f = PifGenerator(self.side_length)  # side_length 默认为 4
        # 初始化各个变量
        f.init_fields(n_fields, bg_mask)
        # 将关节点以各自的方法填入各个变量中
        f.fill(keypoint_sets)
        # fields方法目的:将各个变量去掉 padding,并将intensities的valid_area之外的部分都标记为0
        # 最终的返回值为
        # intensities:(18, h, w) 表明每个位置的confidence(0/1)
        # fields_reg:(17, 2, h, w)
        # fields_scale:(17, h, w)
        # 注意这里并没有返回fields_reg_l
        return f.fields(valid_area)


class PifGenerator(object):
    def __init__(self, side_length, v_threshold=0, padding=10):
        self.side_length = side_length
        self.v_threshold = v_threshold
        self.padding = padding

        self.intensities = None
        self.fields_reg = None
        self.fields_scale = None
        self.fields_reg_l = None

        # sink是一个偏移矩阵,shape为(2, side_length, side_length)
        # 2个(side_length, side_length)分别是x方向和y方向的偏移值
        self.sink = create_sink(side_length)
        self.s_offset = (side_length - 1.0) / 2.0  # 1.5,大概是说每个关节点周围几个单位内confidence为1

        self.log = logging.getLogger(self.__class__.__name__)

    # n_fields 为 17
    def init_fields(self, n_fields, bg_mask):
        field_w = bg_mask.shape[1] + 2 * self.padding  # width + 2*padding
        field_h = bg_mask.shape[0] + 2 * self.padding  # height + 2*padding
        # 18 * h * w
        self.intensities = np.zeros((n_fields + 1, field_h, field_w), dtype=np.float32)  # 加上的那层是背景
        # 17 * 2 * h * w , 用来表示每个点到最近的关节点(x, y)的x_offset和y_offset.
        self.fields_reg = np.zeros((n_fields, 2, field_h, field_w), dtype=np.float32)
        # 17 * h * w , 公式中的 sigma,用来表示关节的尺寸
        self.fields_scale = np.zeros((n_fields, field_h, field_w), dtype=np.float32)
        # np.full 构造一个数组，用指定值填充其元素,这里用的是np.inf
        # 17 * h * w , 猜测是公式中的 b,用来计算 loss 的
        self.fields_reg_l = np.full((n_fields, field_h, field_w), np.inf, dtype=np.float32)

        # bg_mask
        self.intensities[-1] = 1.0  # 将最后一层(背景层)设置为全 1
        # 注意之前bg_mask的初始化是全设置为1,这样直接迁移过来就可,记得要减去padding部分
        self.intensities[-1, self.padding:-self.padding, self.padding:-self.padding] = bg_mask
        # iterations:腐蚀重复的次数   borde_value:输出数组中边框的值
        # todo  不知道他这里腐蚀一下是干什么
        self.intensities[-1] = scipy.ndimage.binary_erosion(self.intensities[-1],
                                                            iterations=int(self.s_offset) + 1,  # 2
                                                            border_value=1)

    def fill(self, keypoint_sets):
        for keypoints in keypoint_sets:
            # keypoints_sets的维数是(人数, 17, 3)
            # keypoints的维数是(17, 3)
            self.fill_keypoints(keypoints)

    def fill_keypoints(self, keypoints):
        visible = keypoints[:, 2] > 0
        if not np.any(visible):
            # 如果一个人的所有 keypoints都没有标注,则直接返回
            return

        # 这里keypoints[visible]获得的是一个[x,y,v]的array,要获得x与y必须再通过一层索引
        # area计算的是将所有keypoints都包含进去的最小面积(最大的x间距 * 最大的y间距)
        # 算出来的其实就是人的面积
        area = (
                (np.max(keypoints[visible, 0]) - np.min(keypoints[visible, 0])) *
                (np.max(keypoints[visible, 1]) - np.min(keypoints[visible, 1]))
        )
        scale = np.sqrt(area)  # 开方,对于同一个人的关节点,其scale值是一样的
        self.log.debug('instance scale = %.3f', scale)

        for f, xyv in enumerate(keypoints):
            # f 是序号,xyv就是[x, y, v]
            if xyv[2] <= self.v_threshold:  # v_threshold 默认值是0
                continue

            self.fill_coordinate(f, xyv, scale)

    def fill_coordinate(self, f, xyv, scale):
        # f 是 0-16的序号
        # s_offset = (side_length - 1.0) / 2.0
        # 这里的ij算的是关节辐射范围的最小值
        ij = np.round(xyv[:2] - self.s_offset).astype(np.int) + self.padding
        minx, miny = int(ij[0]), int(ij[1])
        # 分别加上 side_length 算的就是关节辐射方位的最大值
        maxx, maxy = minx + self.side_length, miny + self.side_length
        # intensities的shape为: 18 * h * w
        if minx < 0 or maxx > self.intensities.shape[2] or \
                miny < 0 or maxy > self.intensities.shape[1]:
            return

        offset = xyv[:2] - (ij + self.s_offset - self.padding)  # todo 这是干啥?得到的值应该不是0就是1吧
        offset = offset.reshape(2, 1, 1)

        # update intensity
        # 以关节点为中心,上下左右(side_length - 1.0) / 2.0 范围内的像素点的confidence设置为1
        self.intensities[f, miny:maxy, minx:maxx] = 1.0  # 更新相应的channal中关节的confidence

        # allow unknown margin in background
        self.intensities[-1, miny:maxy, minx:maxx] = 0.0  # 将背景中对应的区域设置为0

        # update regression(将fields_reg对应部分更新为偏移向量)
        sink_reg = self.sink + offset  # (2, side_length, side_length) + (2, 1, 1) = (2, side_length, side_length)
        # 这一步相当于对偏移矩阵中的向量求了个范数,sink_l中的值就是每个位置中向量的模
        sink_l = np.linalg.norm(sink_reg, axis=0)  # 经过 norm 后的维度为(side_length, side_length),范数计算使用的是二范数
        # fields_reg_l: 17 * h * w , 猜测是公式中的 b,用来计算 loss 的
        # fields_reg_l 是全为inf的矩阵,这里注意sink_l与后面的那块维度是相同的,都是(side_length, side_length)
        mask = sink_l < self.fields_reg_l[f, miny:maxy, minx:maxx]  # todo 这里蕴含了离谁近就是哪个关节点的思想
        # 17 * 2 * h * w,取完indice之后变为(2, side_length, side_length)
        self.fields_reg[f, :, miny:maxy, minx:maxx][:, mask] = sink_reg[:, mask]
        # 17 * h * w
        self.fields_reg_l[f, miny:maxy, minx:maxx][mask] = sink_l[mask]

        # update scale
        # 对于同一个人的关节点,其scale值是一样的
        self.fields_scale[f, miny:maxy, minx:maxx][mask] = scale

    def fields(self, valid_area):
        # 方法目的:将各个变量去掉 padding,并将intensities的valid_area之外的部分都标记为0
        intensities = self.intensities[:, self.padding:-self.padding, self.padding:-self.padding]
        fields_reg = self.fields_reg[:, :, self.padding:-self.padding, self.padding:-self.padding]
        fields_scale = self.fields_scale[:, self.padding:-self.padding, self.padding:-self.padding]

        intensities = mask_valid_area(intensities, valid_area)  # 将valid_area之外的部分都标记为0

        return (
            torch.from_numpy(intensities),  # (18, h, w)
            torch.from_numpy(fields_reg),  # (17, 2, h, w)
            torch.from_numpy(fields_scale),  # (17, h, w)
        )
