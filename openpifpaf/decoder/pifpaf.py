"""Decoder for pif-paf fields."""

from collections import defaultdict
import logging
import time

import numpy as np

from .annotation import Annotation
from .decoder import Decoder
from .utils import (index_field, scalar_square_add_single,
                    normalize_pif, normalize_paf)
from ..data import KINEMATIC_TREE_SKELETON, COCO_PERSON_SKELETON, DENSER_COCO_PERSON_SKELETON

# pylint: disable=import-error
from openpifpaf.functional import (scalar_square_add_constant, scalar_square_add_gauss,
                                   weiszfeld_nd, paf_center)


class PifPaf(Decoder):
    default_force_complete = True
    default_connection_method = 'max'
    default_fixed_b = None
    default_pif_fixed_scale = None

    def __init__(self, stride, *,
                 seed_threshold=0.2,
                 head_names=None,
                 head_indices=None,
                 skeleton=None,
                 debug_visualizer=None,
                 **kwargs):
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.debug('unused arguments %s', kwargs)

        if head_names is None:
            head_names = ('pif', 'paf')

        self.head_indices = head_indices
        if self.head_indices is None:
            self.head_indices = {
                ('paf', 'pif', 'paf'): [1, 2],
                ('pif', 'pif', 'paf'): [1, 2],
            }.get(head_names, [0, 1])

        self.skeleton = skeleton
        if self.skeleton is None:
            paf_name = head_names[self.head_indices[1]]
            if paf_name == 'paf16':
                self.skeleton = KINEMATIC_TREE_SKELETON
            elif paf_name == 'paf44':
                self.skeleton = DENSER_COCO_PERSON_SKELETON
            else:
                self.skeleton = COCO_PERSON_SKELETON

        self.stride = stride
        self.hr_scale = self.stride
        self.seed_threshold = seed_threshold
        self.debug_visualizer = debug_visualizer
        self.force_complete = self.default_force_complete
        self.connection_method = self.default_connection_method
        self.fixed_b = self.default_fixed_b
        self.pif_fixed_scale = self.default_pif_fixed_scale

        self.pif_nn = 16
        self.paf_nn = 1 if self.connection_method == 'max' else 35

    @staticmethod
    def match(head_names):
        return head_names in (
            ('pif', 'paf'),
            ('pif', 'paf44'),
            ('pif', 'paf16'),
            ('paf', 'pif', 'paf'),
            ('pif', 'pif', 'paf'),
            ('pif', 'wpaf'),
        )

    @classmethod
    def cli(cls, parser):
        group = parser.add_argument_group('PifPaf decoder')
        group.add_argument('--fixed-b', default=None, type=float,
                           help='overwrite b with fixed value, e.g. 0.5')
        group.add_argument('--pif-fixed-scale', default=None, type=float,
                           help='overwrite pif scale with a fixed value')
        group.add_argument('--connection-method',
                           default='max', choices=('median', 'max'),
                           help='connection method to use, max is faster')

    @classmethod
    def apply_args(cls, args):
        cls.default_fixed_b = args.fixed_b
        cls.default_pif_fixed_scale = args.pif_fixed_scale
        cls.default_connection_method = args.connection_method
        # arg defined in factory
        cls.default_force_complete = args.force_complete_pose

    def __call__(self, fields):
        start = time.time()
        pif, paf = fields[self.head_indices[0]], fields[self.head_indices[1]]
        if self.debug_visualizer:
            self.debug_visualizer.pif_raw(pif, self.stride)
            self.debug_visualizer.paf_raw(paf, self.stride, reg_components=3)
        paf = normalize_paf(*paf, fixed_b=self.fixed_b)
        pif = normalize_pif(*pif, fixed_scale=self.pif_fixed_scale)

        gen = PifPafGenerator(
            pif, paf,
            stride=self.stride,
            seed_threshold=self.seed_threshold,
            connection_method=self.connection_method,
            pif_nn=self.pif_nn,
            paf_nn=self.paf_nn,
            skeleton=self.skeleton,
            debug_visualizer=self.debug_visualizer,
        )

        annotations = gen.annotations()
        if self.force_complete:
            # 是否强行连接完(只根据pif其实就可以连接完了)
            annotations = gen.complete_annotations(annotations)

        print('annotations', len(annotations), time.time() - start)
        return annotations


class PifPafGenerator(object):
    def __init__(self, pifs_field, pafs_field, *,
                 stride,
                 seed_threshold,
                 connection_method,
                 pif_nn,
                 paf_nn,
                 skeleton,
                 debug_visualizer=None):
        self.pif = pifs_field
        self.paf = pafs_field

        self.stride = stride
        self.seed_threshold = seed_threshold
        self.connection_method = connection_method
        self.pif_nn = pif_nn
        self.paf_nn = paf_nn
        self.skeleton = skeleton

        self.debug_visualizer = debug_visualizer
        self.timers = defaultdict(float)

        # pif init
        self._pifhr, self._pifhr_scales = self._target_intensities()
        self._pifhr_core = self._target_intensities(core_only=True)
        if self.debug_visualizer:
            self.debug_visualizer.pifhr(self._pifhr)
            self.debug_visualizer.pifhr(self._pifhr_core)

        # paf init
        self._paf_forward = None
        self._paf_backward = None
        self._paf_forward, self._paf_backward = self._score_paf_target()

    def _target_intensities(self, v_th=0.1, core_only=False):
        start = time.time()
        # pif的维度可能是之前在utils中那个,(17, 4, h, w),分别为[confidence, x, y, scale]
        # 这里注意对(h, w)进行了细化,变成了(h * stride, w * stride),即得到了论文中的 high resolution part confidence map
        targets = np.zeros((self.pif.shape[0],
                            int(self.pif.shape[2] * self.stride),
                            int(self.pif.shape[3] * self.stride)))
        scales = np.zeros(targets.shape)  # (17, h * stride, w * stride)
        ns = np.zeros(targets.shape)  # (17, h * stride, w * stride)
        for t, p, scale, n in zip(targets, self.pif, scales, ns):
            # t->(h*stride, w*stride), p->(4, h*str, w*str), scale->(h*str, w*str),  n->(h*str, w*str)
            # targets,                 self.pif,             scales,                 ns
            # todo 这里的stride是细化的倍数?在这里设置为了8
            # v, x, y, s分别为confidence, x, y, scale
            v, x, y, s = p[:, p[0] > v_th]
            x = x * self.stride  # x
            y = y * self.stride  # y
            s = s * self.stride  # scale
            if core_only:
                scalar_square_add_gauss(t, x, y, s, v / self.pif_nn, truncate=0.5)
            else:
                #  self.pif_nn这里是16,原因可能是算的是17个关节在这个位置上的累加,所以除16?
                #  还有一个地方需要注意,虽然下面两种函数中都传入了s,但是使用的方法却不一样
                #  在constant中s直接作为width,而在gauss中s经过了2步处理才作为width
                # (double[:, :] field, x_np, y_np, sigma_np, v_np, double truncate=2.0)
                # t是target,x,y是坐标,s是scale
                # 以x,y为中心,半径为0.5 * s 的方形区域内的点置信度加上一个高斯核,以达到high resolution的目的
                scalar_square_add_gauss(t, x, y, s, v / self.pif_nn)
                # (double[:, :] field, x_np, y_np, width_np, double[:] v)
                # 以x,y为中心,半径为 s 的方形区域内的点scale加上s * v
                scalar_square_add_constant(scale, x, y, s, s * v)
                # 以x,y为中心,半径为 s 的方形区域内的点n加上v
                scalar_square_add_constant(n, x, y, s, v)

        if core_only:
            print('target_intensities', time.time() - start)
            return targets

        m = ns > 0
        scales[m] = scales[m] / ns[m]  #
        print('target_intensities', time.time() - start)
        # 这里需要注意targets的shape变为了(17, h * stride, w * stride)
        return targets, scales

    def _score_paf_target(self, pifhr_floor=0.01, score_th=0.1):
        start = time.time()

        scored_forward = []
        scored_backward = []
        # self.paf 的shape为(19, 2, 4, h, w),其中2表示两个关节点,4为[confidence,x,y,logb]
        for c, fourds in enumerate(self.paf):
            # fourds的shape为(2, 4, h, w),4为[confidence,x,y,logb]
            assert fourds.shape[0] == 2
            assert fourds.shape[1] == 4
            # 函数的目标是找出两个关节中的confidence最小值
            # scores的shape是(h, w),其中每个位置放的都是两个关节中较小的confidence的值
            scores = np.min(fourds[:, 0], axis=0)  # todo 难道两个置信度不一样么,为何要求最小?可能训练出来的不一样
            mask = scores > score_th  # 0.1
            # 假设h*w个confidence中>score_th的点的个数为num个,则scores的维数为(num,)
            scores = scores[mask]
            # fourds的shape为(2, 4, num),得到的是一个所有元素值都 >score_th 的矩阵
            # 其中2表示两个关节点,4为[confidence,x,y,logb]
            fourds = fourds[:, :, mask]

            # c 是索引,从0到18,j1i表示的是应当连接在一起的关节点对中的起始关节点的真实索引值
            j1i = self.skeleton[c][0] - 1
            if pifhr_floor < 1.0:  # pifhr_floor = 0.01
                # ij_b的shape为(2, h, w),其中经过了stride的细化
                ij_b = np.round(fourds[0, 1:3] * self.stride).astype(np.int)
                ij_b[0] = np.clip(ij_b[0], 0, self._pifhr.shape[2] - 1)  # w * stride
                ij_b[1] = np.clip(ij_b[1], 0, self._pifhr.shape[1] - 1)  # h * stride
                # _pifhr的shape为(17, h * stride, w * stride)
                # 这个 pifhr_b 就是joint1对应的在pifhr上的confidence值
                pifhr_b = self._pifhr[j1i, ij_b[1], ij_b[0]]
                # 根据confidence对scores进行更新,todo pifhr_floor=0.01或0.9,猜测这里就是对scores进行一个微调
                scores_b = scores * (pifhr_floor + (1.0 - pifhr_floor) * pifhr_b)
            else:
                scores_b = scores
            mask_b = scores_b > score_th  # 0.1
            # concatenate后shape为(7, num2)
            # 7表示[score_b, joint2_x, joint2_y, joint2_b, joint1_x, joint1_y, joint1_b]
            scored_backward.append(np.concatenate((
                # scores_b的维数为(num2,),扩维后为(1, num2)
                np.expand_dims(scores_b[mask_b], 0),  # (1, num2)
                # fourds的shape为(2, 4, h, w)
                # 注意下面的fourds[1]与[0]的顺序,这决定了是backward还是forward
                # (2, 4, num) -> (3, num2)
                fourds[1, 1:4][:, mask_b],  # (3, num2), 3指的是[x, y, logb],x,y表示的是这个点连接的第2个关节点的坐标
                fourds[0, 1:4][:, mask_b],  # (3, num2)
            )))

            j2i = self.skeleton[c][1] - 1
            if pifhr_floor < 1.0:
                ij_f = np.round(fourds[1, 1:3] * self.stride).astype(np.int)
                ij_f[0] = np.clip(ij_f[0], 0, self._pifhr.shape[2] - 1)
                ij_f[1] = np.clip(ij_f[1], 0, self._pifhr.shape[1] - 1)
                pifhr_f = self._pifhr[j2i, ij_f[1], ij_f[0]]
                scores_f = scores * (pifhr_floor + (1.0 - pifhr_floor) * pifhr_f)
            else:
                scores_f = scores
            mask_f = scores_f > score_th
            # concatenate后shape为(7, num3)
            scored_forward.append(np.concatenate((
                np.expand_dims(scores_f[mask_f], 0),  # (1, num3)
                fourds[0, 1:4][:, mask_f],  # (3, num3), 3指的是[x, y, logb]
                fourds[1, 1:4][:, mask_f],  # (3, num3)
            )))

        print('scored paf', time.time() - start)
        # 因为总共要进行19次循环,因此
        # scored_forward的shape为(19, 7, num3)
        # scored_backward的shape为(19, 7, num2)
        return scored_forward, scored_backward

    def annotations(self):
        start = time.time()

        # seeds的shape为 (num, 4)其中num为前面选中的seed的数目,4为(confidence, channel号, x, y)
        # 其中channel号就是关节的序号  x,y是没有经过high resolution的坐标
        seeds = self._pifhr_seeds()  # (n, 4)

        occupied = np.zeros(self._pifhr_scales.shape)  # (17, h * stride, w * stride)
        annotations = []

        for v, f, x, y in seeds:
            # v, f, x, y 分别为 confidence, channel号, x, y
            i = np.clip(int(round(x * self.stride)), 0, occupied.shape[2] - 1)  # hr
            j = np.clip(int(round(y * self.stride)), 0, occupied.shape[1] - 1)
            if occupied[f, j, i]:  # occupied 是新创建的一个
                continue

            # f:channel index   (x, y, v):x,y,confidence     self.skeleton:COCO_PERSON_SKELETON
            ann = Annotation(f, (x, y, v), self.skeleton)  # 给一个人的骨架加入第一个seed点,接着开始grow
            self._grow(ann, self._paf_forward, self._paf_backward)
            ann.fill_joint_scales(self._pifhr_scales, self.stride)
            annotations.append(ann)

            for i, xyv in enumerate(ann.data):
                if xyv[2] == 0.0:
                    continue
                # 在high resolution图像上+1,表示这个关节附近已经被标记过了,不能在这附近找了
                width = ann.joint_scales[i] * self.stride
                scalar_square_add_single(occupied[i],
                                         xyv[0] * self.stride,
                                         xyv[1] * self.stride,
                                         width / 2.0,
                                         1.0)

        if self.debug_visualizer:
            print('occupied annotations field 0')
            self.debug_visualizer.occupied(occupied[0])

        print('keypoint sets', len(annotations), time.time() - start)
        return annotations

    # 函数的作用:对每一个关节点的channel进行循环,按照confidence的值从大到小进行遍历,选出confidence>0.2的点作为连接过程的seed
    # 其中每选择一个点后就把它周围width/2的方形区域内的所有点设置为不可选,猜测这么做可以防止同一个人的一个关节点选多次
    #  todo 其中这里的width是一个超参数
    def _pifhr_seeds(self):
        start = time.time()
        seeds = []
        # self._pifhr, self._pifhr_scales = self._target_intensities()
        # self._pifhr_core = self._target_intensities(core_only=True)
        # _pifhr_core 的shape为 (17, h * stride, w * stride)
        # _pifhr_scales 的shape为 (17, h * stride, w * stride)
        for field_i, (f, s) in enumerate(zip(self._pifhr_core, self._pifhr_scales)):
            # 这里的field_i的值为 0-16
            # f的shape为(h * stride, w * stride)
            # s的shape为(h * stride, w * stride)
            index_fields = index_field(f.shape)  # (2, h * stride, w * stride)
            # 下面 concatenate函数需要注意两个需要连接的矩阵之间使用()来连接
            candidates = np.concatenate((index_fields, np.expand_dims(f, 0)), 0)  # (3, h * stride, w * stride)

            mask = f > self.seed_threshold  # 这里的 threshold 为 0.2
            # (3, num) -> (num, 3) 这里的num是满足阈值的个数
            candidates = np.moveaxis(candidates[:, mask], 0, -1)  # (num, 3),3分别表示2维的坐标索引与新增的1维

            occupied = np.zeros(s.shape)  # (h * stride, w * stride)
            # 按照c[2] 即 confidence 值来从大到小对数据进行排序
            for c in sorted(candidates, key=lambda c: c[2], reverse=True):
                i, j = int(c[0]), int(c[1])  # i, j 分别表示 x, y
                if occupied[j, i]:
                    continue
                width = max(4, s[j, i])
                # scalar_square_add_single(field, x, y, width, value)
                # 函数作用是将 (x,y) 为中心，width 为方形半径的所有值添加一个value值
                # 这里是将选中的seed点周围的width/2范围内的点都设置为不可选
                scalar_square_add_single(occupied, c[0], c[1], width / 2.0, 1.0)
                # 注意,这里append的seed点坐标不是high resolution的
                seeds.append((c[2], field_i, c[0] / self.stride, c[1] / self.stride))

            if self.debug_visualizer:
                if field_i in self.debug_visualizer.pif_indices:
                    print('occupied seed, field {}'.format(field_i))
                    self.debug_visualizer.occupied(occupied)

        seeds = list(sorted(seeds, reverse=True))  # 从大到小排列
        # seeds的shape为 (num, 4)其中num为前面选中的seed的数目,4为(confidence, channel号, x, y)
        # 其中channel号就是关节的序号  x,y是没有经过high resolution的坐标
        if len(seeds) > 500:
            # 这里seeds的选取策略是 把confidence>0.1的点都选上,如果满足条件的点不足500个的话那么就选500个
            if seeds[500][0] > 0.1:  # todo 不知道这个限定条件有啥用啊,seed的threshold不都0.2了么,这个条件一定满足吧
                seeds = [s for s in seeds if s[0] > 0.1]
            else:
                seeds = seeds[:500]

        if self.debug_visualizer:
            self.debug_visualizer.seeds(seeds, self.stride)

        print('seeds', len(seeds), time.time() - start)
        # seeds的shape为 (num, 4)其中num为前面选中的seed的数目,4为(confidence, channel号, x, y)
        # 其中channel号就是关节的序号  x,y是没有经过high resolution的坐标
        return seeds

    def _grow_connection(self, xy, paf_field):
        # paf_field(directed_paf_field)的shape是 (7, num3)
        # xy的shape是(2,)
        assert len(xy) == 2
        assert paf_field.shape[0] == 7

        # source value
        # len(xy) = 2，xy[0]=x, xy[1]=y
        # paf_field的shape为(7, num4), 7 为 score, x1 ,y1, logb1, x2, y2, logb2
        # 其中保存是 可以认为和(x,y)相同起点的paf_field信息 todo 即从众多的关节连接中找出可能属于这个人的关节连接
        paf_field = paf_center(paf_field, xy[0], xy[1], sigma=2.0)
        if paf_field.shape[1] == 0:  # 没有找到这个人的关节连接
            # todo 这个地方可以进行改进,即没有找到这个人的关节连接时可以咋办,可以适当地降低阈值,直到找到这么一条连接
            return 0, 0, 0

        # todo 下面的目标是找出来这么多连接中应该采用哪个
        # source distance
        # xy 与 $a_{x1}^{ij}$,$a_{y1}^{ij}$ 求二范数
        # 这个距离的含义是:一个人两个关节点之间可能有很多的关节连接,
        # d求的是由pif推测的关节点与paf推测出的关节点(两者理想情况下应当为同一关节点)间的距离
        d = np.linalg.norm(np.expand_dims(xy, 1) - paf_field[1:3], axis=0)  # 求了个norm,这里shape为(num4,)
        # todo 这里的 * 3.0 是咋回事,公式中并没有*3
        b_source = paf_field[3] * 3.0  # 相当于公式中的a_{b1}^{ij}
        # b_target = paf_field[6]

        # combined value and source distance
        v = paf_field[0]  # confidence，相当于公式中的a_c
        scores = np.exp(-1.0 * d / b_source) * v  # two-tailed cumulative Laplace   (num4,)

        if self.connection_method == 'median':
            # 传入的是19个关节对应的$a_{x2}^{ij}$,$a_{y2}^{ij}$
            # scores 是公式(3)中的前半部分(f2之前)得到的结果,shape为(num4,)
            return self._target_with_median(paf_field[4:6], scores, sigma=1.0)
        if self.connection_method == 'max':
            # 直接返回 score 最大的 paf
            return self._target_with_maxscore(paf_field[4:7], scores)
        raise Exception('connection method not known')

    # greedy decoding 的代码
    # 传入的target_coordinates是$a_{x2}^{ij}$,$a_{y2}^{ij}$, 维数可能是 2*num4
    # scores 是公式(3)中的前半部分(f2之前)得到的结果,维数是(num4,)
    def _target_with_median(self, target_coordinates, scores, sigma, max_steps=20):
        target_coordinates = np.moveaxis(target_coordinates, 0, -1)  # 2*num4 -> num4*2
        assert target_coordinates.shape[0] == scores.shape[0]  # scores -> (num4,)

        if target_coordinates.shape[0] == 1:
            # 如果只有一个候选者,那么直接返回,这里需要注意下score的表达形式,相当于论文中的f2
            return (target_coordinates[0][0],
                    target_coordinates[0][1],
                    np.tanh(scores[0] * 3.0 / self.paf_nn))
        # sum((num4,2) * (num4,1)) = sum(num4, 2) = (2,)
        # todo 直接乘起来了,然后除以了scores,相当于一个加权求和,求了所有候选坐标的几何中点
        y = np.sum(target_coordinates * np.expand_dims(scores, -1), axis=0) / np.sum(scores)
        if target_coordinates.shape[0] == 2:  # (num4, 2)
            # 如果只有两个候选者,那么返回y的坐标
            return y[0], y[1], np.tanh(np.sum(scores) * 3.0 / self.paf_nn)

        # target_coordinates:(num4, 2)   y:(2,)  scores:(num4,)
        #                       (x_np, y_np, double[:] weights=None)
        y, prev_d = weiszfeld_nd(target_coordinates, y, weights=scores, max_steps=max_steps)

        closest = prev_d < sigma  # sigma default为1
        close_scores = np.sort(scores[closest])[-self.paf_nn:]  # paf_nn为1
        score = np.tanh(np.sum(close_scores) * 3.0 / self.paf_nn)
        return (y[0], y[1], score)

    @staticmethod
    # 传入的target_coordinates是$a_{x2}^{ij}$,$a_{y2}^{ij}$,$a_{b2}^{ij}$ 维数是 3*num4
    # scores 是公式(3)中的前半部分(f2之前)得到的结果,维数是(num4,)
    # 函数的作用是直接选择得分最大的连接作为最终连接
    def _target_with_maxscore(target_coordinates, scores):
        assert target_coordinates.shape[1] == scores.shape[0]

        max_i = np.argmax(scores)  # 返回最大score的索引
        max_entry = target_coordinates[:, max_i]

        score = scores[max_i]
        return max_entry[0], max_entry[1], score

    def _grow(self, ann, paf_forward, paf_backward, th=0.1):
        # scored_forward 的shape为(19, 7, num3)
        # scored_backward 的shape为(19, 7, num2)
        # 当前选择的点是当前已有的ann.data里, confidence值最高的那个点应该连接的点的信息
        # _ : confidence of origin, i : connection index,  forward : forward?,
        # j1i : joint index 1,  (not corrected for forward) j2i : joint index 2,  (not corrected for forward)
        for _, i, forward, j1i, j2i in ann.frontier_iter():  # 循环的次数讲道理应该 <= 19
            if forward:
                # 如果forward为True表示已有点是joint1, 需要放进去的这个点是joint2
                xyv = ann.data[j1i]
                # i 是 connection index,是骨架中的第几个连接
                # paf_forward的shape为(19, 7, num3)
                # forward中7表示[score_b, joint1_x, joint1_y, joint1_b, joint2_x, joint2_y, joint2_b]
                # joint1_x, joint1_y表示这个点连接的第一个关节点的坐标,score_b表示这个点在连接线上的得分
                directed_paf_field = paf_forward[i]
                directed_paf_field_reverse = paf_backward[i]
            else:
                # 已有点是joint2, 需要放进去的这个点是joint1
                xyv = ann.data[j2i]
                # backward中7表示[score_b, joint2_x, joint2_y, joint2_b, joint1_x, joint1_y, joint1_b]
                directed_paf_field = paf_backward[i]
                directed_paf_field_reverse = paf_forward[i]

            # todo 这里放进去的directed_paf_field维度为(7, num)表示有很多人都有这个 关节连接,
            #  而且同一个人的这个关节连接表示方法可能不止一条
            # xyv[:2]是当前选中的关节点,需要给它来建立连接
            # todo 我们需要从所有的num条中选择 这个人的且置信度最高的 一条
            #  猜测是需要比对xyv的坐标与directed_paf_field中joint1的坐标
            # 这里得到的new_xyv就是joint2的坐标,shape为(3,),3表示(x2, y2, score)
            # 这里的score对应的是这个connection的得分,也就是论文中的公式(3)得到的得分
            new_xyv = self._grow_connection(xyv[:2], directed_paf_field)
            if new_xyv[2] < th:
                # 得分太低会被舍弃
                continue

            # reverse match 反向匹配,相当于一个验证的过程
            if th >= 0.1:
                # 从joint2往回匹配,看看是否能匹配出来xyv
                reverse_xyv = self._grow_connection(new_xyv[:2], directed_paf_field_reverse)
                if reverse_xyv[2] < th:
                    # 匹配失败,得分太低
                    continue
                if abs(xyv[0] - reverse_xyv[0]) + abs(xyv[1] - reverse_xyv[1]) > 1.0:
                    # 匹配失败,距离xyv太远
                    continue

            # 匹配成功
            new_xyv = (new_xyv[0], new_xyv[1], np.sqrt(new_xyv[2] * xyv[2]))  # geometric mean, score更新为几何均值
            # 将结果保存到data中,成功找到了一个人的应该相连的另一个关节
            if forward:
                if new_xyv[2] > ann.data[j2i, 2]:
                    ann.data[j2i] = new_xyv
            else:
                if new_xyv[2] > ann.data[j1i, 2]:
                    ann.data[j1i] = new_xyv

    def complete_annotations(self, annotations):
        start = time.time()

        paf_forward_c, paf_backward_c = self._score_paf_target(pifhr_floor=0.9, score_th=0.0001)

        for ann in annotations:
            unfilled_mask = ann.data[:, 2] == 0.0
            self._grow(ann, paf_forward_c, paf_backward_c, th=1e-8)
            now_filled_mask = ann.data[:, 2] > 0.0
            updated = np.logical_and(unfilled_mask, now_filled_mask)
            ann.data[updated, 2] = np.minimum(0.001, ann.data[updated, 2])
            ann.fill_joint_scales(self._pifhr_scales, self.stride)

        print('complete annotations', time.time() - start)
        return annotations
