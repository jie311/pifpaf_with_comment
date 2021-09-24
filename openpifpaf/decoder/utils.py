"""Utilities for decoders."""

import numpy as np


# 整个方法的目的是返回一个索引矩阵,索引矩阵的维数为(num, shape),num表示shape是几维的
# 使用flip的目的是将x,y坐标进行颠倒
# 即颠倒后顺序变为(0,0)、(1,0)、(2,0)、(3,0)
def index_field(shape):
    # np.indices 返回一个对应 shape的切片数组,例如shape为(2,3)时等价于 x[:2,:3]
    yx = np.indices(shape, dtype=float)
    # np.filp axis=0 时将每一行看作一个整体,行的顺序发生颠倒
    xy = np.flip(yx, axis=0)
    return xy


def weiszfeld_nd(x, init_y, weights=None, epsilon=1e-8, max_steps=20):
    """Weighted Weiszfeld step."""
    if weights is None:
        weights = np.ones(x.shape[0])
    weights = np.expand_dims(weights, -1)
    weights_x = weights * x

    y = init_y
    for _ in range(max_steps):
        prev_y = y

        denom = np.linalg.norm(x - prev_y, axis=-1, keepdims=True) + epsilon
        y = (
                np.sum(weights_x / denom, axis=0) /
                np.sum(weights / denom, axis=0)
        )
        if np.sum(np.abs(prev_y - y)) < 1e-2:
            return y, denom

    return y, denom


def sparse_bilinear_kernel(coord, value):
    l = coord.astype(int)
    g = np.meshgrid(*((ll, ll + 1) for ll in l))
    g = list(zip(*(gg.reshape(-1) for gg in g)))

    v = [np.prod(1.0 - np.abs(coord - corner)) * value for corner in g]
    return g, v


class Sparse2DGaussianField(object):
    def __init__(self, data=None, nearest_neighbors=25):
        if data is None:
            data = np.zeros((0, 3))

        self.nearest_neighbors = nearest_neighbors
        self.data = data

    def value(self, xy, sigma):
        mask = np.logical_and(
            np.logical_and(self.data[0] > xy[0] - 2 * sigma,
                           self.data[0] < xy[0] + 2 * sigma),
            np.logical_and(self.data[1] > xy[1] - 2 * sigma,
                           self.data[1] < xy[1] + 2 * sigma),
        )
        diff = np.expand_dims(xy, -1) - self.data[:2, mask]
        if diff.shape[1] == 0:
            return 0.0

        gauss_1d = np.exp(-0.5 * diff ** 2 / sigma ** 2)
        gauss = np.prod(gauss_1d, axis=0)

        v = np.sum(gauss * self.data[2, mask])
        return np.tanh(v * 3.0 / self.nearest_neighbors)

    def values(self, xys, sigmas):
        assert xys.shape[-1] == 2
        if xys.shape[0] == 0:
            return np.zeros((0,))

        if isinstance(sigmas, float):
            sigmas = np.full((xys.shape[0],), sigmas)
        if hasattr(sigmas, 'shape') and sigmas.shape[0] == 1 and xys.shape[0] > 1:
            sigmas = np.full((xys.shape[0],), sigmas[0])

        return np.stack([self.value(xy, sigma) for xy, sigma in zip(xys, sigmas)])


def normalize_paf(intensity_fields, j1_fields, j2_fields, j1_fields_logb, j2_fields_logb, *, fixed_b=None):
    # (19, output_h, output_w) -> (19, 1, output_h, output_w)
    intensity_fields = np.expand_dims(intensity_fields, 1)
    # 对两个 logb 先做exp,然后再扩维,两个都是 [19, 1, output_h, output_w]
    j1_fields_b = np.expand_dims(np.exp(j1_fields_logb), 1)
    j2_fields_b = np.expand_dims(np.exp(j2_fields_logb), 1)
    if fixed_b:
        j1_fields_b = np.full_like(j1_fields_b, fixed_b)
        j2_fields_b = np.full_like(j2_fields_b, fixed_b)
    # j1_fields 的shape是(19, 2, h, w), index_fields 的shape是(2, h, w)
    index_fields = index_field(j1_fields[0, 0].shape)
    # (2, h, w) -> (1, 2, h, w)
    index_fields = np.expand_dims(index_fields, 0)
    # j1_fields3的shape为(19, 4, h, w) 4为 [confidence, x, y, spread b]
    # todo intensity_fields表示这个点在关节上的置信度
    #      index_fields + j1_fields表示离这个点连接的第1个关节点的坐标
    #      index_fields + j2_fields表示离这个点连接的第2个关节点的坐标
    #      j1_fields_b是用来计算loss的
    j1_fields3 = np.concatenate((intensity_fields, index_fields + j1_fields, j1_fields_b), axis=1)
    j2_fields3 = np.concatenate((intensity_fields, index_fields + j2_fields, j2_fields_b), axis=1)
    # (19, 2, 4, h, w)
    paf = np.stack((j1_fields3, j2_fields3), axis=1)
    return paf


def normalize_pif(joint_intensity_fields, joint_fields, _, scale_fields, *, fixed_scale=None):
    # todo 这里是17还是18存疑,单从下面的concatenate操作可以看出应该是17
    # joint_intensity_fields:(17, h, w)  ->  (17, 1, h, w)
    joint_intensity_fields = np.expand_dims(joint_intensity_fields.copy(), 1)
    # scale_fields: (17, h, w)  ->  (17, 1, h, w)
    scale_fields = np.expand_dims(scale_fields, 1)
    if fixed_scale is not None:
        scale_fields[:] = fixed_scale
    # 整个方法的目的是返回一个(2, h, w)大小的索引矩阵, 其中x, y坐标是颠倒的
    index_fields = index_field(joint_fields.shape[-2:])
    # (2, h, w) -> (1, 2, h, w)
    index_fields = np.expand_dims(index_fields, 0)
    # joint_fields 据说就是 joint_offset_fields,维数是(17, 2, h, w)
    # (17, 2, h, w) + (1, 2, h, w) = (17, 2, h, w)
    # 这里的位置索引信息加上了偏移向量
    # 例如index_fields处的值为(1, 2), 这个地方对应的joint_fields值为(3, 4),那么
    # 意思就是(4, 6)处是关键点的坐标值,即(1, 2)加上偏置(3, 4),表示离(1, 2)位置最近的关节点的坐标是(4, 6)
    # todo 原先的每个点代表的偏移向量的信息,这么操作完后每个点的值为离这个点最近的关节点的坐标
    joint_fields = index_fields + joint_fields
    # 维数为(17, 4, h, w), 分别为 [confidence, x, y, scale]
    return np.concatenate((joint_intensity_fields, joint_fields, scale_fields), axis=1, )


def normalize_pifs(joint_intensity_fields, joint_fields, scale_fields, *, fixed_scale=None):
    joint_intensity_fields = np.expand_dims(joint_intensity_fields.copy(), 1)
    scale_fields = np.expand_dims(scale_fields, 1)
    if fixed_scale is not None:
        scale_fields[:] = fixed_scale

    index_fields = index_field(joint_fields.shape[-2:])
    index_fields = np.expand_dims(index_fields, 0)
    joint_fields = index_fields + joint_fields

    return np.concatenate(
        (joint_intensity_fields, joint_fields, scale_fields),
        axis=1,
    )


# 将 (x,y) 为中心，width 为方形半径的所有值添加一个value值
def scalar_square_add_single(field, x, y, width, value):
    minx = max(0, int(round(x - width)))
    miny = max(0, int(round(y - width)))
    maxx = max(minx + 1, min(field.shape[1], int(round(x + width)) + 1))
    maxy = max(miny + 1, min(field.shape[0], int(round(y + width)) + 1))
    field[miny:maxy, minx:maxx] += value
