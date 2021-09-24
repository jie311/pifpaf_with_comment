import functools
import math
import numpy as np


@functools.lru_cache(maxsize=64)
# lru_cache 的参数 maxsize 代表能缓存几个函数执行的结果
def create_sink(side):
    if side == 1:
        return np.zeros((2, 1, 1))
    # side 为 4 时,sink1d = [ 1.5  0.5 -0.5 -1.5]
    sink1d = np.linspace((side - 1.0) / 2.0, -(side - 1.0) / 2.0, num=side, dtype=float)
    sink = np.stack((
        sink1d.reshape(1, -1).repeat(side, axis=0),  # 重复四遍 (4, 4)
        sink1d.reshape(-1, 1).repeat(side, axis=1),  # 转置后重复四遍 (4, 4)
    ), axis=0)
    # stack后shape为(2, side, side),当side为4时为(2, 4, 4)
    return sink


# intensities:(18, h, w)
def mask_valid_area(intensities, valid_area):
    if valid_area is None:
        return intensities
    # 之前的操作已经将intensities中的关节的部分标注成了1
    # 猜测valid_area = [x1, y1, w, h]
    # 下面的4步操作是将所有valid_area之外的地方变成 0
    intensities[:, :int(valid_area[1]), :] = 0  # 第一步
    intensities[:, :, :int(valid_area[0])] = 0  # 第二步
    max_i = int(math.ceil(valid_area[1] + valid_area[3]))
    max_j = int(math.ceil(valid_area[0] + valid_area[2]))
    if max_i < intensities.shape[1]:
        intensities[:, max_i:, :] = 0  # 第三步
    if max_j < intensities.shape[2]:
        intensities[:, :, max_j:] = 0  # 第四步

    return intensities
