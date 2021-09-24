# cython: infer_types=True
cimport cython
from libc.math cimport exp, fabs, sqrt
import numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
def scalar_square_add_constant(double[:, :] field, x_np, y_np, width_np, double[:] v):
    # clip 是截取函数,np.clip(a,a_min,a_max,out=None),函数作用是将数组a中的所有数限定到范围a_min和a_max中
    # a：输入矩阵；
    # a_min：被限定的最小值，所有比a_min小的数都会强制变为a_min；
    # a_max：被限定的最大值，所有比a_max大的数都会强制变为a_max；
    # out：可以指定输出矩阵的对象，shape与a相同

    # 猜测
    # x_np 是传入的一组 x 坐标
    # y_np 是传入的一组 y 坐标
    minx_np = np.round(x_np - width_np).astype(np.int)  # 一维数组,下同
    minx_np = np.clip(minx_np, 0, field.shape[1] - 1)
    miny_np = np.round(y_np - width_np).astype(np.int)
    miny_np = np.clip(miny_np, 0, field.shape[0] - 1)
    maxx_np = np.round(x_np + width_np).astype(np.int)
    maxx_np = np.clip(maxx_np + 1, minx_np + 1, field.shape[1])
    maxy_np = np.round(y_np + width_np).astype(np.int)
    maxy_np = np.clip(maxy_np + 1, miny_np + 1, field.shape[0])

    cdef long[:] minx = minx_np
    cdef long[:] miny = miny_np
    cdef long[:] maxx = maxx_np
    cdef long[:] maxy = maxy_np

    cdef Py_ssize_t i, xx, yy

    # for minxx, minyy, maxxx, maxyy, vv in zip(minx, miny, maxx, maxy, v):
    for i in range(minx.shape[0]):
        for xx in range(minx[i], maxx[i]):
            for yy in range(miny[i], maxy[i]):
                field[yy, xx] += v[i]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def scalar_square_add_gauss(double[:, :] field, x_np, y_np, sigma_np, v_np, double truncate=2.0):
    # truncate -> 截断

    sigma_np = np.maximum(1.0, sigma_np)
    width_np = np.maximum(1.0, truncate * sigma_np)

    # x_np 是传入的一组 x 坐标, y_np 是传入的一组 y 坐标。
    # 下面的目的是防止越界
    # [minx:maxx, miny:maxy]区域的长和宽都是2*width_np
    minx_np = np.round(x_np - width_np).astype(np.int)
    minx_np = np.clip(minx_np, 0, field.shape[1] - 1)
    miny_np = np.round(y_np - width_np).astype(np.int)
    miny_np = np.clip(miny_np, 0, field.shape[0] - 1)
    maxx_np = np.round(x_np + width_np).astype(np.int)
    maxx_np = np.clip(maxx_np + 1, minx_np + 1, field.shape[1])
    maxy_np = np.round(y_np + width_np).astype(np.int)
    maxy_np = np.clip(maxy_np + 1, miny_np + 1, field.shape[0])

    cdef double[:] x = x_np  # 原始未判断越界的坐标
    cdef double[:] y = y_np
    cdef double[:] sigma = sigma_np
    cdef long[:] minx = minx_np
    cdef long[:] miny = miny_np
    cdef long[:] maxx = maxx_np
    cdef long[:] maxy = maxy_np
    cdef double[:] v = v_np

    cdef Py_ssize_t i, xx, yy
    cdef Py_ssize_t l = minx.shape[0] # 注意这里，变量l
    cdef double deltax, deltay
    cdef double vv

    # for minxx, minyy, maxxx, maxyy, vv in zip(minx, miny, maxx, maxy, v):
    # 玛德,下面这是l不是1,是一个变量! minx.shape[0]
    # (x[i],y[i])为中心,上下左右宽度为width_np 的范围内的所有confidence做一个高斯
    for i in range(l): # 第一个循环num层
        for xx in range(minx[i], maxx[i]):
            # x[i],y[i]为原来的坐标
            deltax = xx - x[i]
            for yy in range(miny[i], maxy[i]):
                deltay = yy - y[i]
                # 这里相比于前面的scalar_square_add_constant方法,相当于给v[i]加了个高斯核
                # todo 从传入进的参数可知,这里的v[i]先*了个1/16,这是因为下面算的是一个累加,总共有17个关节?
                vv = v[i] * exp(-0.5 * (deltax**2 + deltay**2) / sigma[i]**2)
                field[yy, xx] += vv


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
#  x_np, y_np, double[:] weights 对应的传入参数为 target_coordinates:(num4, 2)   y:(2,)  scores:(num4,)
def weiszfeld_nd(x_np, y_np, double[:] weights=None, double epsilon=1e-8, Py_ssize_t max_steps=20):
    """Weighted Weiszfeld algorithm."""
    # 算法的目的是:找出到其他所有点距离和最短的点,即求中位中心
    if weights is None:
        weights = np.ones(x_np.shape[0])

    cdef double[:, :] x = x_np  # (num4, 2)
    cdef double[:] y = y_np # (2,)
    cdef double[:, :] weights_x = np.zeros_like(x)  # (num4, 2)
    for i in range(weights_x.shape[0]):
        for j in range(weights_x.shape[1]):
            weights_x[i, j] = weights[i] * x[i, j]

    cdef double[:] prev_y = np.zeros_like(y)
    cdef double[:] y_top = np.zeros_like(y)
    cdef double y_bottom
    denom_np = np.zeros_like(weights) # (num4,)
    cdef double[:] denom = denom_np #(num4,)
    # 迭代
    for s in range(max_steps):
        prev_y[:] = y
        for i in range(denom.shape[0]):
            # 这里的epsilon是ε TODO 作用是防止值过小?
            # denom 中放的是所有关节点到(所有关节点的加权平均点)之间的距离值
            denom[i] = sqrt((x[i][0] - prev_y[0])**2 + (x[i][1] - prev_y[1])**2) + epsilon

        y_top[:] = 0.0 # (2,)
        y_bottom = 0.0
        for j in range(denom.shape[0]):
            # weights_x[j, 0]与weights_x[j, 1] 是关节j的公式3中的s(a,x)
            # denom[j] 是关节j 到(所有关节点的加权平均点)之间的距离值
            y_top[0] += weights_x[j, 0] / denom[j]
            y_top[1] += weights_x[j, 1] / denom[j]
            y_bottom += weights[j] / denom[j]
        y[0] = y_top[0] / y_bottom
        y[1] = y_top[1] / y_bottom

        if fabs(y[0] - prev_y[0]) + fabs(y[1] - prev_y[1]) < 1e-2:
            return y_np, denom_np
    # TODO 这里为啥这么返回
    return y_np, denom_np


@cython.boundscheck(False)
@cython.wraparound(False)
def paf_mask_center(double[:, :] paf_field, double x, double y, double sigma=1.0):
    mask_np = np.zeros((paf_field.shape[1],), dtype=np.uint8)
    cdef unsigned char[:] mask = mask_np

    for i in range(mask.shape[0]):
        mask[i] = (
            paf_field[1, i] > x - sigma * paf_field[3, i] and
            paf_field[1, i] < x + sigma * paf_field[3, i] and
            paf_field[2, i] > y - sigma * paf_field[3, i] and
            paf_field[2, i] < y + sigma * paf_field[3, i]
        )

    return mask_np != 0


@cython.boundscheck(False)
@cython.wraparound(False)
# paf_field的shape为(7, num3)   x, y都是一个数
# 7 为 score, x1 ,y1, logb1, x2, y2, logb2
# 传入的 sigma = 2
# todo 函数的目的是判断x1,y1中哪些可以认定为和x,y是同一个关节点
def paf_center(double[:, :] paf_field, double x, double y, double sigma=1.0):
    result_np = np.empty_like(paf_field) # (7, num3)
    cdef double[:, :] result = result_np
    cdef unsigned int result_i = 0
    cdef bint take

    for i in range(paf_field.shape[1]): # num3
        # 找出在关节点(x, y) 2*logb1范围内的点,认定它们同属一个关节点
        take = (
                x - sigma * paf_field[3, i] < paf_field[1, i] < x + sigma * paf_field[3, i] and
                y - sigma * paf_field[3, i] < paf_field[2, i] < y + sigma * paf_field[3, i]
        )
        if not take:
            continue

        result[:, result_i] = paf_field[:, i] # 将满足条件的关节点保存起来
        result_i += 1

    return result_np[:, :result_i] # (7, num4)
