import numpy as np

#
#
# def index_field(shape):
#     # np.indices 返回一个对应 shape的切片数组,例如shape为(2,3)时等价于 x[:2,:3]
#     yx = np.indices(shape, dtype=float)
#     # np.filp axis=0 时将每一行看作一个整体,行的顺序发生颠倒
#     xy = np.flip(yx, axis=0)
#     return xy
#
#
# f = np.arange(12).reshape(3, 4)
# print(f"f.shape:{f.shape}")
# index_fields = index_field(f.shape)  # 获得一个f.shape大小的索引矩阵(y在前,方便处理图片)
# print(f"index_fields.shape:{index_fields.shape}")
# candidates = np.concatenate((index_fields, np.expand_dims(f, 0)), 0)
# print(f"candidates.shape:{candidates.shape}")
# print(candidates)  # (3,3,4)
# print("#################################")
# seed_threshold = 6
# mask = f > seed_threshold  # 大于阈值
# print(candidates[:, mask])
# candidates = np.moveaxis(candidates[:, mask], 0, -1)
# # print(candidates)


# class A:
#     def __init__(self, a):
#         self.t = 10
#         print("init")
#
#     @classmethod
#     def abc(cls, a):
#         print("classmethod")
#
#     @staticmethod
#     def sdf(a):
#         print("staticmethod")
#
#
# class B(A):
#     pass
#
#
# class C(A):
#     pass

#
# for c in A.__subclasses__():
#     c("ad")


# def create_sink(side):
#     if side == 1:
#         return np.zeros((2, 1, 1))
#     sink1d = np.linspace((side - 1.0) / 2.0, -(side - 1.0) / 2.0, num=side, dtype=float)
#     sink = np.stack((
#         sink1d.reshape(1, -1).repeat(side, axis=0),
#         sink1d.reshape(-1, 1).repeat(side, axis=1),
#     ), axis=0)
#     return sink
#
#
# print(create_sink(3))
# a = np.array([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]).reshape(6, 2)
# b = np.array([1, 2, 3, 4, 5, 6])
# print(a.shape)
# print(b.shape)
# c = a * np.expand_dims(b, -1)
# print(c.shape)
# d = np.sum(c, axis=0)
# print(d)
# print(d.shape)
# a = np.zeros((19, 2))
# b = np.zeros((19, 1))
# c = np.sum(a * b, axis=0)
# print(c.shape)
# a = np.zeros((7, 19))
# b = a[3] * 3.0
# print(b.shape)
# def h():
#     print("afadsf")
# def fab(max):
#     n, a, b = 0, 0, 1
#     while n < max:
#         yield b  # 使用 yield
#         # print b
#         a, b = b, a + b
#         n = n + 1
#
#
# for n in fab(100):
#     print(n)
# import matplotlib.pyplot as plt
#
# x = [0, 1]
# y = [0, 1]
# fig, ax = plt.subplots()
# ax.plot(x, y,
#         linewidth=1,
#         solid_capstyle='round')
# plt.show()
# import numpy as np

# a = [2, 3, 4, 5, 6, 7, 8, 9]
# b = np.stack(a)
# print(b[::1])
import scipy.ndimage


# a = np.full((7, 7), 1, dtype=np.float32)
# a[1:6, 2:5] = 0
# print(a)
# b = scipy.ndimage.binary_erosion(a, iterations=1, border_value=1)
# print(b)
# a = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
# a = a.reshape((5, 2))
# print(a)
# b = a[:, 0] > 3
# print(b)
# print(a[b])
# print(np.max(a[b, 0]))
# for i,j in enumerate(a):
#     print(i,j)
# side = 4
# sink1d = np.linspace((side - 1.0) / 2.0, -(side - 1.0) / 2.0, num=side, dtype=float)
# # print(sink1d)
# sink = np.stack((
#     sink1d.reshape(1, -1).repeat(side, axis=0),
#     sink1d.reshape(-1, 1).repeat(side, axis=1),
# ), axis=0)
# s = np.zeros((2, 1, 1))
# sink_reg = sink + s
# joint1_offset = np.array([-0.5, 0.5]).reshape((2, 1, 1))
# joint2_offset = np.array([0.5, -0.5]).reshape((2, 1, 1))
# sink1 = sink + joint1_offset
# sink2 = sink + joint2_offset
# # print(sink1)
# # print(sink2)
# sink_l = np.minimum(np.linalg.norm(sink1, axis=0),
#                     np.linalg.norm(sink2, axis=0))
# print(np.linalg.norm(sink1, axis=0))
# print(np.linalg.norm(sink2, axis=0))
# print(sink_l)
# a = np.ones((19, 4, 3, 4))
# b = np.ones((19, 4, 3, 4))
# # c = a + b
# c = np.stack((a, b), axis=1)
# print(c.shape)
# fourds = np.arange(96).reshape(2, 4, 3, 4)
# # print(fourds[:, 0])
# fourds[0, 0, 0, 0] = 100
# scores = np.min(fourds[:, 0], axis=0)
# mask = scores > 5.5
# print(scores[mask])
# print(fourds[:, :, mask])
#
# def index_field(shape):
#     # np.indices 返回一个对应 shape的切片数组,例如shape为(2,3)时等价于 x[:2,:3]
#     yx = np.indices(shape, dtype=float)
#     # np.filp axis=0 时将每一行看作一个整体,行的顺序发生颠倒
#     xy = np.flip(yx, axis=0)
#     return xy


#
#
# f = np.arange(96).reshape(16, 2, 3)
# index_fields = index_field((3, 4))
# print(index_fields)
# candidates = np.concatenate((index_fields, np.expand_dims(f, 0)), 0)
# print(candidates.shape)

# yx = np.indices((3, 3), dtype=float)
# print(yx)
#
# for i, (a, b) in enumerate(zip(np.ones((17, 3, 4)), np.zeros((17, 3, 4)))):
#     print(f'{i},a.shape:{a.shape},b.shape:{b.shape}')
#     # print(b)

# a = np.array([[9, 4, 51, 34, 365, 6], [6, 4, 51, 34, 365, 16]])
# for b in sorted(a, key=lambda c: c[0], reverse=True):
#     print(b)

# a = (1, 2, 3)
# b = np.zeros((5, 3))
# b[0] = a
# print(b)

# skeleton = [
#     [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
#     [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
#     [2, 4], [3, 5], [4, 6], [5, 7]]
# a = (np.asarray(skeleton) - 1).tolist()
# print(a)

# a = [1, 2, 3]
# b = [2.4]
# print(a + b)

# print(1 < 2 < 3 < 1)

# python train --lr=1e-3 --momentum=0.95 --epochs=75 --lr-decay 60 70 --batch-size=8 --basenet=resnet50block5 --head-quad=1 --headnets pif paf --square-edge=401 --regression-loss=laplace --lambdas 30 2 2 50 3 3 --crop-fraction=0.5 --freeze-base=1

# print(np.round(1.5))

# def create_sink(side):
#     if side == 1:
#         return np.zeros((2, 1, 1))
#     # side 为 4 时,sink1d = [ 1.5  0.5 -0.5 -1.5]
#     sink1d = np.linspace((side - 1.0) / 2.0, -(side - 1.0) / 2.0, num=side, dtype=float)
#     sink = np.stack((
#         sink1d.reshape(1, -1).repeat(side, axis=0),  # 重复四遍 (4, 4)
#         sink1d.reshape(-1, 1).repeat(side, axis=1),  # 转置后重复四遍 (4, 4)
#     ), axis=0)
#     # stack后shape为(2, side, side),当side为4时为(2, 4, 4)
#     return sink
#
#
# s = create_sink(3)
# print(s)
# a = np.array([3, 4])
# print(a-s)

# fmargin = 0.1
# num = 3
# frange = np.linspace(fmargin, 1.0 - fmargin, num=num)
# print(frange)

# def create_sink(side):
# # #     if side == 1:
# #         return np.zeros((2, 1, 1))
# #     # side 为 4 时,sink1d = [ 1.5  0.5 -0.5 -1.5]
# #     sink1d = np.linspace((side - 1.0) / 2.0, -(side - 1.0) / 2.0, num=side, dtype=float)
# #     sink = np.stack((
# #         sink1d.reshape(1, -1).repeat(side, axis=0),  # 重复四遍 (4, 4)
# #         sink1d.reshape(-1, 1).repeat(side, axis=1),  # 转置后重复四遍 (4, 4)
# #     ), axis=0)
# #     # stack后shape为(2, side, side),当side为4时为(2, 4, 4)
# #     return sink
# #
# #
# # joint1 = np.array([10, 10, 1])
# # joint2 = np.array([20, 20, 1])
# #
# # offset = joint2[:2] - joint1[:2]
# # offset_d = np.linalg.norm(offset)  # 求得关节间臂的长度
# # # dynamically create s
# # # min_size 默认为3
# # s = max(3, int(offset_d * 0))
# # sink = create_sink(s)  # 创建偏移矩阵 shape为(2, s, s)
# # s_offset = (s - 1.0) / 2.0  # 和前面一样的,这里的s相当于之前的side_length
# # # pixel coordinates of top-left joint pixel
# # joint1ij = np.round(joint1[:2] - s_offset)
# # joint2ij = np.round(joint2[:2] - s_offset)
# # offsetij = joint2ij - joint1ij  # 可能比offset更准确一些,其实差不多
# #
# # # 动态地设置中间点的个数
# # num = max(2, int(np.ceil(offset_d)))
# # # np.spacing(1) = 2.2204e-16
# # fmargin = min(0.4, (s_offset + 1) / (offset_d + np.spacing(1)))
# # # fmargin = 0.0
# # frange = np.linspace(fmargin, 1.0 - fmargin, num=num)
# #
# # padding = 10
# # fields_reg_l = np.zeros((30, 30))
# # fields_reg_l.fill(100)
# # # print(fields_reg_l)
# # for f in frange:
# #     fij = np.round(joint1ij + f * offsetij) + padding
# #     # 以fij为起点，长和宽为s的区域
# #     fminx, fminy = int(fij[0]), int(fij[1])
# #     fmaxx, fmaxy = fminx + s, fminy + s
# #     # print(fminx, fmaxx)
# #     fxy = (fij - padding) + s_offset  # 对应的真实的xy坐标,≈joint1[:2] + f * offsetij
# #     # precise floating point offset of sinks
# #     joint1_offset = (joint1[:2] - fxy).reshape(2, 1, 1)  # 与 joint1的偏移,≈ f * offsetij
# #     joint2_offset = (joint2[:2] - fxy).reshape(2, 1, 1)  # 与 joint2的偏移
# #     sink1 = sink + joint1_offset
# #     sink2 = sink + joint2_offset
# #     print(f"sink1:{sink1}")
#      print(f"sink2:{sink2}")
# #     # sink_l = np.minimum(np.linalg.norm(sink1, axis=0),
# #     #                     np.linalg.norm(sink2, axis=0))
# #     # # print(sink_l)
# #     # mask = sink_l < fields_reg_l[fminy:fmaxy, fminx:fmaxx]
# #      #fields_reg_l[fminy:fmaxy, fminx:fmaxx][mask] = sink_l[mask]

# shape = (3, 4)
#
#
# def index_field(shape):
#     # np.indices 返回一个对应 shape的切片数组,例如shape为(2,3)时等价于 x[:2,:3]
#     yx = np.indices(shape, dtype=float)
#     # np.filp axis=0 时将每一行看作一个整体,行的顺序发生颠倒
#     xy = np.flip(yx, axis=0)
#     return xy
#
# print(index_field(shape).shape)

# a = np.arange(24).reshape((2, 3, 4))
# b, c = a[:, a[0] > 5.5]
# print(b.shape, c)

# a = np.arange(24)
# print(a[:500])

# a = set()
# a.add((123, True))
# a.add((123, False))
# print(a)
#
# a = np.arange(12).reshape(2, 6)
# print(a.sum(axis=1).shape)

a = np.arange(12)
print(sorted(a))