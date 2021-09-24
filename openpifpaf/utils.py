import math
import numpy as np

from .data import COCO_KEYPOINTS, HFLIP


# 方法的目的：将coco数据集中keypoint水平翻转一下
# 即原先的数据形式是   开始关节：xyv
# 操作完后变成了       与之对应的关节：xyv
# 这里的与之对称的关节例如左手对应右手
def horizontal_swap_coco(keypoints):
    target = np.zeros(keypoints.shape)
    # source_i 从 0 开始, xyv分别表示x坐标,y坐标,以及可见性
    # v = 0，没有标注；v = 1，有标注不可见；v = 2，有标注可见
    for source_i, xyv in enumerate(keypoints):
        source_name = COCO_KEYPOINTS[source_i]  # 关节名称
        target_name = HFLIP.get(source_name)  # 与之对称的关节名称
        if target_name:
            target_i = COCO_KEYPOINTS.index(target_name)  # 与之对称的关节的序号
        else:
            target_i = source_i  # 没有配对的话就是它自己本身(如鼻子)
        target[target_i] = xyv  # 这里的target_i值是一个一一映射
    return target


def mask_valid_image(image, valid_area):
    image[:, :int(valid_area[1]), :] = 0
    image[:, :, :int(valid_area[0])] = 0
    max_i = int(math.ceil(valid_area[1] + valid_area[3]))
    max_j = int(math.ceil(valid_area[0] + valid_area[2]))
    if max_i < image.shape[1]:
        image[:, max_i:, :] = 0
    if max_j < image.shape[2]:
        image[:, :, max_j:] = 0
