import numpy as np


# 对图片你的keypoints以及valid_area进行缩放,并且生成一个 bg_mask
# 生成的bg_mask是 对那些没有标注任何可见的关节点的区域(以一个人的box为范围)标注为0, 其余为1.
# 意思就是这人有box, 但box里没有任何一个可见的关节点(所有关节点的confidence都为0), 那这个box区域内的值就全为0.
class AnnRescaler(object):
    def __init__(self, input_output_scale, n_keypoints):
        self.input_output_scale = input_output_scale  # 一个正整数
        self.n_keypoints = n_keypoints

    def __call__(self, anns, width_height_original):
        keypoint_sets = self.anns_to_keypoint_sets(anns)
        # keypoint_sets第1维可能是所有人的个数,第2维可能是17,第3维可能是(x,y,v)
        keypoint_sets[:, :, :2] /= self.input_output_scale  # 对关节点x,y坐标的位置进行缩放

        # background mask
        # mask 是一个 shape 为 height*width 的矩阵
        bg_mask = self.anns_to_bg_mask(width_height_original, anns)
        # ::n 表示 隔n个选一个,这就表示self.input_output_scale必须为一个>1的正整数
        bg_mask = bg_mask[::self.input_output_scale, ::self.input_output_scale]

        # valid area 指的是有效的区域,将来需要学习的区域
        # 对valid_area 进行缩放
        valid_area = None
        if anns and 'valid_area' in anns[0]:
            valid_area = anns[0]['valid_area']
            valid_area = (
                valid_area[0] / self.input_output_scale,
                valid_area[1] / self.input_output_scale,
                valid_area[2] / self.input_output_scale,
                valid_area[3] / self.input_output_scale,
            )

        return keypoint_sets, bg_mask, valid_area

    # 将ann['keypoints']中的信息单独提取出来形成一个 set
    def anns_to_keypoint_sets(self, anns):
        """Ignore annotations of crowds."""
        # 忽略掉挤在一起的人,将所有的 keypoints 提取出来
        keypoint_sets = [ann['keypoints'] for ann in anns if not ann['iscrowd']]
        if not keypoint_sets:
            return np.zeros((0, self.n_keypoints, 3))
        # 这里的 stack 作用主要是将列表转化成 numpy数组
        return np.stack(keypoint_sets)

    @staticmethod
    # 根据anns来返回一个mask
    # mask 是一个 shape 为 height*width 的矩阵,对于有bounding box 的地方mask值为0,背景处值为1
    def anns_to_bg_mask(width_height, anns, include_annotated=True):
        """Create background mask taking crowded annotations into account."""
        # [::-1]是进行倒序操作,创建一个width_height的倒序维数的全True矩阵
        # 猜测这里的width_height是一个数组，[width, height]
        # mask 是一个 shape 为 height*width 的全True矩阵
        mask = np.ones(width_height[::-1], dtype=np.bool)
        for ann in anns:
            # 如果iscrowd = 0 且 有keypoints标注信息 则直接跳过
            if include_annotated and not ann['iscrowd'] and \
                    'keypoints' in ann and np.any(ann['keypoints'][:, 2] > 0):  # v=1，有标注不可见；v=2，有标注可见
                continue
            # iscrowd = 1 或者 只有人的标记框但是没有keypoints的信息
            if 'mask' not in ann:  # 如果图片没有标注 mask
                bb = ann['bbox'].copy()
                # "bbox": [x, y, width, height]
                bb[2:] += bb[:2]  # convert width and height to x2 and y2
                # "bbox": [x, y, x + width, y + height]
                bb[0] = np.clip(bb[0], 0, mask.shape[1] - 1)  # shape[1] 为 width
                bb[1] = np.clip(bb[1], 0, mask.shape[0] - 1)  # shape[0] 为 height
                bb[2] = np.clip(bb[2], 0, mask.shape[1] - 1)
                bb[3] = np.clip(bb[3], 0, mask.shape[0] - 1)
                bb = bb.astype(np.int)  # 转化为整数
                mask[bb[1]:bb[3] + 1, bb[0]:bb[2] + 1] = 0  # 将bounding box对应的mask设置为0
                continue
            mask[ann['mask']] = 0  # 这样 'mask' not in ann 判断就为 False 了
        return mask
