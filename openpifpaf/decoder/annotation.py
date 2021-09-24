import numpy as np


class Annotation(object):
    # 传入的 f:channel index; (x, y, v):x,y,confidence;  self.skeleton:COCO_PERSON_SKELETON
    def __init__(self, j, xyv, skeleton):
        # j:channel号  xyv是一个3元组,其中值为(x,y,confidence) skeleton是COCO_PERSON_SKELETON,放在data.py中
        n_joints = len(set(i for c in skeleton for i in c))  # 1-17的一个set
        self.data = np.zeros((n_joints, 3))  # (17, 3)
        self.joint_scales = None
        self.data[j] = xyv  # 这里可以直接使用三元组给3维数组赋值,    这里的值都是seed的
        # 将 data.py中的骨架列表整体-1
        self.skeleton_m1 = (np.asarray(skeleton) - 1).tolist()  # 减去1是为了可以直接当成索引

    def fill_joint_scales(self, scales, hr_scale):
        self.joint_scales = np.zeros((self.data.shape[0],))
        for xyv_i, xyv in enumerate(self.data):
            if xyv[2] == 0.0:
                continue
            scale_field = scales[xyv_i]
            i = max(0, min(scale_field.shape[1] - 1, int(round(xyv[0] * hr_scale))))
            j = max(0, min(scale_field.shape[0] - 1, int(round(xyv[1] * hr_scale))))
            self.joint_scales[xyv_i] = scale_field[j, i] / hr_scale

    def score(self):
        v = self.data[:, 2]
        return 0.1 * np.max(v) + 0.9 * np.mean(np.square(v))
        # return np.mean(np.square(v))

    # todo 函数的意思就是找到目前所有应该连接但是还没有连接的线(一个>0,一个=0)(两个都>0说明已经完成连接了)(因为一开始只有一个seed点)。
    #      这些线可能有正向的(seed作为起点),也可能有反向的(seed作为终点)
    def frontier(self):
        """Frontier to complete annotation.
        Format: (
            confidence of origin,
            connection index,
            forward?,
            joint index 1,  (not corrected for forward)
            joint index 2,  (not corrected for forward)
        )
        """
        # 列表相加相当于直接拼接,列表中的元素都是相当于以元组的形式出现的
        return sorted([
                          (self.data[j1i, 2], connection_i, True, j1i, j2i)
                          # connection_i表示的是连接的index,j1i和j2i是连接的起始关节与终止关节的index
                          for connection_i, (j1i, j2i) in enumerate(self.skeleton_m1)
                          # 下面说明对于一个人体来说,j1i这个关节已经找到,j2i这个关节还未找到,那么我们就需要把它们记录下来
                          # todo 这里的True表示j1i找到,j2i没找到,是一个正向的连接
                          if self.data[j1i, 2] > 0.0 and self.data[j2i, 2] == 0.0
                      ] + [
                          # 不清楚为啥没有两者都是seed的情况,即都>0?这种情况是可能出现的,我们不需要管他,
                          # 因为这么操作下去总会连接完一个人体
                          # 这里的False表示j2i找到了,j1i还没找到,这是一个反向的连接
                          (self.data[j2i, 2], connection_i, False, j1i, j2i)
                          for connection_i, (j1i, j2i) in enumerate(self.skeleton_m1)
                          if self.data[j2i, 2] > 0.0 and self.data[j1i, 2] == 0.0
                      ], reverse=True)

    def frontier_iter(self):
        block_frontier = set()
        while True:
            # f是一个元组 confidence of origin, connection index, forward?,
            #             joint index 1,  (not corrected for forward)
            #             joint index 2,  (not corrected for forward)
            unblocked_frontier = [f for f in self.frontier()
                                  if (f[1], f[2]) not in block_frontier]
            if not unblocked_frontier:
                break
            # 从这个列表frontier里取出confidence最高的连接返回出去
            # 并将连接号与正向还是反向放入集合中,表明已经做过这个连接了
            # todo 这个地方我怎么感觉只需要连接号就足够了呢?放入forward没啥鸟用吧
            first = unblocked_frontier[0]
            yield first   # 返回出去(confidence, connection index, forward?, joint index1, joint index2),当前最高的confidence
            block_frontier.add((first[1], first[2]))

    def scale(self):
        m = self.data[:, 2] > 0.5
        if not np.any(m):
            return 0.0
        return max(
            np.max(self.data[m, 0]) - np.min(self.data[m, 0]),
            np.max(self.data[m, 1]) - np.min(self.data[m, 1]),
        )


class AnnotationWithoutSkeleton(object):
    def __init__(self, j, xyv, n_joints):
        self.data = np.zeros((n_joints, 3))
        self.joint_scales = None
        self.data[j] = xyv

    def fill_joint_scales(self, scales, hr_scale):
        self.joint_scales = np.zeros((self.data.shape[0],))
        for xyv_i, xyv in enumerate(self.data):
            if xyv[2] == 0.0:
                continue
            scale_field = scales[xyv_i]
            i = max(0, min(scale_field.shape[1] - 1, int(round(xyv[0] * hr_scale))))
            j = max(0, min(scale_field.shape[0] - 1, int(round(xyv[1] * hr_scale))))
            self.joint_scales[xyv_i] = scale_field[j, i] / hr_scale

    def score(self):
        v = self.data[:, 2]
        return 0.1 * np.max(v) + 0.9 * np.mean(np.square(v))
        # return np.mean(np.square(v))

    def scale(self):
        m = self.data[:, 2] > 0.5
        if not np.any(m):
            return 0.0
        return max(
            np.max(self.data[m, 0]) - np.min(self.data[m, 0]),
            np.max(self.data[m, 1]) - np.min(self.data[m, 1]),
        )
