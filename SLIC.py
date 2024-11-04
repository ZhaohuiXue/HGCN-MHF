import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries
from sklearn import preprocessing



def SegmentsLabelProcess(labels):
    '''
    对labels做后处理，防止出现label不连续现象
    '''
    labels = np.array(labels, np.int64)
    H, W = labels.shape
    ls = list(set(np.reshape(labels, [-1]).tolist()))

    dic = {}
    for i in range(len(ls)):
        dic[ls[i]] = i

    new_labels = labels
    for i in range(H):
        for j in range(W):
            new_labels[i, j] = dic[new_labels[i, j]]
    return new_labels


class SLIC(object):
    def __init__(self, HSI, FLAG, n_segments=1000, compactness=20, max_iter=20, sigma=0, min_size_factor=0.3,
                 max_size_factor=2):
        self.n_segments = n_segments
        self.compactness = compactness
        self.max_iter = max_iter
        self.min_size_factor = min_size_factor
        self.max_size_factor = max_size_factor
        self.sigma = sigma
        # 数据standardization标准化,即提前全局BN
        height, width, bands = HSI.shape  # 原始高光谱数据的三个维度
        data = np.reshape(HSI, [height * width, bands])
        min_max = preprocessing.StandardScaler()
        data = min_max.fit_transform(data)
        self.data = np.reshape(data, [height, width, bands])
        self.FLAG = FLAG

    def get_Q_and_S_and_Segments(self):
        # 执行 SLIC 并得到Q(nxm),S(m*b)
        img = self.data
        (h, w, d) = img.shape
        # 计算超像素S以及相关系数矩阵Q
        # data_pca = self.pca_processing(img, 3)
        segments = slic(img, n_segments=self.n_segments, compactness=self.compactness, max_num_iter=self.max_iter,
                        convert2lab=False, sigma=self.sigma, enforce_connectivity=True,
                        min_size_factor=self.min_size_factor, max_size_factor=self.max_size_factor, slic_zero=False)

        # 判断超像素label是否连续,否则予以校正
        if segments.max() + 1 != len(list(set(np.reshape(segments, [-1]).tolist()))):
            segments = SegmentsLabelProcess(segments)
        self.segments = segments
        superpixel_count = segments.max() + 1  # =segments.max() + 1
        self.superpixel_count = superpixel_count
        print("superpixel_count", superpixel_count)

        # ######################################显示超像素图片
        if self.FLAG == 1 or self.FLAG == 3:
            out = mark_boundaries(img[:, :, [50, 27, 17]], segments, color=(1, 1, 0)) # IP, Salinas
        # out = mark_boundaries(img[:, :, [57, 27, 17]], segments, color=(255,255,0))
        if self.FLAG == 2:
            out = mark_boundaries(img[:, :, [102, 56, 31]], segments, color=(1, 1, 0)) #PU
        if self.FLAG == 4:
            out = mark_boundaries(img[:, :, [58, 28, 8]], segments, color=(1, 1, 0)) #hhk
        # plt.figure()
        # plt.imshow(out, cmap='jet')
        # plt.xticks([])
        # plt.yticks([])
        # # plt.show()
        # # plt.axis('off')
        # plt.savefig("segment_superpixel" + str(self.FLAG), dpi=300, bbox_inches='tight')

        segments_1 = np.reshape(segments, [-1])
        S = np.zeros([superpixel_count, d], dtype=np.float32)  # 超像素特征
        Q = np.zeros([w * h, superpixel_count], dtype=np.float32)  # 像素与超像素联系矩阵
        x = np.reshape(img, [-1, d])  # Flatten(x)

        x_center = np.zeros([superpixel_count], dtype=np.float32)
        y_center = np.zeros([superpixel_count], dtype=np.float32)

        for i in range(superpixel_count):
            idx = np.where(segments_1 == i)[0]
            count = len(idx)
            pixels = x[idx]
            superpixel = np.sum(pixels, 0) / count
            S[i] = superpixel
            Q[idx, i] = 1

            seg_idx = np.where(segments == i)
            x_center[i] = np.mean(seg_idx[0])
            y_center[i] = np.mean(seg_idx[1])

        self.S = S
        self.Q = Q

        return Q, S, self.segments, x_center, y_center

    def get_A(self):
        '''
         根据 segments 判定邻接矩阵
        :return:
        '''
        A = np.zeros([self.superpixel_count, self.superpixel_count], dtype=np.float32)
        (h, w) = self.segments.shape
        for i in range(h - 1):
            for j in range(w - 1):
                sub = self.segments[i:i + 2, j:j + 2]
                sub_max = np.max(sub).astype(np.int32)
                sub_min = np.min(sub).astype(np.int32)
                # if len(sub_set)>1:
                if sub_max != sub_min:
                    idx1 = sub_max
                    idx2 = sub_min
                    # if A[idx1, idx2] != 0:
                    #     continue
                    # pix1 = self.S[idx1]
                    # pix2 = self.S[idx2]
                    # diss = np.exp(-np.sum(np.square(pix1 - pix2)) / (2*sigma ** 2))
                    # A[idx1, idx2] = A[idx2, idx1] = diss
                    A[idx1, idx2] = A[idx2, idx1] = 1
        return A


class Segment(object):
    def __init__(self, data, n_component, FLAG):
        self.data = data
        self.n_component = n_component
        self.height, self.width, self.bands = data.shape
        self.FLAG = FLAG

    def SLIC_Process(self, img, scale=25):
        n_segments_init = self.height * self.width / scale
        print("n_segments_init", n_segments_init)
        FLAG = self.FLAG
        # # IP
        if FLAG == 1:
            myslic = SLIC(img, FLAG, n_segments=n_segments_init, compactness=0.1, sigma=1, min_size_factor=0.1,
                          max_size_factor=10)
        # # PU
        if FLAG == 2:
            myslic = SLIC(img, FLAG, n_segments=n_segments_init, compactness=0.1, sigma=1.5, min_size_factor=0.1,
                          max_size_factor=10)
        # # salinas
        if FLAG == 3:
            myslic = SLIC(img, FLAG, n_segments=n_segments_init, compactness=0.01, sigma=0.5, min_size_factor=0.1,
                          max_size_factor=10)
        if FLAG == 4:
            myslic = SLIC(img, FLAG, n_segments=n_segments_init, compactness=0.1, sigma=1, min_size_factor=0.1,
                          max_size_factor=10)
        # IP
        # if FLAG == 1:
        #     myslic = SLIC(img, FLAG, n_segments=n_segments_init, compactness=0.01, sigma=1.5, min_size_factor=0.1,
        #                   max_size_factor=10)
        # # # PU
        # if FLAG == 2:
        #     myslic = SLIC(img, FLAG, n_segments=n_segments_init, compactness=0.01, sigma=1.5, min_size_factor=0.1,
        #                   max_size_factor=10)
        # # # salinas
        # if FLAG == 3:
        # myslic = SLIC(img, FLAG, n_segments=n_segments_init, compactness=0.1, sigma=1.5, min_size_factor=0.1,
        #                   max_size_factor=10)
        Q, S, Segments, x_center, y_center = myslic.get_Q_and_S_and_Segments()
        A = myslic.get_A()
        return Q, S, A, Segments, x_center, y_center