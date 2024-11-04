import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import random
from sklearn import metrics
import time
import torch
import SLIC
import model
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sample_type = ['ratio', 'same'][1]
# FLAG =1, indian
# FLAG =2, paviaU
# FLAG =3, salinas

for (FLAG, train_num, Scale) in [(1, 5, 100)]:
# for (FLAG, train_num, Scale) in [(1,5,100),(2,5,400),(3,5,500)]:
    torch.cuda.empty_cache()
    OA_ALL = []
    AA_ALL = []
    KPP_ALL = []
    AVG_ALL = []
    Train_Time_ALL = []
    Test_Time_ALL = []

    Seed_List = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 随机种子点
    # Seed_List = [0]
    if FLAG == 1:
        data_mat = sio.loadmat('./datasets/Indian_pines_corrected.mat')
        data = data_mat['indian_pines_corrected']
        gt_mat = sio.loadmat('./datasets/Indian_pines_gt.mat')
        gt = gt_mat['indian_pines_gt']
        # 参数预设

        class_count = 16  # 样本类别数
        learning_rate = 5e-4 # 学习率5e-4
        max_epoch = 500  # 迭代次数
        dataset_name = "IP_"  # 数据集名称
        # superpixel_scale=100
        pass
    if FLAG == 2:
        data_mat = sio.loadmat('./datasets/PaviaU.mat')
        data = data_mat['paviaU']
        gt_mat = sio.loadmat('./datasets/PaviaU_gt.mat')
        gt = gt_mat['paviaU_gt']

        # 参数预设

        class_count = 9  # 样本类别数
        learning_rate = 5e-4  # 学习率
        max_epoch = 500  # 迭代次数
        dataset_name = "PU_"  # 数据集名称
        # superpixel_scale = 100
        pass
    if FLAG == 3:
        data_mat = sio.loadmat('./datasets/Salinas_corrected.mat')
        data = data_mat['salinas_corrected']
        gt_mat = sio.loadmat('./datasets/Salinas_gt.mat')
        gt = gt_mat['salinas_gt']

        # 参数预设

        class_count = 16  # 样本类别数
        learning_rate = 5e-4  # 学习率
        max_epoch = 500  # 迭代次数
        dataset_name = "salinas_"  # 数据集名称
        # superpixel_scale = 100
        pass
    if FLAG == 4:
        data_mat = sio.loadmat('./datasets/WHU_Hi_HongHu.mat')
        data = data_mat['WHU_Hi_HongHu']
        gt_mat = sio.loadmat('./datasets/WHU_Hi_HongHu_gt.mat')
        gt = gt_mat['WHU_Hi_HongHu_gt']

        # 参数预设

        class_count = 22  # 样本类别数
        learning_rate = 5e-4  # 学习率
        max_epoch = 500  # 迭代次数
        dataset_name = "WHU-Hi-HongHu_"  # 数据集名称
        # superpixel_scale = 200
        pass
    ###########
    superpixel_scale = Scale  #########################
    train_samples_per_class = train_num  # 当定义为每类样本个数时,则该参数更改为训练样本数
    val_samples = class_count
    train_ratio = 0.01
    m, n, d = data.shape  # 高光谱数据的三个维度

    # 数据standardization标准化,即提前全局BN
    orig_data = data
    height, width, bands = data.shape  # 原始高光谱数据的三个维度
    # data = np.reshape(data, [height * width, bands])
    # minMax = preprocessing.StandardScaler()
    # data = minMax.fit_transform(data)
    # data = np.reshape(data, [height, width, bands])


    # # 打印每类样本个数
    # gt_reshape=np.reshape(gt, [-1])
    # for i in range(class_count):
    #     idx = np.where(gt_reshape == i + 1)[-1]
    #     samplesCount = len(idx)
    #     print(samplesCount)
    color_map_dict = {

        'IP_': np.array([[0, 168, 132], [76, 0, 115], [0, 0, 0], [190, 255, 232], [255, 0, 0],
                                [115, 0, 0], [205, 205, 102], [137, 90, 68], [215, 158, 158], [255, 115, 223],
                                [0, 0, 255], [156, 156, 156], [115, 223, 255], [0, 255, 0], [255, 255, 0],
                                [255, 170, 0]], dtype=np.uint8),

        'PU_': np.array([[0, 0, 255], [76, 230, 0], [255, 190, 232], [255, 0, 0], [156, 156, 156],
                        [255, 255, 115], [0, 255, 197], [132, 0, 168], [0, 0, 0]], dtype=np.uint8),



        'salinas_': np.array([[0, 168, 132], [76, 0, 115], [0, 0, 0], [190, 255, 232], [255, 0, 0],
                             [115, 0, 0], [205, 205, 102], [137, 90, 68], [215, 158, 158], [255, 115, 223],
                             [0, 0, 255], [156, 156, 156], [115, 223, 255], [0, 255, 0], [255, 255, 0],
                             [255, 170, 0]], dtype=np.uint8),

        'WHU-Hi-HongHu_': np.array([[255, 0, 0], [0, 0, 0], [176, 48, 96], [255, 255, 0], [255, 127, 80],
                   [0, 255, 0], [0, 205, 0], [0, 139, 0], [127, 255, 212], [160, 32, 240],
                   [216, 191, 216], [0, 0, 255], [0, 0, 139], [218, 112, 214], [160, 82, 45],
                   [0, 255, 255], [255, 165, 0], [127, 255, 0], [139, 139, 0], [0, 139, 139],
                   [205, 181, 205], [238, 154, 0]], dtype=np.uint8),

    }

    def GT_To_One_Hot(gt, class_count):
        '''
        Convet Gt to one-hot labels
        :param gt:
        :param class_count:
        :return:
        '''
        GT_One_Hot = []  # 转化为one-hot形式的标签
        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                temp = np.zeros(class_count, dtype=np.float32)
                if gt[i, j] != 0:
                    temp[int(gt[i, j]) - 1] = 1
                GT_One_Hot.append(temp)
        GT_One_Hot = np.reshape(GT_One_Hot, [height, width, class_count])
        return GT_One_Hot

    def convert_A_to_list(A:np.ndarray):
        M, N = A.shape[0], A.shape[1]
        A_list = []
        for i in range(M):
            for j in range(i+1, N):
                if A[i, j] != 0:
                    A_list.append(tuple([i, j]))
        return A_list


    pathset = []
    hopset = []

    def Get_k_hop(k_hop, n_graph):
        print(' Generating ', k_hop, ' hop HyperEdge')
        e = []
        # new_graph = np.zeros_like(n_graph)
        SuperPixel_count = n_graph.shape[0]
        for center in range(SuperPixel_count):
            path = [center]
            e_hop = DFS_get_path(center, k_hop, path, n_graph)
            e_hop.insert(0, center)
            e.append(tuple(e_hop))
            pathset.clear()
            hopset.clear()
        # new_graph = new_graph + np.eye(SuperPixel_count)
        return e

    def DFS_get_path(start_node, n_k, get_path, A):

        if len(get_path) == n_k + 1:
            pathset.append(get_path[:])
            k_hop = get_path[-1]
            if k_hop not in hopset:
                hopset.append(k_hop)
        else:
            sub_node = list(np.where(A[start_node] != 0)[0])
            for next_node in sub_node:
                if next_node not in get_path:
                    get_path.append(next_node)
                    DFS_get_path(next_node, n_k, get_path, A)
                    get_path.pop()
        return hopset

    def knn_neighbors(data, hop_graph, K=3):
        """
        对每个超像素选择K个最近的邻居超像素节点。
        """
        print(' From 2-hop Generating KNN HyperEdge K=', K)
        # 计算距离矩阵
        # SuperPixel_count = data.shape[0]
        # distances = torch.zeros((SuperPixel_count, SuperPixel_count))
        distances = torch.norm(data[:, None, :] - data[None, :, :], dim=2)
        # knn_indices = torch.argsort(distances, dim=1)
        e = []
        # 获取hop图中的 K 个最近邻索引
        for i, graph in enumerate(hop_graph):
            if len(graph) > K:
                diss = distances[i, graph]
                knn_indices = torch.argsort(diss)[:K]
                e.append(tuple([graph[index] for index in knn_indices]))
                # print()
            elif len(graph) <= K:
                e.append(graph)
            # else:
        return e

    # def generate_hyperedge_list_by_distance(S:np.ndarray, d_ratio:float=0.001, sigma=10):
    #     ratio = d_ratio*2
    #     feature = S
    #     v_num = feature.shape[0]
    #     e_list = []
    #     dis_mat = np.zeros((v_num, v_num))
    #     for i in range(v_num):
    #         for j in range(v_num):
    #             # diss = np.sqrt(np.sum(np.square(feature[i]-feature[j])))/feature.shape[1]
    #             diss = np.arccos(np.exp(-np.sum(np.square(feature[i] - feature[j])) / (2 * sigma ** 2)))# 光谱距离相似度
    #             # diss = torch.dist(feature[i],feature[j])
    #             dis_mat[i, j] = diss
    #     dis_mat_reshape = dis_mat.reshape((-1))
    #     dis_mat_sort = np.flipud(np.sort(dis_mat_reshape))
    #     # dis_max = dis_mat_sort[v_num]
    #     threshold = dis_mat_sort[int(ratio*(v_num**2-v_num))+v_num]
    #     dis_mat1 = np.where(dis_mat==1, np.zeros((v_num,v_num)), dis_mat)
    #     for i in range(v_num):
    #         # list = [i]
    #         idx = np.where(dis_mat1[i] > threshold)
    #         list = idx[0].tolist()
    #         list.append(i)
    #         list.sort()
    #         if list != [i] and len(list) < 25:
    #             e_list.extend([list])
    #     lst = []
    #     for el in e_list:
    #         if lst.count(el) < 1:
    #             lst.append(el)
    #     return lst

    def generate_W_e(X: torch.tensor, hyperedge, superpixel_center):
        print(' Calculating W')
        X = np.array(X)
        e = hyperedge
        # w = []
        w_spa = []
        w_fea = []
        if FLAG == 1:
            sigma1 = 100
            sigma2 = 100
        if FLAG == 2:
            sigma1 = 500
            sigma2 = 500
        if FLAG == 3:
            sigma1 = 1000
            sigma2 = 100
        if FLAG == 4:
            sigma1 = 500
            sigma2 = 500
        superpixel_count = X.shape[0]
        spatial_distance = np.zeros((superpixel_count, superpixel_count))
        for i in range(superpixel_count):
            for j in range(superpixel_count):
                x_coordinate = np.array([superpixel_center[0][i], superpixel_center[1][i]])
                y_coordinate = np.array([superpixel_center[0][j], superpixel_center[1][j]])
                spatial_distance[i, j] = np.sqrt(np.sum(np.square(x_coordinate - y_coordinate)))
        feature_distance = np.zeros((superpixel_count, superpixel_count))
        for i in range(superpixel_count):
            for j in range(superpixel_count):
                x_feature = X[i]
                y_feature = X[j]
                feature_distance[i, j] = np.sum(np.square(x_feature - y_feature))
        # sigma2 = 15
        # superpixel_distance = np.sum(spatial_distance)/len(np.where(spatial_distance != 0)[0])
        for edge in e:
            edge = torch.Tensor(edge)
            n = edge.shape[0]
            Distance_matrix = np.zeros((n, n))
            Similarity_matrix = np.zeros((n, n))
            # weight_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i != j:
                        x = int(edge[i])
                        y = int(edge[j])
                        Distance_matrix[i, j] = spatial_distance[x, y]
                        # Similarity_matrix[i, j] = np.sin(np.arccos(np.exp(-np.sum(np.square(X[x] - X[y])) / (2 * sigma ** 2))))
                        Similarity_matrix[i, j] = feature_distance[x, y]
            # average_Distance = np.sum(Distance_matrix)/len(np.where(Distance_matrix != 0)[0])
            # average_Similarity_Distance = np.sum(Similarity_matrix)/len(np.where(Similarity_matrix != 0)[0])
            Distance_matrix1 = np.exp(-Distance_matrix/(2*sigma1**2))
            Similarity_matrix1 = np.exp(-Similarity_matrix/(2*sigma2**2))
            np.fill_diagonal(Distance_matrix1, 0)
            np.fill_diagonal(Similarity_matrix1, 0)
            Spatial_Distance = np.sum(Distance_matrix1) / len(np.where(Distance_matrix1 != 0)[0])
            Similarity_Distance = np.sum(Similarity_matrix1) / len(np.where(Similarity_matrix1 != 0)[0])
            w_spa.append(Spatial_Distance)
            w_fea.append(Similarity_Distance)
            # w.append(1)

        W_spa = torch.Tensor(w_spa)
        W_fea = torch.Tensor(w_fea)
        W = torch.min(W_spa, W_fea)
        return W

    def cal_Dv_De(H: torch.Tensor):
        H_dense = H
        edge_degrees = torch.sum(H_dense, dim=0)
        node_degrees = torch.sum(H_dense, dim=-1)
        indices_edge = torch.arange(edge_degrees.size(0)).unsqueeze(0).repeat(2, 1).to(H.device)
        indices_node = torch.arange(node_degrees.size(0)).unsqueeze(0).repeat(2, 1).to(H.device)
        e_diag = torch.sparse_coo_tensor(indices_edge, edge_degrees,
                                         size=(edge_degrees.size(0), edge_degrees.size(0))).to_dense().to(device)
        v_diag = torch.sparse_coo_tensor(indices_node, node_degrees,
                                         size=(node_degrees.size(0), node_degrees.size(0))).to_dense().to(device)
        return e_diag, v_diag

    def build_incidence_matrix(nodes, hyperedges):
            n_nodes = len(nodes)
            n_hyperedges = len(hyperedges)

            incidence_matrix = torch.zeros((n_nodes, n_hyperedges), dtype=torch.float32)

            for i, node in enumerate(nodes):
                for j, hyperedge in enumerate(hyperedges):
                    if node in hyperedge:
                        incidence_matrix[i, j] = 1.0

            return incidence_matrix

    ls = SLIC.Segment(data, class_count - 1, FLAG)
    tic0 = time.time()
    Q, S, A, Seg, x_center, y_center = ls.SLIC_Process(data, scale=superpixel_scale)
    superpixel_center = (x_center, y_center)
    toc0 = time.time()
    SLIC_Time = toc0 - tic0
    Q = torch.from_numpy(Q).to(device)

    print("SLIC costs time: {}".format(SLIC_Time))
    print("start generate hypergraph")
    if isinstance(A, np.ndarray):
        pass
    elif isinstance(A, torch.Tensor):
        A = A.cpu().numpy()
    # A = A.numpy()
    hyperedges = []
    # # from Graph
    hyperedge_A = convert_A_to_list(A)
    hyperedges.extend(hyperedge_A)
    hyperedge_1hop = Get_k_hop(1, A)
    hyperedges.extend(hyperedge_1hop)
    # hyperedges.extend(hyperedge_2hop)
    # # #  from Feature
    S_tensor = torch.from_numpy(S)
    hyperedge_2hop = Get_k_hop(2, A)
    # hyperedge_3hop = Get_k_hop(3, A)
    if FLAG == 1:
        knn_num = 6
    if FLAG == 2:
        knn_num = 9
    if FLAG == 3:
        knn_num = 12
    if FLAG == 4:
        knn_num = 6
    hyperedge_knn = knn_neighbors(S_tensor, hyperedge_2hop, K=knn_num)
    hyperedges.extend(hyperedge_knn)
    # 计算权重，度矩阵，度量矩阵
    hyperedges_num = len(hyperedges)
    w_e = generate_W_e(S_tensor, hyperedges, superpixel_center)
    W = torch.diag(w_e)
    superpixel_count = S.shape[0]
    nodes = list(range(superpixel_count))
    incidence_mat_H = build_incidence_matrix(nodes, hyperedges)
    D_e, D_v = cal_Dv_De(incidence_mat_H)
    print("successfully generate hypergraph")
    A = torch.from_numpy(A).to(device)
    incidence_mat_H = incidence_mat_H.to(device)
    W = W.to(device)

    for curr_seed in Seed_List:
        # 随机选取指定数量训练，最多不超过该类地物一半。方式：给出训练数据与测试数据的GT
        random.seed(curr_seed)
        gt_reshape = np.reshape(gt, [-1])
        train_rand_idx = []
        val_rand_idx = []

        for i in range(class_count):
            idx = np.where(gt_reshape == i + 1)[-1]
            samplesCount = len(idx)
            rand_list = [j for j in range(samplesCount)]  # 用于随机的列表

            if sample_type == 'ratio':  # 按每类比例选取
                rand_idx = random.sample(rand_list,
                                         np.ceil(samplesCount * train_ratio).astype('int32'))  # 随机数数量 四舍五入(改为上取整)
                rand_real_idx_per_class = idx[rand_idx]
                train_rand_idx.append(rand_real_idx_per_class)

            elif sample_type == 'same':
                real_train_samples_per_class = train_samples_per_class
                if real_train_samples_per_class > samplesCount/2:
                    real_train_samples_per_class = int(samplesCount/2)
                    rand_idx = random.sample(rand_list, real_train_samples_per_class)  # 随机数数量 向下取整
                else:
                    rand_idx = random.sample(rand_list, real_train_samples_per_class)
                rand_real_idx_per_class_train = idx[rand_idx[0:real_train_samples_per_class]]
                train_rand_idx.append(rand_real_idx_per_class_train)

            # print('class ', i, ' ', len(rand_idx))  # 输出每类选取样本数量
        # train_rand_idx = np.array(train_rand_idx)
        val_rand_idx = np.array(val_rand_idx)
        train_data_index = []

        for c in range(len(train_rand_idx)):
            a = train_rand_idx[c]
            for j in range(a.shape[0]):
                train_data_index.append(a[j])

        # training set
        train_data_index = np.array(train_data_index)
        train_data_index = set(train_data_index)
        # all GT
        all_data_index = [i for i in range(len(gt_reshape))]
        all_data_index = set(all_data_index)

        # background
        background_idx = np.where(gt_reshape == 0)[-1]
        background_idx = set(background_idx)
        test_data_index = all_data_index - train_data_index - background_idx

        # 从测试集中随机选取部分样本作为验证集
        val_data_count = int(val_samples)  # 验证集数量
        val_data_index = random.sample(test_data_index, val_data_count)
        val_data_index = set(val_data_index)

        test_data_index = test_data_index - val_data_index
        # 将训练集 验证集 测试集 整理
        test_data_index = list(test_data_index)
        train_data_index = list(train_data_index)
        val_data_index = list(val_data_index)

        # 获取训练样本的标签图
        train_samples_gt = np.zeros(gt_reshape.shape)
        for i in range(len(train_data_index)):
            train_samples_gt[train_data_index[i]] = gt_reshape[train_data_index[i]]
            pass

        # 获取测试样本的标签图
        test_samples_gt = np.zeros(gt_reshape.shape)
        for i in range(len(test_data_index)):
            test_samples_gt[test_data_index[i]] = gt_reshape[test_data_index[i]]
            pass

        Test_GT = np.reshape(test_samples_gt, [m, n])  # 测试样本图

        # 获取验证集样本的标签图
        val_samples_gt = np.zeros(gt_reshape.shape)
        for i in range(len(val_data_index)):
            val_samples_gt[val_data_index[i]] = gt_reshape[val_data_index[i]]
            pass

        train_samples_gt = np.reshape(train_samples_gt, [height, width])
        test_samples_gt = np.reshape(test_samples_gt, [height, width])
        val_samples_gt = np.reshape(val_samples_gt, [height, width])

        train_samples_gt_onehot = GT_To_One_Hot(train_samples_gt, class_count)
        test_samples_gt_onehot = GT_To_One_Hot(test_samples_gt, class_count)
        val_samples_gt_onehot = GT_To_One_Hot(val_samples_gt, class_count)

        train_samples_gt_onehot = np.reshape(train_samples_gt_onehot, [-1, class_count]).astype(int)
        test_samples_gt_onehot = np.reshape(test_samples_gt_onehot, [-1, class_count]).astype(int)
        val_samples_gt_onehot = np.reshape(val_samples_gt_onehot, [-1, class_count]).astype(int)

        ############制作训练数据和测试数据的gt掩膜.根据GT将带有标签的像元设置为全1向量##############
        # 训练集
        train_label_mask = np.zeros([m * n, class_count])
        temp_ones = np.ones([class_count])
        train_samples_gt = np.reshape(train_samples_gt, [m * n])
        for i in range(m * n):
            if train_samples_gt[i] != 0:
                train_label_mask[i] = temp_ones
        train_label_mask = np.reshape(train_label_mask, [m * n, class_count])

        # 测试集
        test_label_mask = np.zeros([m * n, class_count])
        # temp_ones = np.ones([class_count])
        test_samples_gt = np.reshape(test_samples_gt, [m * n])
        for i in range(m * n):
            if test_samples_gt[i] != 0:
                test_label_mask[i] = temp_ones
        test_label_mask = np.reshape(test_label_mask, [m * n, class_count])

        # 验证集
        val_label_mask = np.zeros([m * n, class_count])
        # temp_ones = np.ones([class_count])
        val_samples_gt = np.reshape(val_samples_gt, [m * n])
        for i in range(m * n):
            if val_samples_gt[i] != 0:
                val_label_mask[i] = temp_ones
        val_label_mask = np.reshape(val_label_mask, [m * n, class_count])


        print("seed:{}".format(str(curr_seed)))

        # 转到GPU

        train_samples_gt = torch.from_numpy(train_samples_gt.astype(np.float32)).to(device)
        test_samples_gt = torch.from_numpy(test_samples_gt.astype(np.float32)).to(device)
        val_samples_gt = torch.from_numpy(val_samples_gt.astype(np.float32)).to(device)

        train_samples_gt_onehot = torch.from_numpy(train_samples_gt_onehot.astype(np.float32)).to(device)
        test_samples_gt_onehot = torch.from_numpy(test_samples_gt_onehot.astype(np.float32)).to(device)
        val_samples_gt_onehot = torch.from_numpy(val_samples_gt_onehot.astype(np.float32)).to(device)

        train_label_mask = torch.from_numpy(train_label_mask.astype(np.float32)).to(device)
        test_label_mask = torch.from_numpy(test_label_mask.astype(np.float32)).to(device)
        val_label_mask = torch.from_numpy(val_label_mask.astype(np.float32)).to(device)

        net_input = np.array(data, np.float32)
        net_input = torch.from_numpy(net_input.astype(np.float32)).to(device)


        net = model.HGCN_MHF(height, width, bands, class_count, Q, W_e=W,
                              Hyperedge=hyperedges, H=incidence_mat_H, D_v=D_v, D_e=D_e)

        print('参数量:'+str(sum(p.numel() for p in net.parameters())))
        # net = HGNNP.CEGCN(height, width, bands, class_count, Q, A)

        # print("parameters", net.parameters(), len(list(net.parameters())))
        net.to(device)


        def compute_loss(predict: torch.Tensor, real_label_onehot: torch.Tensor, real_label_mask: torch.Tensor):
            real_labels = real_label_onehot
            we = -torch.mul(real_labels, torch.log(predict))
            we = torch.mul(we, real_label_mask)  # we=-real_labels*log(predict)*real_label_mask
            pool_cross_entropy = torch.sum(we)
            return pool_cross_entropy


        zeros = torch.zeros([m * n]).to(device).float()


        def evaluate_performance(network_output, train_samples_gt, train_samples_gt_onehot, require_AA_KPP=False,
                                 printFlag=True):
            if require_AA_KPP == False:
                with torch.no_grad():
                    available_label_idx = (train_samples_gt != 0).float()  # 有效标签的坐标,用于排除背景
                    available_label_count = available_label_idx.sum()  # 有效标签的个数
                    correct_prediction = torch.where(
                        torch.argmax(network_output, 1) == torch.argmax(train_samples_gt_onehot, 1), available_label_idx,
                        zeros).sum()
                    OA = correct_prediction.cpu() / available_label_count
                    return OA
            else:
                with torch.no_grad():
                    # 计算OA
                    available_label_idx = (train_samples_gt != 0).float()  # 有效标签的坐标,用于排除背景
                    available_label_count = available_label_idx.sum()  # 有效标签的个数
                    correct_prediction = torch.where(
                        torch.argmax(network_output, 1) == torch.argmax(train_samples_gt_onehot, 1), available_label_idx,
                        zeros).sum()
                    OA = correct_prediction.cpu() / available_label_count
                    OA = OA.cpu().numpy()

                    # 计算AA
                    zero_vector = np.zeros([class_count])
                    output_data = network_output.cpu().numpy()
                    train_samples_gt = train_samples_gt.cpu().numpy()
                    train_samples_gt_onehot = train_samples_gt_onehot.cpu().numpy()
                    output_data = np.reshape(output_data, [m * n, class_count])
                    idx = np.argmax(output_data, axis=-1)
                    for z in range(output_data.shape[0]):
                        if ~(zero_vector == output_data[z]).all():
                            idx[z] += 1
                    # idx = idx + train_samples_gt
                    count_perclass = np.zeros([class_count])
                    correct_perclass = np.zeros([class_count])
                    for x in range(len(train_samples_gt)):
                        if train_samples_gt[x] != 0:
                            count_perclass[int(train_samples_gt[x] - 1)] += 1
                            if train_samples_gt[x] == idx[x]:
                                correct_perclass[int(train_samples_gt[x] - 1)] += 1
                    test_AC_list = correct_perclass / count_perclass
                    test_AA = np.average(test_AC_list)

                    # 计算KPP
                    test_pre_label_list = []
                    test_real_label_list = []
                    output_data = np.reshape(output_data, [m * n, class_count])
                    idx = np.argmax(output_data, axis=-1)
                    idx = np.reshape(idx, [m, n])
                    for ii in range(m):
                        for jj in range(n):
                            if Test_GT[ii][jj] != 0:
                                test_pre_label_list.append(idx[ii][jj] + 1)
                                test_real_label_list.append(Test_GT[ii][jj])
                    test_pre_label_list = np.array(test_pre_label_list)
                    test_real_label_list = np.array(test_real_label_list)
                    kappa = metrics.cohen_kappa_score(test_pre_label_list.astype(np.int16),
                                                      test_real_label_list.astype(np.int16))
                    test_kpp = kappa

                    # 输出
                    if printFlag:
                        print("test OA=", OA, "AA=", test_AA, 'kpp=', test_kpp)
                        print('acc per class:')
                        print(test_AC_list)

                    OA_ALL.append(OA)
                    AA_ALL.append(test_AA)
                    KPP_ALL.append(test_kpp)
                    AVG_ALL.append(test_AC_list)

                    # 保存数据信息
                    f = open('./results/' + dataset_name + '_results.txt', 'a+')
                    str_results = '\n======================' \
                                  + " learning rate=" + str(learning_rate) \
                                  + " epochs=" + str(max_epoch) \
                                  + " train number=" + str(train_num) \
                                  + " val num=" + str(val_samples) \
                                  + " ======================" \
                                  + "\nOA=" + str(OA) \
                                  + "\nAA=" + str(test_AA) \
                                  + '\nkpp=' + str(test_kpp) \
                                  + '\nacc per class:' + str(test_AC_list) + "\n"
                    # + '\ntrain time:' + str(time_train_end - time_train_start) \
                    # + '\ntest time:' + str(time_test_end - time_test_start) \
                    f.write(str_results)
                    f.close()
                    return OA


        # 训练
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4)  # ,weight_decay=0.0001
        best_loss = 99999
        net.train()
        tic1 = time.perf_counter()
        for i in range(max_epoch+1):
            optimizer.zero_grad()  # zero the gradient buffers
            output = net(net_input)
            loss = compute_loss(output, train_samples_gt_onehot, train_label_mask)
            loss.backward(retain_graph=False)
            optimizer.step()  # Does the update
            if i % 10 == 0:
                with torch.no_grad():
                    net.eval()
                    output = net(net_input)
                    trainloss = compute_loss(output, train_samples_gt_onehot, train_label_mask)
                    trainOA = evaluate_performance(output, train_samples_gt, train_samples_gt_onehot)
                    valloss = compute_loss(output, val_samples_gt_onehot, val_label_mask)
                    valOA = evaluate_performance(output, val_samples_gt, val_samples_gt_onehot)
                    print("{}\ttrain loss={}\t train OA={} val loss={}\t val OA={}".format(str(i + 1), trainloss, trainOA,
                                                                                           valloss, valOA))

                    if valloss < best_loss:
                        best_loss = valloss
                        torch.save(net.state_dict(), "./model/best_model.pt")
                        print('save model...')
                torch.cuda.empty_cache()
                net.train()
        toc1 = time.perf_counter()
        print("\n\n====================training done. starting evaluation...========================\n")
        training_time = toc1 - tic1
        Train_Time_ALL.append(training_time)

        torch.cuda.empty_cache()
        with torch.no_grad():
            net.load_state_dict(torch.load("./model/best_model.pt"))
            net.eval()
            tic2 = time.perf_counter()
            output = net(net_input)
            toc2 = time.perf_counter()
            testloss = compute_loss(output, test_samples_gt_onehot, test_label_mask)
            testOA = evaluate_performance(output, test_samples_gt, test_samples_gt_onehot, require_AA_KPP=True,
                                          printFlag=False)
            print("{}\ttest loss={}\t test OA={}".format(str(i + 1), testloss, testOA))
            # 计算
            classification_map = torch.argmax(output, 1).reshape([height, width]).cpu()
            palette = color_map_dict.get(dataset_name)
            map = palette[classification_map]
            map[gt == 0, :] = np.array([255, 255, 255], dtype='uint8')
            plt.figure()
            plt.imshow(map, cmap='jet')
            plt.xticks([])
            plt.yticks([])
            plt.savefig(os.path.join("./results/", 'HGCN-MHF_' + dataset_name + str(train_num ) + '_' + str(testOA)+'.png'), bbox_inches='tight', dpi=300)
            testing_time = toc2 - tic2
            Test_Time_ALL.append(testing_time)
        torch.cuda.empty_cache()
        del net

    OA_ALL = np.array(OA_ALL)
    AA_ALL = np.array(AA_ALL)
    KPP_ALL = np.array(KPP_ALL)
    AVG_ALL = np.array(AVG_ALL)
    Train_Time_ALL = np.array(Train_Time_ALL)
    Test_Time_ALL = np.array(Test_Time_ALL)

    print("\ntrain_num={}".format(train_num ),
          "\n==============================================================================")
    print('OA=', np.mean(OA_ALL), '+-', np.std(OA_ALL))
    print('AA=', np.mean(AA_ALL), '+-', np.std(AA_ALL))
    print('Kpp=', np.mean(KPP_ALL), '+-', np.std(KPP_ALL))
    print('AVG=', np.mean(AVG_ALL, 0), '+-', np.std(AVG_ALL, 0))
    print("Average training time:{}".format(np.mean(Train_Time_ALL)))
    print("Average testing time:{}".format(np.mean(Test_Time_ALL)))

    # 保存数据信息
    f = open('./results/' + dataset_name + '_results.txt', 'a+')
    str_results = '\n\n************************************************' \
                  + "\ntrain_num={}".format(train_num) \
                  + "\nknn_num={}".format(knn_num) \
                  + "\nscale={}".format(Scale) \
                  + '\nOA=' + str(np.mean(OA_ALL)) + '+-' + str(np.std(OA_ALL)) \
                  + '\nAA=' + str(np.mean(AA_ALL)) + '+-' + str(np.std(AA_ALL)) \
                  + '\nKpp=' + str(np.mean(KPP_ALL)) + '+-' + str(np.std(KPP_ALL)) \
                  + '\nAVG=' + str(np.mean(AVG_ALL, 0)) + '+-' + str(np.std(AVG_ALL, 0)) \
                  + "\nAverage training time:{}".format(np.mean(Train_Time_ALL)) \
                  + "\nAverage testing time:{}".format(np.mean(Test_Time_ALL))
    f.write(str_results)
    f.close()


