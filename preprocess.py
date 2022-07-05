import scipy.io as sio
import os
from nilearn.connectome import ConnectivityMeasure
import numpy as np
import torch
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from scipy.spatial import distance
import random
from nilearn import connectome
from scipy.io import loadmat
from grakel.kernels import ShortestPath
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
conn_measure = connectome.ConnectivityMeasure(kind='correlation')


eps = 2.2204e-16
def normalize_graph(A):
    deg_inv_sqrt = A.sum(dim=-1).clamp(min=0.).pow(-0.5)
    A = deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(-2)
    return A

dataname = 'FTD'
rootpath = r"your_dir"
dim = 512


def ori_data(subject_IDs, labels):
    num_nodes = subject_IDs.size
    Time = []

    y = np.zeros([num_nodes, 1])

    for i in range(num_nodes):
        y[i] = int(labels[subject_IDs[i]]) - 1

    for name in subject_IDs:
        name_path = os.path.join(rootpath, name)
        feature = []
        for time_dir in os.listdir(name_path):
            if time_dir[-3:] == ".1D":
                feature_dir = os.path.join(name_path, time_dir)
                i = 0
                for line in open(feature_dir, "r"):
                    if i == 0:
                        i += 1
                        continue
                    temp = line[:-1].split('\t')
                    feature.append([float(x) for x in temp])
        Time.append(np.array(feature))

    random_list = []
    random_list_np = np.array([])
    random_list_selected = []
    time = Time[0].shape[0]
    windownumber = 5
    windowsize = int(time/windownumber)
    strite = 5
    mask = (np.triu_indices(110)[0], np.triu_indices(110)[1] + 1)
    for i in range(windownumber):
        all_feature = []
        for x in Time:
            time = x.shape[0]
            k = windowsize * i

            ppc = conn_measure.fit_transform([x[k:k + windowsize]])[0]
            # ppc = np.corrcoef(x[k:k+windowsize].T)
            feature_i = ppc[mask]
            # print(np.where(np.isnan(feature_i)))
            temp = feature_i.astype(np.float32)
            all_feature.append(temp)

        random_list.append(all_feature)
        try:
            random_list_np = np.concatenate((random_list_np, all_feature), 0)
        except:
            random_list_np = all_feature

    A_selected_list = []
    A_selected_mean = []
    pca = PCA(n_components=dim)
    pca.fit(random_list_np)
    for feature in random_list:
        x_selected = pca.transform(feature)
        random_list_selected.append(x_selected)

        distv = distance.pdist(feature, metric='correlation')
        dist = distance.squareform(distv)
        sigma = np.mean(dist)
        graph = np.exp(- dist ** 2 / (2 * sigma ** 2))
        A_selected_list.append(graph)

    data_all_list = np.array({'t': windownumber,
                              'data_feature_list': np.array(random_list),
                              'data_feature_list_st': np.array(random_list_selected),
                              'data_label': y,
                              'data_graph_list': np.array(A_selected_list).astype(np.float32),
                              },
                             dtype=object)
    np.save(r'your_dir/Ori_' + dataname +'.npy', data_all_list, allow_pickle=True)

    print("Done")

def load_data(dataset='FTD', seed = 4 ,fold = 5, remake = True, splite = 20  ):  ##handwritten citeseer cora flower
    dir = 'data/data/' + dataset
    topk = 10
    # mat_dir_2 = dir + 'NC'
    mat_dir_1 = dir
    info1 = loadmat(mat_dir_1) #data/handwritten.mat
    dir_root_np = os.path.join('.\data',dataset,str(topk),str(fold),str(splite), str(seed))
    if not os.path.exists(dir_root_np):
        os.makedirs(dir_root_np)

    save_name = dir_root_np+'data.npy'
    if not os.path.exists(save_name) or remake:
        # data_FTD_HHF = np.array(info1['HHF']).astype(np.float32)
        # data_FTD_HLF = np.array(info1['HLF']).astype(np.float32)
        # data_FTD_LHF = np.array(info1['LHF']).astype(np.float32)
        # data_FTD_LLF = np.array(info1['LLF']).astype(np.float32)
        data_FTD_view1 = np.array(info1['view1']).astype(np.float32)
        data_FTD_view2 = np.array(info1['view2']).astype(np.float32)
        data_FTD_view3 = np.array(info1['view3']).astype(np.float32)
        data_FTD_view4 = np.array(info1['view4']).astype(np.float32)
        data_FTD_view5 = np.array(info1['view5']).astype(np.float32)

        A_FTD_view1 = np.array(info1['graph1']).astype(np.float32)
        A_FTD_view2 = np.array(info1['graph2']).astype(np.float32)
        A_FTD_view3 = np.array(info1['graph3']).astype(np.float32)
        A_FTD_view4 = np.array(info1['graph4']).astype(np.float32)
        A_FTD_view5 = np.array(info1['graph5']).astype(np.float32)

        # data_all = np.stack([data_FTD_view1, data_FTD_view2, data_FTD_view3, data_FTD_view4, data_FTD_view5])
        data_all = np.stack([A_FTD_view1, A_FTD_view2, A_FTD_view3, A_FTD_view4, A_FTD_view5])
        data_feature_4000 =  np.array(info1['ori_feature']).astype(np.float32)
        gk = ShortestPath(normalize=True)
        list = []
        # for g in info1['ori_grap']:
        #     list.append([np.matrix(g)])
        # K_train = gk.fit_transform(list)

        N = data_feature_4000.shape[0]
        ################STA|matric|###############
        data_graph_4000 = torch.from_numpy(np.array(info1['kernel_matrix']).astype(np.float32))
        maxadj = torch.topk(data_graph_4000, k=25, dim=1, sorted=False,
                            largest=True).values[:, -1].view(N, 1).repeat(1, N)
        maxadj = (data_graph_4000 >= maxadj) + 0
        sparse_graph = normalize_graph(data_graph_4000 * maxadj)
        I_input = torch.eye(N)
        # data_graph_4000 = sparse_graph * 0.8+ I_input * 0.2
        data_graph_4000 = data_graph_4000 #+ I_input
        ################END|matric|###############

        data_label = np.array(info1['label'])


        view = 5
        data_graph_list = []
        ori_graph_list = []

        for data_feature in [A_FTD_view1, A_FTD_view2, A_FTD_view3, A_FTD_view4, A_FTD_view5]:
            # data_feature_4000_list_view = []
            ori_graph_list_view = []
            data_graph_list_view = []
            for i in range(N):
                ori_graph = torch.from_numpy(data_feature[i])
                maxadj = torch.topk(ori_graph, k=topk, dim=1, sorted=False,
                                    largest=True).values[:, -1].view(90,1).repeat(1,90)
                maxadj = (ori_graph >= maxadj) + 0
                sparse_graph = normalize_graph(ori_graph * maxadj)
                I_input = torch.eye(90)
                sparse_graph = sparse_graph*0.5 + I_input *0.5
                data_graph_list_view.append(sparse_graph.numpy())
                ori_graph_list_view.append(ori_graph.numpy())

                # data_feature_4000 = None
                # for j in range(90):
                #     if data_feature_4000 is None:
                #         data_feature_4000 = ori_graph[j][j:]
                #     else:
                #         data_feature_4000 = np.hstack([data_feature_4000, ori_graph[j][j:]])
                # data_feature_4000_list_view.append(data_feature_4000.astype(np.float32))

            data_graph_list.append(np.array(data_graph_list_view))
            ori_graph_list.append(np.array(ori_graph_list_view))
            # data_feature_4000_list.append(np.array(data_feature_4000_list_view))

        kf = StratifiedShuffleSplit(n_splits=5, train_size=splite/100)
        train_list = []
        test_list = []
        for train_index, test_index in kf.split(data_feature_4000, data_label):
            train_list.append(train_index)
            test_list.append(test_index)
            # if splite == 20:
            #     train_list.append(test_index)
            #     test_list.append(train_index)
            # else:
            #     train_list.append(train_index)
            #     test_list.append(test_index)
        my_train_list = []
        my_test_list = []
        # if splite == 20:
        #     my_train_list.append(train_list[])
        #     my_test_list.append()

        save_np = np.array({'n': N,
                            'data_feature':data_all,
                            'data_feature_4000':data_feature_4000,
                            'data_label':np.array(data_label),
                            'ori_graph': np.array(ori_graph_list),
                            'data_graph': np.array(data_graph_list),
                            'data_graph_4000':np.array(data_graph_4000),
                            'train_list':np.array(train_list),
                            'test_list': np.array(test_list),
                            }, dtype=object)

        np.save(dir_root_np+'data.npy', save_np, allow_pickle=True)
    full_data = np.load(os.path.join(save_name), allow_pickle=True).tolist()
    return full_data