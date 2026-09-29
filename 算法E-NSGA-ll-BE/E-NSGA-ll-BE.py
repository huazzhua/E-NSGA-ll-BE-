from sklearn import preprocessing
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from crossover import *
from env_select import env_select
from evolved_mutate import *
from guide_select import *
from initPops import *
from nonDominationSort import *
import pandas as pd
import time

# 输出结果导入
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

np.random.seed(1)
random.seed(1)


def top_select(X, score, percentage=0.05):  # 根据特征的重要性评分，选择最优的特征子集
    """
    以百分比的形式保留特征和数据
    :param X: ndarray(x,y)
    :param score: list
    :param percentage: float
    :return:
    X: ndarray(x,y)
    score: list or ndarray(x,)
    """
    score = np.ndarray.flatten(score)
    length = int(percentage * len(score))
    sorted_index = np.argsort(-score)
    sorted_index = sorted_index[0:length]
    X = X[:, sorted_index]
    score = score[sorted_index]
    return X, score


def reliefFScore(X, y, k=5):  # 计算每个特征的重要性，基于最近邻的距离来评估特征。
    """
    计算每个特征的重要性
    :param X: ndarray(x,y)
    :param y: ndarray(x,)
    :param k: int
    :return: list
    """
    n_samples, n_features = X.shape

    # calculate pairwise distances between instances
    distance = pairwise_distances(X, metric='manhattan')

    score = np.zeros(n_features)

    # the number of sampled instances is equal to the number of total instances
    for idx in range(n_samples):
        near_hit = []
        near_miss = dict()

        self_fea = X[idx, :]
        c = np.unique(y).tolist()

        stop_dict = dict()
        for label in c:
            stop_dict[label] = 0
        del c[c.index(y[idx])]

        p_dict = dict()
        p_label_idx = float(len(y[y == y[idx]])) / float(n_samples)

        for label in c:
            p_label_c = float(len(y[y == label])) / float(n_samples)
            p_dict[label] = p_label_c / (1 - p_label_idx)
            near_miss[label] = []

        distance_sort = []
        distance[idx, idx] = np.max(distance[idx, :])

        for i in range(n_samples):
            distance_sort.append([distance[idx, i], int(i), y[i]])
        distance_sort.sort(key=lambda x: x[0])

        for i in range(n_samples):
            # find k nearest hit points
            if distance_sort[i][2] == y[idx]:
                if len(near_hit) < k:
                    near_hit.append(distance_sort[i][1])
                elif len(near_hit) == k:
                    stop_dict[y[idx]] = 1
            else:
                # find k nearest miss points for each label
                if len(near_miss[distance_sort[i][2]]) < k:
                    near_miss[distance_sort[i][2]].append(distance_sort[i][1])
                else:
                    if len(near_miss[distance_sort[i][2]]) == k:
                        stop_dict[distance_sort[i][2]] = 1
            stop = True
            for (key, value) in stop_dict.items():
                if value != 1:
                    stop = False
            if stop:
                break

        # update reliefF score
        near_hit_term = np.zeros(n_features)
        for ele in near_hit:
            near_hit_term = np.array(abs(self_fea - X[ele, :])) + np.array(near_hit_term)

        near_miss_term = dict()
        for (label, miss_list) in near_miss.items():
            near_miss_term[label] = np.zeros(n_features)
            for ele in miss_list:
                near_miss_term[label] = np.array(abs(self_fea - X[ele, :])) + np.array(near_miss_term[label])
            score += near_miss_term[label] / (k * p_dict[label])
        score -= near_hit_term / k
    return score


def load_csv(filename):  # 加载 CSV 文件并将数据分解为特征和标签。
    """
    读取 filename 位置的数据集，并分解为数据与类标签   ·
    :param filename: str
    :return:
    X: ndarray(x,y)
    Y: ndarray(x,)
    """
    data = pd.read_csv(filename)
    # data.info()
    X = data.values[:, 0:-1]
    Y = data.values[:, -1]
    print("nSamples", X.shape[0])   #  样本数目
    print("nFeatures", X.shape[1])  # 特征数目
    _, Y = np.unique(Y, return_inverse=True)
    return X, Y


def function(pops):
    if pops.ndim == 1:
        pops = pops.reshape(1, -1)
    fits = np.zeros((len(pops), 2))
    # 提前划分一次验证集，避免重复划分
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=666)
    for j in range(pops.shape[0]):
        index = np.where(pops[j] != 0)[0]
        if np.count_nonzero(pops[j]) == 0:
            fits[j][0] = 1
            fits[j][1] = 1
        else:
            X_subset_train = X_train[:, index]
            X_subset_test = X_test[:, index]
            clf.fit(X_subset_train, Y_train)
            acc = clf.score(X_subset_test, Y_test)  # 单次验证
            fits[j][0] = np.sum(pops[j]) / dim
            fits[j][1] = 1 - acc
    return fits


def E_NSGA_ll_BE(nIter, nChr, nPop, pc, pm, score, feature_list):
    Iter = 1
    pops = initPops(nPop, nChr, score)
    fits = function(pops)

    # # 初始化 feature_list 确保它在使用前有初值
    # feature_list = np.arange(nChr).tolist()  # 用于存储特征索引的列表

    while Iter < nIter:
        Iter += 1
        # 动态pm：前期大（探索），后期小（收敛）
        min_pm = 0.05  # 最小变异概率
        current_pm = pm * (1 - Iter / nIter) + min_pm * (Iter / nIter)  # 从0.2平滑过渡到0.05

        ranks = nonDominationSort(pops, fits)  # 非支配排序
        distances = crowdingDistanceSort(pops, fits, ranks)  # 拥挤度
        pops, fits = g_select(nPop, pops, fits, ranks, distances)
        chr_pops = crossover(pops, pc)
        chr_pops = e_mutate(chr_pops, current_pm, scores=score, fits=fits)   # 使用改进的模型时加上，evolved_mutate（, scores=score, fits=fits）
        chr_fits = function(chr_pops)
        pops, fits = env_select(pops, fits, chr_pops, chr_fits)
        if pops.shape[1] > 100:
            if (Iter + 4) % 15 == 0:
                feature_counts = np.sum(pops, axis=0)
            if (Iter + 3) % 15 == 0:
                feature_counts += np.sum(pops, axis=0)
            if (Iter + 2) % 15 == 0:
                feature_counts += np.sum(pops, axis=0)
            if (Iter + 1) % 15 == 0:
                feature_counts += np.sum(pops, axis=0)
            if Iter % 15 == 0:
                feature_counts += np.sum(pops, axis=0)
                temp_f_list = []
                pos = np.where(feature_counts != 0)[0]
                for ii in range(len(pos)):
                    temp_f_list.append(feature_list[pos[ii]])
                pops = pops[:, feature_counts != 0]
                feature_list = copy.deepcopy(temp_f_list)
                fits = function(pops)
                feature_counts = np.zeros(pops.shape[1])
        if Iter == nIter:
            ranks = nonDominationSort(pops, fits)
            select_index = np.where(ranks == 0)[0]
            # 关键修复：过滤掉超出种群行数的索引（防止越界）
            select_index = select_index[select_index < pops.shape[0]]
            for s_index in select_index:
                position = np.arange(pops.shape[1])
                np.random.shuffle(position)
                for k in position:
                    if pops[s_index][k] == 1:
                        pops[s_index][k] = 0
                        temp = function(pops[s_index])
                        if temp[0][1] <= fits[s_index][1]:
                            fits[s_index][0] = np.copy(temp[0][0])
                            fits[s_index][1] = np.copy(temp[0][1])
                        else:
                            pops[s_index][k] = 1
                    elif pops[s_index][k] == 0:
                        pops[s_index][k] = 1
                        temp = function(pops[s_index])
                        if temp[0][1] < fits[s_index][1]:
                            fits[s_index][0] = np.copy(temp[0][0])
                            fits[s_index][1] = np.copy(temp[0][1])
                        else:
                            pops[s_index][k] = 0
    return pops,fits


if __name__ == "__main__":
    # 开始计时
    start_time = time.perf_counter()

    clf = KNeighborsClassifier(n_neighbors=3, algorithm='kd_tree', n_jobs=-1)

    nIter = 150
    nPop = 100
    pc = 0.7
    pm = 0.15
    file_path = r"E:\数据集\最小‐最大归一化方法数据集\colon.csv"  # 放数据集地址的地方
    X, Y = load_csv(file_path)

    minmax = preprocessing.MinMaxScaler()

    f_score = reliefFScore(X, Y)  # 计算每个

    X, score = top_select(X, f_score)

    kf = KFold(n_splits=5, shuffle=True, random_state=666)

    dim = X.shape[1]

    feature_list = np.arange(dim).tolist()

    final_pop, final_fits = E_NSGA_ll_BE(nIter, dim, nPop, pc, pm, score, feature_list)

    # 结束计时
    end_time = time.perf_counter()
    running_time = end_time - start_time

    print(f"算法运行时间: {running_time:.4f} 秒")

    # 排序：根据fits中的准确率（第二列）从小到大选择最优个体
    best_index = np.argmin(final_fits[:, 1])  # 获取准确率最小（即准确率最高）的索引
    best_solution = final_pop[best_index]
    print("最优特征子集:", best_solution)

    # 计算稀疏性
    sparsity = np.sum(best_solution) / len(best_solution)
    print(f"特征选择的稀疏性: {sparsity} (减少了 {100 * (1 - sparsity)}% 的特征)")

    # 输出准确率
    best_fitness = final_fits[best_index]  # 最优解的适应度
    print("最优特征子集的准确率:", 1 - best_fitness[1])  # 因为fits中的第二列是错误率，所以下面要减去

    # 提取最终非支配目标集
    ranks = nonDominationSort(final_pop, final_fits)
    pareto_fits = final_fits[ranks == 0]

    # HV需要的参数
    hv_ref_point = np.array([1.1, 1.1])

    print("Pareto目标值：")
    print(pareto_fits)

    print("HV参考点：")
    print(hv_ref_point)



