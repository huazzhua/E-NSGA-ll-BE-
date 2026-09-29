import numpy as np
import copy
np.random.seed(1)

# 第四步

def crossover(pops, pc):
    chrPops = np.copy(pops)  # 浅拷贝保持高效
    nPop, nChr = chrPops.shape
    for i in range(0, nPop - 1, 2):  # 成对交叉，避免越界
        if np.random.rand() < pc:  # 按概率触发交叉
            # 随机选择两个不同的交叉点（确保c1 < c2）
            c1 = np.random.randint(0, nChr - 1)
            c2 = np.random.randint(c1 + 1, nChr)
            # 交换[c1, c2)区间内的特征（保留两端的优秀组合）
            chrPops[i, c1:c2], chrPops[i+1, c1:c2] = \
                chrPops[i+1, c1:c2].copy(), chrPops[i, c1:c2].copy()
    return chrPops