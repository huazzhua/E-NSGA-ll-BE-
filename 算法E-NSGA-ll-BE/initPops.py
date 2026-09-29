"""
种群初始化
"""
import numpy as np
import random

np.random.seed(1)
random.seed(1)

# 第一步

def initPops(nPop, nChr, Score):
    pops = np.zeros((nPop, nChr))
    # 归一化特征重要性（转为概率）
    norm_score = (Score - np.min(Score)) / (np.max(Score) - np.min(Score) + 1e-8)
    for i in range(nPop):
        # 每个个体的特征数量随机（保留20%-80%特征）
        n_feats = np.random.randint(int(0.2 * nChr), int(0.8 * nChr) + 1)
        # 按重要性概率选择特征（高重要性特征更可能被选中）
        selected = np.random.choice(nChr, size=n_feats, replace=False, p=norm_score / np.sum(norm_score))
        # 如果数据集0太多就用下面这份代码
        # # 改成安全代码
        # total = np.sum(norm_score)
        #
        # if total == 0:
        #     # 分数全0，随机均匀选择
        #     p = None
        # else:
        #     p = norm_score / total
        #
        # # 安全选择
        # valid_indices = np.where(norm_score > 0)[0]
        # n_valid = len(valid_indices)
        #
        # if n_valid >= n_feats:
        #     selected = np.random.choice(nChr, size=n_feats, replace=False, p=p)
        # else:
        #     # 非零特征不够，全部选中 + 随机补全
        #     selected = valid_indices.tolist()
        #     remaining = n_feats - n_valid
        #     others = np.random.choice(
        #         [i for i in range(nChr) if i not in valid_indices],
        #         size=remaining,
        #         replace=False
        #     )
        #     selected = np.concatenate([selected, others])
        pops[i, selected] = 1
    return pops