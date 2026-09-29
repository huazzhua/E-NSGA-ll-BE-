
import random
import numpy as np
np.random.seed(1)
random.seed(1)

def e_mutate(pops, pm, scores=None, fits=None):  # 新增fits参数（个体适应度）
    nPop = pops.shape[0]
    nChr = pops.shape[1]
    for i in range(nPop):
        # 对适应度差的个体（错误率高），提高整体变异概率并优先变异低重要性特征
        if fits is not None:
            error_rate = fits[i][1]  # 错误率（适应度第二列）
            # 适应度越差，基础变异概率越高（1.2~2倍）
            dynamic_pm = pm * (1 + min(error_rate * 2, 1))  # 错误率高的个体变异更激进
        else:
            dynamic_pm = pm

        for j in range(nChr):
            if scores is not None:
                norm_score = (scores[j] - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-8)
                current_pm = dynamic_pm * (1 - norm_score)  # 仍保留高重要性特征低变异的逻辑
            else:
                current_pm = dynamic_pm

            # 额外逻辑：对低适应度个体，强制开启部分高重要性未被选中的特征
            if fits is not None and error_rate > np.mean(fits[:,1]) and pops[i][j] == 0 and norm_score > 0.7:
                current_pm += 0.3  # 高重要性特征未被选中时，强制提高变异概率（更易开启）

            if np.random.rand() < current_pm:
                pops[i][j] = 1 - pops[i][j]
    return pops