"""
快速非支配排序
"""
import random

import numpy as np

np.random.seed(1)
random.seed(1)

# 第二步

def nonDominationSort(pops, fits):
	"""快速非支配排序算法
	Params:
		pops: 种群，nPop * nChr 数组
		fits: 适应度， nPop * nF 数组
	Return:
		ranks: 每个个体所对应的等级，一维数组
	"""
	nPop = pops.shape[0]
	nF = 2  # 目标函数的个数
	ranks = np.zeros(nPop, dtype=np.int32)
	nPs = np.zeros(nPop)  # 每个个体p被支配解的个数
	sPs = []  # 每个个体支配的解的集合，把索引放进去
	for i in range(nPop):
		iSet = []  # 解i的支配解集
		for j in range(nPop):
			if i == j:
				continue
			isDom1 = fits[i] <= fits[j]
			isDom2 = fits[i] < fits[j]
			if np.sum(isDom1) == nF and np.sum(isDom2) >= 1:
				iSet.append(j)
			if np.sum(~isDom2) == nF and np.sum(~isDom1) >= 1:
				nPs[i] += 1
		sPs.append(iSet)  # 添加i支配的解的索引
	r = 0  # 当前等级为 0， 等级越低越好
	indices = np.arange(nPop)
	while sum(nPs == 0) != 0:
		rIdices = indices[nPs == 0]  # 当前被支配数为0的索引
		ranks[rIdices] = r
		for rIdx in rIdices:
			iSet = sPs[rIdx]
			nPs[iSet] -= 1
		nPs[rIdices] = -1  # 当前等级的被支配数设置为负数
		r += 1
	return ranks


# 拥挤度排序算法
def crowdingDistanceSort(pops, fits, ranks):
	"""拥挤度排序算法
	Params:
		pops: 种群，nPop * nChr 数组
		fits: 适应度， nPop * nF 数组
		ranks：每个个体对应的等级，一维数组
	Return：
		dis: 每个个体的拥挤度，一维数组
	"""
	nPop = pops.shape[0]
	nF = fits.shape[1]  # 目标个数
	dis = np.zeros(nPop)
	nR = ranks.max()  # 最大等级
	indices = np.arange(nPop)
	for r in range(nR + 1):
		rIdices = indices[ranks == r]  # 当前等级种群的索引
		rPops = pops[ranks == r]  # 当前等级的种群
		rFits = fits[ranks == r]  # 当前等级种群的适应度
		rSortIdices = np.argsort(rFits, axis=0)  # 对纵向排序的索引
		rSortFits = np.sort(rFits, axis=0)
		fMax = np.max(rFits, axis=0)
		fMin = np.min(rFits, axis=0)
		n = len(rIdices)
		for i in range(nF):
			orIdices = rIdices[rSortIdices[:, i]]  # 当前操作元素的原始位置
			dis[orIdices[0]] = np.inf
			dis[orIdices[n - 1]] = np.inf
			if fMax[i] == fMin[i]:
				continue
			j = 1
			while n > 2 and j < n - 1:
				dis[orIdices[j]] += (rSortFits[j + 1][i] - rSortFits[j - 1][i]) / \
				                    (fMax[i] - fMin[i])
				j += 1
	return dis
