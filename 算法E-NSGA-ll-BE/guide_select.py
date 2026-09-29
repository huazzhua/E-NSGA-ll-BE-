"""
选择算子
"""
import math
import random
import numpy as np
np.random.seed(1)
random.seed(1)

# 第六步  选择新的种群：淘汰不优秀的个体

def g_select(pool, pops, fits, ranks, distances):
	num_select = 3
	nPop, nChr = pops.shape
	nF = fits.shape[1]
	newPops = np.zeros((pool, nChr))
	newFits = np.zeros((pool, nF))
	indices = np.arange(nPop).tolist()
	i = 0
	while i < pool:
		lambda1 = np.zeros(num_select)
		lambda2 = np.zeros(num_select)
		P = np.zeros(num_select)
		accP = np.zeros(num_select)
		index = random.sample(indices, num_select)  # 随机挑选个体
		tempPops = pops[index, :]
		tempFits = fits[index, :]
		tempDistances = distances[index]
		tempRanks = ranks[index]
		error = tempFits[:, 1]
		sortIndex1 = np.argsort(error)
		sortIndex2 = np.argsort(-tempDistances)
		c = 0
		for m in sortIndex2:
			c += 1
			lambda2[m] = c
		j = 0
		for m in sortIndex1:
			j += 1
			lambda1[m] = j
			P[m] = math.exp(-(lambda1[m] * lambda2[m] * (tempRanks[m] + 1) **0.5))  # 对等级开平方，削弱惩罚
		p_sum = sum(P)
		for n in range(len(P)):
			P[n] = P[n] / p_sum
		# 轮盘赌
		k = 0
		for n in range(len(P)):
			if k == 0:
				accP[n] = P[n]
				k = 1
			else:
				accP[n] = accP[n - 1] + P[n]
		r = random.random()
		for q in range(len(accP)):
			if k == 1:
				k = 2
				if 0 <= r <= accP[q]:
					newPops[i] = tempPops[q]
					newFits[i] = tempFits[q]
			elif accP[q - 1] < r <= accP[q]:
				newPops[i] = tempPops[q]
				newFits[i] = tempFits[q]

		i += 1
	return newPops, newFits

