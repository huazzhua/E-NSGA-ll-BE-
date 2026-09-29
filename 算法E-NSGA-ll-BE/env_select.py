"""
种群的合并和优选
"""
import numpy as np

from nonDominationSort import *

np.random.seed(1)
random.seed(1)

# 第三步

def env_select(pops, fits, chrPops, chrFits):
	"""种群合并与优选
	Return:
		newPops, newFits
	"""
	nPop, nChr = pops.shape
	nF = fits.shape[1]
	# 合并父代种群和子代种群构成一个新种群
	MergePops = np.vstack((pops, chrPops))
	MergeFits = np.vstack((fits, chrFits))
	unique_index = np.unique(MergePops, axis=0, return_index=True)[1]
	MergePops = MergePops[unique_index]
	MergeFits = MergeFits[unique_index]
	if len(MergePops) < nPop:
		return MergePops, MergeFits
	MergeRanks = nonDominationSort(MergePops, MergeFits)
	MergeDistances = crowdingDistanceSort(MergePops, MergeFits, MergeRanks)
	max_r = np.max(MergeRanks)
	indices = np.arange(MergePops.shape[0])
	r = 0
	i = 0
	select_index = []
	rIndices = indices[MergeRanks == r]  # 当前等级为r的索引
	while i + len(rIndices) <= nPop and r <= max_r:
		select_index.extend(rIndices)
		r += 1  # 当前等级+1
		i += len(rIndices)
		rIndices = indices[MergeRanks == r]  # 当前等级为r的索引

	if i < nPop and r <= max_r:
		rDistances = MergeDistances[rIndices]  # 当前等级个体的拥挤度
		rSortedIdx = np.argsort(rDistances)[::-1]  # 按照距离排序 由大到小
		surIndices = rIndices[rSortedIdx[:(nPop - i)]]
		select_index.extend(surIndices)
	return MergePops[select_index], MergeFits[select_index]
