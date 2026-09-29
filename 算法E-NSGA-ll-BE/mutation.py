import random
import numpy as np

np.random.seed(1)
random.seed(1)

def mutate(pops, pm):
    nPop = pops.shape[0]
    nChr = pops.shape[1]
    for i in range(nPop):
        if np.random.rand() < pm:
            for j in range(nChr):
                if np.random.rand() < pm:
                    if pops[i][j] == 1:
                        pops[i][j] = 0
                    else:
                        pops[i][j] = 1
    return pops
