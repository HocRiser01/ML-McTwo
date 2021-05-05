import numpy as np
from minepy import MINE

def mic(x, y):
	mine = MINE()
	mine.compute_score(x, y)
	return mine.mic()

def McOne(F, C, r):
	s, k = F.shape
	micFC = [-1] * k
	Subset = [-1] * k
	numSubset = 0
	for i in range(k):
		micFC[i] = mic(F[:, i], C)
		if micFC[i] >= r:
			Subset[numSubset] = i
			numSubset += 1
	Subset = Subset[0:numSubset]
	Subset.sort(key = lambda x: micFC[x], reverse = True)
	mask = [True] * numSubset
	for e in range(numSubset):
		if mask[e]:
			for q in range(e + 1, numSubset):
				if mask[q] and mic(F[:, Subset[e]], F[:, Subset[q]]) >= micFC[Subset[q]]:
					mask[q] = False
	return F[:, np.array(Subset)[mask]]