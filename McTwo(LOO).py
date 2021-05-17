import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def Acc(FS, C):
	s = FS.shape[0]
	C = C.astype('int')
	NN = KNeighborsClassifier(n_neighbors = 1)
	pred = []
	for i in range(s):
		NN.fit(FS[[x for x in range(s) if x != i]],C[[x for x in range(s) if x != i]])
		pred.append(NN.predict(FS[[i]]).tolist()[0])
	pred = np.array(pred)
	return (np.mean(pred[np.where(C == 0)] == C[np.where(C == 0)]) +
			np.mean(pred[np.where(C == 1)] == C[np.where(C == 1)])) / 2

def McTwo(FR, C):
	s, k = FR.shape
	curAcc = -1
	curSet = []
	leftSet = [x for x in range(k)]
	while True:
		tempAcc, idx = -1, -1
		for x in leftSet:
			tmpAcc = Acc(FR[:, curSet + [x]], C)
			if tmpAcc > tempAcc:
				tempAcc = tmpAcc
				idx = x
		if tempAcc > curAcc:
			curAcc = tempAcc
			curSet = curSet + [idx]
			leftSet.remove(idx)
		else:
			break
	return FR[:, curSet]
