import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

def Acc(FS, C):
	C = C.astype('int')
	kf = KFold(n_splits = 5)
	mAcc = []
	for train_index, test_index in kf.split(FS):
		X_train, X_test = FS[train_index], FS[test_index]
		y_train, y_test = C[train_index], C[test_index]
		pred = KNeighborsClassifier(n_neighbors = 1).fit(X_train, y_train).predict(X_test)
		mAcc.append((np.sum(pred[y_test == 0] == 0) / len(pred) + np.sum(pred[y_test == 1] == 1) / len(pred)) / 2)
	return np.mean(mAcc)

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