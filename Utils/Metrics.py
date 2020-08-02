import os

from sklearn.metrics import roc_curve, auc, average_precision_score
from matplotlib import pyplot as plt


def ROC(true, pred, show=False, wpath=None, name=None):
	FPR, TPR, thresh = roc_curve(true, pred)
	AUC = auc(FPR, TPR)
	
	if show or wpath is not None:
		plt.title("ROC")
		plt.plot(FPR, TPR, "b", label="AUC = %0.2f" % AUC)
		plt.legend(loc="lower right")
		plt.plot([0, 1], [0, 1], "r--")
		plt.xlim([0, 1])
		plt.ylim([0, 1])
		plt.ylabel("True Positive Rate")
		plt.xlabel("False Positive Rate")
		
		if show:
			plt.show()
			
		if wpath is not None:
			os.makedirs(wpath, exist_ok=True)
			
			name = name if name is not None else "ROC_curve"
			name = os.path.splitext(name)[0] + ".png"
			
			plt.savefig(os.path.join(wpath, name))

		plt.close()
			
	return AUC


def average_precision(true, pred):
	return average_precision_score(true, pred)