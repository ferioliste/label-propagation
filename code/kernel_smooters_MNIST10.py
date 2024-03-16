import numpy as np
import label_propagation as lb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from mnist_10 import *
from scipy.linalg import norm
from kernel_smoothers import *

if __name__ == "__main__":
	reduced_images, reduced_labels = read_mnist_scaled__10("../data/")
	n, m = reduced_images.shape
	sigma = 2.

	acc_cmn = np.empty(20)
	acc_thresh = np.empty(20)

	ll = list(range(10, 210, 10))
	for l in ll:
		learning_labels = reduced_labels[0:l]
		Q = compute_Q(learning_labels)
		print(Q)
		validation_labels = reduced_labels[l:]
		collection_of_f = np.empty((n-l, 10))
		for i in range(10):
		    tmp_labels = np.empty(len(learning_labels))
		    for k in range(len(learning_labels)):
		        if learning_labels[k] == i:
		            tmp_labels[k] = 1
		        else:
		            tmp_labels[k] = 0
		    collection_of_f[:, i] = fu_KS(reduced_images, learning_labels, h = sigma**2)

		computed_labels_thresh = lb.thresh(collection_of_f)
		computed_labels_cmn = lb.CMN(collection_of_f, Q)

		acc_cmn[(l-2)//10] = accuracy(validation_labels, computed_labels_cmn)
		acc_thresh[(l-2)//10] = accuracy(validation_labels, computed_labels_thresh)

	print("CMN: ", acc_cmn)
	print("Thresh: ", acc_thresh)
	
	plt.plot(ll, acc_cmn, '-o', label="KN: CMN", c = "red")
	plt.plot(ll, acc_thresh, '-o', label="KN: Thresh", c = "green")
	
	y_1 = [0.31754386, 0.61884422, 0.69395466, 0.79318182, 0.79746835, 0.78527919,
       0.79491094, 0.80433673, 0.81099744, 0.82282051, 0.83753213, 0.8371134,
       0.83049096, 0.83419689, 0.85220779, 0.85130208, 0.86788512, 0.86151832,
       0.86404199, 0.86315789]

	y_2 = [0.09849624, 0.11557789, 0.12040302, 0.22878788, 0.35620253, 0.40177665,
		   0.43486005, 0.51913265, 0.54603581, 0.55025641, 0.56786632, 0.57757732,
		   0.58604651, 0.61243523, 0.62,       0.63151042, 0.66684073, 0.67408377,
		   0.6847769,  0.69368421]

	plt.plot(x, y_1, '--o', label="LP: CMN")
	plt.plot(x, y_2, '--o', label="LP: Thresh")
	plt.xlabel("labeled set size")
	plt.ylabel("accuracy")
	plt.legend()
	plt.title("10 digits classification\nlabel propagation vs kernel smoothers")
	plt.savefig("../plots/KSvsProp_mnist_10.pdf", format='pdf')
