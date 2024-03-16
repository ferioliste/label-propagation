import numpy as np
import label_propagation as lb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from mnist__1_2 import *
from scipy.linalg import norm

def find_minima(arr):
	minima = []
	for i in range(1,arr.size-1):
		if arr[i-1] > arr[i] and arr[i+1] > arr[i]:
			minima.append(arr[i])
	return np.array(minima)

def test_set():
	points = np.empty((44, 2), dtype='d')
	
	labels = np.array([0] + [1] + [0]*21 + [1]*21)
	
	points[0] = np.array([-3*0.95,2])
	points[1] = np.array([3,0])
	
	points[2:9,0] = 0.95*np.arange(-3,4)
	points[2:9,1] = np.array([2+2*0.95]*7)
	
	points[9:16,0] = 0.95*np.arange(-3,4)
	points[9:16,1] = np.array([2+0.95]*7)
	
	points[16:22,0] = 0.95*np.arange(-2,4)
	points[16:22,1] = np.array([2]*6)
	
	points[22] = np.array([0, 1+1./3])
	points[23] = np.array([0, 1-1./3])
	
	points[24:30,0] = np.arange(-3,3)
	points[24:30,1] = np.array([0]*6)
	
	points[30:37,0] = np.arange(-3,4)
	points[30:37,1] = np.array([-1]*7)
	
	points[37:44,0] = np.arange(-3,4)
	points[37:44,1] = np.array([-2]*7)
	
	return points, labels

def fu_KS(X, fl, K = lambda t : np.exp(-t**2), h = 1):
	n = X.shape[0]
	l = fl.size
	u = n-l
	
	s = np.empty((u,l))
	fu = np.empty(u)
	for j in range(u):
		for i in range(l):
			s[j,i] = K(norm(X[l+j]-X[i])/h)
		fu[j]= np.dot(s[j],fl)/np.sum(s[j])
	return fu

if __name__ == "__main__":
	images_one_two, labels_one_two = read_mnist_scaled__1_2("../data/")
	n, m = images_one_two.shape
	sigma = 1.8
	
	ll = list(range(2, 102, 10))
	
	acc_cmn = np.empty(10)
	acc_thresh_adaptive = np.empty(10)
	acc_thresh = np.empty(10)
	acc_nn = np.empty(10)
	for l in ll:
		learning_labels = labels_one_two[0:l]-1
		validation_labels = labels_one_two[l:]-1
		f = fu_KS(images_one_two, learning_labels, h = sigma**2)

		collection_of_f = np.stack((1-f, f), axis=-1, dtype=float)

		Q = compute_Q(learning_labels)
		computed_labels_thresh = lb.thresh(collection_of_f)
		computed_labels_thresh_adaptive = lb.thresh_adaptive_binary(collection_of_f)
		computed_labels_cmn = lb.CMN(collection_of_f, Q)

		acc_cmn[(l-2)//10] = accuracy(validation_labels, computed_labels_cmn)
		acc_thresh_adaptive[(l-2)//10] = accuracy(validation_labels, computed_labels_thresh_adaptive)
		acc_thresh[(l-2)//10] = accuracy(validation_labels, computed_labels_thresh)
	
	print("CMN: ", acc_cmn)
	print("Adaptive Thresh: ", acc_thresh_adaptive)
	print("Thresh: ", acc_thresh)

	plt.plot(ll, acc_cmn, '-o', label="KN: CMN", c = "red")
	plt.plot(ll, acc_thresh_adaptive, '-o', label="KN: Adaptive Thresh", c = "blue")
	plt.plot(ll, acc_thresh, '-o', label="KN: Thresh", c = "green")
	
	y_1 = [0.80618744, 0.83638026, 0.95316804, 0.94926199, 0.94995366, 0.95297952, 0.95275959, 0.95441729, 0.95845137, 0.96252372]
	y_2 = [0.98908098, 0.98903108, 0.98898072, 0.98892989, 0.98887859, 0.98882682, 0.98877456, 0.9887218,  0.98866856, 0.9886148]
	y_3 = [0.5, 0.50411335, 0.51239669, 0.52905904, 0.67933272, 0.79981378, 0.8171188, 0.82048872, 0.83050047, 0.91176471]
	y_4 = [0.75705187, 0.85466179, 0.9214876, 0.94464945, 0.94717331, 0.94646182, 0.9494855, 0.94924812, 0.95089707, 0.96204934]
	y_5 = [0.75705187, 0.84552102, 0.91368228, 0.93773063, 0.94068582, 0.94227188, 0.94153414, 0.93843985, 0.93909348, 0.95066414]

	plt.plot(ll, y_1, '--o', label="LP: CMN", c = "red")
	plt.plot(ll, y_2, '--o', label="LP: Adaptive Thresh", c = "blue")
	plt.plot(ll, y_3, '--o', label="LP: Thresh", c = "green")
	
	plt.xlabel("labeled set size")
	plt.ylabel("accuracy")
	plt.legend()
	plt.title("1-2 digits classification\nlabel propagation vs kernel smoothers")
	plt.savefig("./plots/KSvsProp_mnist_binary-1-2.pdf", format='pdf')
	#plt.show()
	
if False:
	points, labels = test_set()
	
	l = 2
	fl = labels[0:l]
	
	fu = fu_KS(points, fl)
	
	guessed_labels = np.floor(fu + 0.5)
	print(points[l:].shape)
	color_map = {-1: 'black', 0: 'red', 1: 'blue', 2: 'black', 3: 'green', 4: 'orange', 5: 'purple', 6: 'brown', 7: 'pink'}
	marker_map = {-1: '', 0: '+', 1: 'o', 2: '$*$', 3: 'x', 4: 'orange', 5: 'purple', 6: 'brown', 7: 'pink'}

	for i in range(2):
		if marker_map[i] == 'o':
			plt.scatter(points[l:][guessed_labels == i,0], points[l:][guessed_labels == i,1], edgecolors = color_map[i], marker = marker_map[i], linewidth = 0.5, s = 30, facecolors='none')
			plt.scatter(points[:l][labels[:l] == i,0], points[:l][labels[:l] == i,1], edgecolors = color_map[i], marker = marker_map[i], linewidth = 2, s = 160, facecolors='none')
		else:
			plt.scatter(points[l:][guessed_labels == i,0], points[l:][guessed_labels == i,1], c = color_map[i], marker = marker_map[i], linewidth = 0.5, s = 30, facecolors='none')
			plt.scatter(points[:l][labels[:l] == i,0], points[:l][labels[:l] == i,1], c = color_map[i], marker = marker_map[i], linewidth = 2, s = 160, facecolors='none')
	plt.savefig("../plots/KS_test_set.pdf", format='pdf')
	plt.show()
