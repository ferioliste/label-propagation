import numpy as np
import label_propagation as lb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

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

if __name__ == "__main__":
	points, labels = test_set()

	# Plot the entropy function H
	l = 2
	learning_labels = labels[0:l]
	eps = [0, 0.1, 0.01, 0.001, 0.0001]

	length = 150
	X = np.linspace(0.2, 2.1, length)
	Y = np.empty((len(eps), length))
	for i in range(len(eps)):
		for j in range(length):
			Y[i, j] = lb.entropy(points, learning_labels, X[j], eps[i])
		if eps[i] == 0:
			plt.plot(X, Y[i, :], label="unsmoothed")
		else:
			plt.plot(X, Y[i, :], label="$\epsilon = 10^{-%d}$" % i)
		minima = find_minima(Y[i])
		print(np.max(minima))
	plt.xlabel("$\sigma$")
	plt.ylabel("entropy")
	plt.legend()
	plt.savefig("./plots/test_set_entropies.pdf", format='pdf')
	plt.show()

	color_map = {-1: 'black', 0: 'red', 1: 'blue', 2: 'black', 3: 'green', 4: 'orange', 5: 'purple', 6: 'brown', 7: 'pink'}
	marker_map = {-1: '', 0: '+', 1: 'o', 2: '$*$', 3: 'x', 4: 'orange', 5: 'purple', 6: 'brown', 7: 'pink'}

	for i in range(2):
		if marker_map[i] == 'o':
			plt.scatter(points[l:][labels[l:] == i,0], points[l:][labels[l:] == i,1], edgecolors = color_map[i], marker = marker_map[i], linewidth = 0.5, s = 30, facecolors='none')
			plt.scatter(points[:l][labels[:l] == i,0], points[:l][labels[:l] == i,1], edgecolors = color_map[i], marker = marker_map[i], linewidth = 2, s = 160, facecolors='none')
		else:
			plt.scatter(points[l:][labels[l:] == i,0], points[l:][labels[l:] == i,1], c = color_map[i], marker = marker_map[i], linewidth = 0.5, s = 30, facecolors='none')
			plt.scatter(points[:l][labels[:l] == i,0], points[:l][labels[:l] == i,1], c = color_map[i], marker = marker_map[i], linewidth = 2, s = 160, facecolors='none')
	plt.savefig("./plots/test_set.pdf", format='pdf')
	plt.show()
