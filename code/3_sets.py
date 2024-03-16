import numpy as np
import label_propagation as lb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

def generate_3groups(n, noise=0.25):
	dx = np.random.random(((n//3)*3,1))*3.5
	d1y = np.random.random((n//3,1))*0.5 + 0
	d2y = np.random.random((n//3,1))*0.5 + 1.25
	d3y = np.random.random((n//3,1))*0.5 + 2.5

	points = np.hstack((dx, np.vstack((d1y, d2y, d3y))))
	labels = np.hstack((0*np.ones(n//3,dtype = 'i'), 1*np.ones(n//3,dtype = 'i'), 2*np.ones(n//3,dtype = 'i')))

	return points, labels

k = 3
n = 180
l = 3
u = n-l
X, labels = generate_3groups(n)

labelled_id = random.sample(range(n), l)
labels_set = set(labels)
while set(labels[labelled_id]) != labels_set:
	labelled_id = random.sample(range(n), l)
unlabelled_id = list(set(range(n)) - set(labelled_id))

X_l = X[labelled_id]
X_u = X[unlabelled_id]
X_lu = np.vstack((X_l,X_u))
labels_l = labels[labelled_id]
print(labels_l)
fl = lb.labels_to_binary(labels_l)

W = lb.W_common_sigma(X_lu, 0.22)
fu = np.empty((u,k))

for i in range(k):
	fu[:,i] = lb.fu(fl[:,i], W, l, u)

labels_u = lb.CMN(fu, (1./k)*np.ones(k))

color_map = {-1: 'black', 0: 'red', 1: 'blue', 2: 'black', 3: 'green', 4: 'orange', 5: 'purple', 6: 'brown', 7: 'pink'}
marker_map = {-1: '', 0: '+', 1: 'o', 2: '$*$', 3: 'x', 4: 'orange', 5: 'purple', 6: 'brown', 7: 'pink'}

for i in range(k):
	if marker_map[i] == 'o':
		plt.scatter(X_u[labels_u == i,0], X_u[labels_u == i,1], edgecolors = color_map[i], marker = marker_map[i], linewidth = 0.5, s = 30, facecolors='none')
		plt.scatter(X_l[labels_l == i,0], X_l[labels_l == i,1], edgecolors = color_map[i], marker = marker_map[i], linewidth = 2, s = 160, facecolors='none')
	else:
		plt.scatter(X_u[labels_u == i,0], X_u[labels_u == i,1], c = color_map[i], marker = marker_map[i], linewidth = 0.5, s = 30, facecolors='none')
		plt.scatter(X_l[labels_l == i,0], X_l[labels_l == i,1], c = color_map[i], marker = marker_map[i], linewidth = 2, s = 160, facecolors='none')
plt.savefig("./plots/3sets.pdf", format='pdf')
plt.show()
