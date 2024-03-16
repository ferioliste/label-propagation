import numpy as np
import label_propagation as lb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

def generate_k_elices(n, k, noise):
	s = n//k
	n = s*k
	
	points = np.empty((n, 3))
	labels = np.empty(n, dtype = 'i')
	
	t = np.linspace(0, 1, num=s) * 400 * np.pi/180
	for i in range(k):
		points[(i*s):(i*s + s), 0] = np.cos(t + i*2*np.pi/k) + np.random.normal(size = s) * noise
		points[(i*s):(i*s + s), 1] = np.sin(t + i*2*np.pi/k) + np.random.normal(size = s) * noise
		points[(i*s):(i*s + s), 2] = t + np.random.normal(size = s)/10 * noise
		labels[(i*s):(i*s + s)] = i*np.ones(s)
	return points, labels

k = 4
n = (186//k)*k
l = 18
u = n-l
X, labels = generate_k_elices(n, k, 0.05)

labelled_id = random.sample(range(n), l)
labels_set = set(labels)
while set(labels[labelled_id]) != labels_set:
	labelled_id = random.sample(range(n), l)
unlabelled_id = list(set(range(n)) - set(labelled_id))

X_l = X[labelled_id]
X_u = X[unlabelled_id]
X_lu = np.vstack((X_l,X_u))
labels_l = labels[labelled_id]

fl = lb.labels_to_binary(labels_l)
print(fl)

W = lb.W_common_sigma(X_lu, 0.43)
fu = np.empty((u,k))

for i in range(k):
	fu[:,i] = lb.fu(fl[:,i], W, l, u)

labels_u = lb.CMN(fu, (1./k)*np.ones(k))

color_map = {-1: 'black', 0: 'red', 1: 'blue', 2: 'black', 3: 'green', 4: 'orange', 5: 'purple', 6: 'brown', 7: 'pink'}
marker_map = {-1: '', 0: '+', 1: 'o', 2: '$*$', 3: 'x', 4: 'orange', 5: 'purple', 6: 'brown', 7: 'pink'}
colors_u = [color_map[lb] for lb in labels_u]
colors_l = [color_map[lb] for lb in labels_l]
marker_u = [marker_map[lb] for lb in labels_u]
marker_l = [marker_map[lb] for lb in labels_l]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
for i in range(k):
	if marker_map[i] == 'o':
		ax.scatter(X_u[labels_u == i,0], X_u[labels_u == i,1], X_u[labels_u == i,2], edgecolors = color_map[i], cmap='bwr', depthshade=False, marker = marker_map[i], linewidth = 0.5, s = 30, facecolors='none')
		ax.scatter(X_l[labels_l == i,0], X_l[labels_l == i,1], X_l[labels_l == i,2], edgecolors = color_map[i], cmap='bwr', depthshade=False, marker = marker_map[i], linewidth = 2, s = 160, facecolors='none')
	else:
		ax.scatter(X_u[labels_u == i,0], X_u[labels_u == i,1], X_u[labels_u == i,2], c = color_map[i], cmap='bwr', depthshade=False, marker = marker_map[i], linewidth = 0.5, s = 30, facecolors='none')
		ax.scatter(X_l[labels_l == i,0], X_l[labels_l == i,1], X_l[labels_l == i,2], c = color_map[i], cmap='bwr', depthshade=False, marker = marker_map[i], linewidth = 2, s = 160, facecolors='none')
plt.savefig("../plots/elices" + str(k) + ".pdf", format='pdf')
plt.show()
