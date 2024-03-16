import numpy as np
import label_propagation as lb
rng = np.random.default_rng(seed=2002)


def accuracy(validation_labels, computed_labels):
    accuracy = 0
    n = len(computed_labels)
    for i in range(n):
        if computed_labels[i] == validation_labels[i]:
            accuracy += 1
    return accuracy/n


def compute_Q(labels):
    n = len(labels)
    Q = np.empty(10)
    for i in range(10):
        Q[i] = len(labels[(labels == i)])/n
    return Q

def read_mnist__10(data_path):
    train_data = np.loadtxt(data_path + "mnist_train.csv", delimiter=",")

    # Separate images and labels
    labels = train_data[:, 0]
    images = train_data[:, 1:]

    indices = rng.choice(len(labels), size=4000, replace=False)
    return images[indices], labels[indices]

def read_mnist_scaled__10(data_path):
    reduced_images, reduced_labels = read_mnist__10(data_path)
    return reduced_images/255, reduced_labels

if __name__ == "__main__":
	# Execute labeling algorithm
	reduced_images, reduced_labels = read_mnist_scaled__10("../data/")
	n, m = reduced_images.shape
	sigma = 2.
	W = lb.W_common_sigma(reduced_images, sigma)

	acc_cmn = np.empty(20)
	acc_thresh = np.empty(20)
	acc_nn = np.empty(20)
	acc_rbf = np.empty(20)

	for l in range(10, 210, 10):
		learning_labels = reduced_labels[0:l]
		Q = compute_Q(learning_labels)
		validation_labels = reduced_labels[l:]
		collection_of_f = np.empty((n-l, 10))
		rbf_labels = np.empty((l, 10))
		for i in range(10):
		    tmp_labels = np.empty(len(learning_labels))
		    for k in range(len(learning_labels)):
		        if learning_labels[k] == i:
		            tmp_labels[k] = 1
		        else:
		            tmp_labels[k] = 0
		    rbf_labels[:,i] = tmp_labels
		    collection_of_f[:, i] = lb.fu(tmp_labels, W, l, n-l)

		computed_labels_rbf = lb.RBF(rbf_labels, W)
		computed_labels_thresh = lb.thresh(collection_of_f)
		computed_labels_cmn = lb.CMN(collection_of_f, Q)
		computed_labels_nn = lb.NN(learning_labels, reduced_images)

		acc_cmn[(l-2)//10] = accuracy(validation_labels, computed_labels_cmn)
		acc_thresh[(l-2)//10] = accuracy(validation_labels, computed_labels_thresh)
		acc_nn[(l-2)//10] = accuracy(validation_labels, computed_labels_nn)
		acc_rbf[(l-2)//10] = accuracy(validation_labels, computed_labels_rbf)

	print("CMN: ", acc_cmn)
	print("Thresh: ", acc_thresh)
	print("1NN: ", acc_nn)
	print("RBF: ", acc_rbf)
