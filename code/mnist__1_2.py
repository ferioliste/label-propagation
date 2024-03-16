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
    Q = np.empty(2)
    for i in range(2):
        Q[i] = len(labels[(labels == i)])/n
    return Q


def read_mnist__1_2(data_path):
    train_data = np.loadtxt(data_path + "mnist_train.csv", delimiter=",")

    # Separate images and labels
    labels = train_data[:, 0]
    images = train_data[:, 1:]

    # Get a random set including only '1' and '2' with size 1100 for each category
    images_one = images[(labels == 1)]
    labels_one = labels[(labels == 1)]
    indices = rng.choice(len(images_one), size=1100, replace=False)
    images_one = images_one[indices]
    labels_one = labels_one[indices]

    images_two = images[(labels == 2)]
    labels_two = labels[(labels == 2)]
    indices = rng.choice(len(images_two), size=1100, replace=False)
    images_two = images_two[indices]
    labels_two = labels_two[indices]

    # Merge both sets in an alternating fashion (1,2,1,2,...)
    images_one_two = np.ravel([images_one, images_two], order="F").reshape(784, 2200).T
    labels_one_two = np.dstack((labels_one, labels_two)).flatten()
    return images_one_two, labels_one_two


def read_mnist_scaled__1_2(data_path):
    images_one_two, labels_one_two = read_mnist__1_2(data_path)
    return images_one_two/255, labels_one_two


if __name__ == "__main__":
    # Execute labeling algorithm
    images_one_two, labels_one_two = read_mnist_scaled__1_2("../data/")
    n, m = images_one_two.shape

    # sigma found by binary search (could be improved in the future)
    sigma = 1.8
    W = lb.W_common_sigma(images_one_two, sigma)

    acc_cmn = np.empty(10)
    acc_thresh_adaptive = np.empty(10)
    acc_thresh = np.empty(10)
    acc_nn = np.empty(10)
    acc_rbf = np.empty(10)
    for l in range(2, 102, 10):
        learning_labels = labels_one_two[0:l]-1
        validation_labels = labels_one_two[l:]-1
        f = lb.fu(learning_labels, W, l, n-l)

        collection_of_f = np.stack((1-f, f), axis=-1, dtype=float)

        Q = compute_Q(learning_labels)
        computed_labels_thresh = lb.thresh(collection_of_f)
        computed_labels_thresh_adaptive = lb.thresh_adaptive_binary(collection_of_f)
        computed_labels_cmn = lb.CMN(collection_of_f, Q)
        computed_labels_nn = lb.NN(learning_labels, images_one_two)
        computed_labels_rbf = lb.RBF(np.stack((1-learning_labels, learning_labels), axis=-1, dtype=float), W)

        acc_cmn[(l-2)//10] = accuracy(validation_labels, computed_labels_cmn)
        acc_thresh_adaptive[(l-2)//10] = accuracy(validation_labels, computed_labels_thresh_adaptive)
        acc_thresh[(l-2)//10] = accuracy(validation_labels, computed_labels_thresh)
        acc_nn[(l-2)//10] = accuracy(validation_labels, computed_labels_nn)
        acc_rbf[(l-2)//10] = accuracy(validation_labels, computed_labels_rbf)

    print("CMN: ", acc_cmn)
    print("Adaptive Thresh: ", acc_thresh_adaptive)
    print("Thresh: ", acc_thresh)
    print("1NN: ", acc_nn)
    print("RBF: ", acc_rbf)
