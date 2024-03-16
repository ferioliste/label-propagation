import numpy as np
import label_propagation as lb
import perceptron
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


def read_mnist(data_path):
    train_data = np.loadtxt(data_path + "mnist_train.csv", delimiter=",")

    # Separate images and labels
    labels = train_data[:, 0]
    images = train_data[:, 1:]

    images_odd = []
    labels_odd = []
    images_even = []
    labels_even = []

    for i in range(10):
        images_filtered = images[(labels == i)]
        indices = rng.choice(len(images_filtered), size=400, replace=False)
        if i % 2 == 0:
            images_even.append(images_filtered[indices])
            labels_even.append(np.zeros(400))
        else:
            images_odd.append(images_filtered[indices])
            labels_odd.append(np.ones(400))

    # Merge both sets in an alternating fashion
    images_odd_even = np.ravel([images_odd, images_even], order="F").reshape(784, 4000).T
    labels_odd_even = np.dstack((labels_odd, labels_even)).flatten()

    return images_odd_even, labels_odd_even


def read_mnist_scaled(data_path):
    images, labels = read_mnist(data_path)
    return images/255, labels


if __name__ == "__main__":
    images, labels = read_mnist_scaled("../data/")
    sigma = 2.5

    W = lb.W_common_sigma(images, sigma)

    acc_cmn = np.empty(10)
    acc_thresh = np.empty(10)
    acc_perceptron = np.empty(10)
    acc_cmn_external = np.empty(10)
    acc_thresh_external = np.empty(10)
    for l in range(2, 102, 10):
        learning_labels = labels[:l]
        validation_labels = labels[l:]
        f = lb.fu(learning_labels, W, l, images.shape[0]-l)

        collection_of_f = np.stack((1-f, f), axis=-1, dtype=float)

        # Thresh and CMN
        Q = compute_Q(learning_labels)
        acc_thresh[(l-2)//10] = accuracy(validation_labels, lb.thresh(collection_of_f))
        acc_cmn[(l-2)//10] = accuracy(validation_labels, lb.CMN(collection_of_f, Q))

        # Perceptron
        w = perceptron.perceptron_train(images[:l], learning_labels, 10)
        perc_labels = perceptron.perceptron_labels(images, w)
        acc_perceptron[(l-2)//10] = accuracy(validation_labels, perc_labels[l:])

        # Incorporate Perceptron as External Classifier
        f_external = lb.fu_external_classifier(learning_labels, W, 0.1, perc_labels)
        collection_of_f_external = np.stack((1-f_external, f_external), axis=-1, dtype=float)

        acc_thresh_external[(l-2)//10] = accuracy(validation_labels, lb.thresh(collection_of_f_external))
        acc_cmn_external[(l-2)//10] = accuracy(validation_labels, lb.CMN(collection_of_f_external, Q))

    print("CMN: ", acc_cmn)
    print("Thresh: ", acc_thresh)
    print("Perceptron: ", acc_perceptron)
    print("CMN (external): ", acc_cmn_external)
    print("Thresh (external): ", acc_thresh_external)
