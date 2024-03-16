
import numpy as np
import mnist__1_2 as mnist


def accuracy(perceptron_labels, valid_labels):
    acc = 0
    for perceptron_label, valid_label in zip(perceptron_labels, valid_labels):
        if perceptron_label == valid_label:
            acc += 1
    return acc / perceptron_labels.shape[0]


def perceptron_train(data, labels, num_iter):
    # set weights to zero
    w = np.zeros(shape=(1, data.shape[1]))
    misclassified_ = []
    for epoch in range(num_iter):
        misclassified = 0
        for x, label in zip(data, labels):
            y = np.dot(w, x)
            target = 1.0 if (y > 0) else 0.0
            delta = (label - target)
            if (delta):  # misclassified
                misclassified += 1
                w += (delta * x)
        misclassified_.append(misclassified)
    return w


def perceptron_labels(data, w):
    labels = np.empty(data.shape[0])
    for i in range(data.shape[0]):
        y = np.dot(w, data[i])
        labels[i] = 1.0 if (y > 0) else 0.0
    return labels


if __name__ == "__main__":
    data, labels = mnist.read_mnist_scaled__1_2("../data/")
    labels -= 1
    w = perceptron_train(data[:20], labels[:20], 10)
    perc_labels = perceptron_labels(data, w)
    acc = accuracy(perc_labels, labels)
    print(acc)
