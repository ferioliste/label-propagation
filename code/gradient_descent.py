import numpy as np
import label_propagation as lb
import mnist__1_2 as mnist
rng = np.random.default_rng(seed=2002)


def P_gradient(X, sigma, eps):
    """Compute the matrix P without solving a LGS or using an inverse matrix"""
    n = X.shape[0]
    X_norm = np.sum(X ** 2, axis=-1)
    X_diff = (X_norm[:, None] + X_norm[None, :] - 2 * np.dot(X, X.T))
    W = np.exp(-(1/sigma**2) * X_diff)

    W_row_sums = W.sum(axis=1)
    P = W / W_row_sums[:, np.newaxis]
    P = eps * np.full((n, n), 1/n) + (1-eps) * P

    W_d_sigma = 2 * W * (1/sigma**3) * X_diff
    W_d_sigma_rows_sum = W_d_sigma.sum(axis=1)
    P_d_sigma = (W_d_sigma - (P * W_d_sigma_rows_sum[:, np.newaxis])) / W_row_sums[:, np.newaxis]
    return P, P_d_sigma


def entropy_gradient(X, sigma, eps):
    n = X.shape[0]
    P, P_d_sigma = P_gradient(X, sigma, eps)
    f = np.linalg.solve(np.identity(n-l) - P[l:, l:], P[l:, 0:l]@learning_labels)
    f_d_sigma = np.linalg.solve(np.identity(n-l) - P[l:, l:], P_d_sigma[l:, l:]@f + P_d_sigma[l:, 0:l]@learning_labels)
    return (1/len(f))*np.sum(np.log((1 - f)/f) * f_d_sigma)


def gradient_descent(sigma, eps, learning_rate=0.01, iter=1000):
    for i in range(iter):
        sigma_grad = entropy_gradient(images_one_two, sigma, eps)
        sigma -= sigma_grad * learning_rate
        if i % 10 == 0:
            print("Iteration: ", i, "Sigma: ", sigma)
    return sigma


if __name__ == "__main__":
    # Read in data and execute gradient descent
    images_one_two, labels_one_two = mnist.read_mnist_scaled__1_2("../data/")

    l = 100
    learning_labels = labels_one_two[0:l]-1
    sigma = 2.
    eps = 0.0001
    gradient_descent(sigma, eps, 0.01, 1000)
