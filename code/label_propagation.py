import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


def fu(fl, W, l, u):
    """Compute the function f for the unlabeled data u"""
    Duu = np.diag(np.ravel(W[l:, :].sum(axis=1)))
    return np.linalg.solve(Duu - W[l:, l:], W[l:, 0:l]@fl)


def fu_external_classifier(fl, W, eta, h):
    P = P_fast(W)
    l = len(fl)
    n = len(h)
    return np.linalg.solve(np.identity(n-l) - (1 - eta) * P[l:, l:], (1 - eta) * P[l:, 0:l]@fl + eta * h[l:])


def guess_sigma(fl, W_0, l, u):
    sigmas = np.arange(0.25, 2, 0.05)
    hentropies = np.empty(sigmas.size)
    for i in range(sigmas.size):
        hentropies[i] = hentropy(fl, np.power(W_0, 1./sigmas[i]), l, u)
        print(i, np.power(W_0, 1./sigmas[i]))
    plt.plot(sigmas, hentropies)
    plt.show()


def hentropy(fl, W, l, u):
    fu_ = fu(fl, W, l, u)
    return (1/u)*np.sum(H(fu_))


def H(f):
    return -f*np.log(f) - (1-f)*np.log(1-f)


def entropy(X, fl, sigma, eps):
    n = X.shape[0]
    l = len(fl)
    W = W_common_sigma(X, sigma)
    P = eps * np.full((n, n), 1/n) + (1-eps) * P_fast(W)
    f = np.linalg.solve(np.identity(n-l) - P[l:, l:], P[l:, 0:l]@fl)
    return (1/len(f))*np.sum(-f*np.log(f) - (1-f)*np.log(1-f))


def labels_to_binary(labels):
    k = len(set(labels))
    l = len(labels)
    binary_labels = np.empty((l, k))
    for i in range(l):
        binary_labels[i] = np.zeros(k)
        binary_labels[i, labels[i]] = 1
    return binary_labels


def CMN(fu, Q):
    """Compute labels given f via class mass normalization (CMN)"""
    n = fu.shape[0]
    labels = np.empty(n)
    fu = normalize(fu, norm="l1", axis=0)
    for i in range(n):
        labels[i] = np.argmax(fu[i, :] * Q)
    return labels


def thresh_adaptive_binary(fu):
    """Compute labels given f via adaptive harmonic threshold (works only for
    0-1 classification)"""
    u = fu.shape[0]
    labels = np.empty(u)
    q = np.median(fu[:, 1])
    for i in range(u):
        if fu[i, 1] > q:
            labels[i] = 1
        else:
            labels[i] = 0
    return labels


def thresh(fu):
    """Compute labels given f via harmonic threshold (thresh)"""
    u = fu.shape[0]
    labels = np.empty(u)
    for i in range(u):
        labels[i] = np.argmax(fu[i, :])
    return labels


def NN(fl, X):
    """Compute labels given f via 1-Nearest-Neighbour (NN)"""
    l = len(fl)
    u = X.shape[0]-l
    distances = np.empty((l, u))
    for i in range(l):
        for j in range(u):
            distances[i, j] = np.linalg.norm(X[i] - X[l+j], 2)
    labels = np.empty(u)
    for i in range(u):
        labels[i] = fl[np.argmin(distances[:, i])]
    return labels


def RBF(fl, W):
    """Compute labels given f via RBF"""
    l = len(fl)
    u = W.shape[1]-len(fl)
    v = W[l:, :l]@fl
    labels = np.empty(u)
    for i in range(u):
        labels[i] = np.argmax(v[i, :])
    return labels


def W(X, sigma):
    """Compute the weight matrix W for the given data (X is matrix with vector
    x_i as i-th row)"""
    n, m = X.shape
    return np.fromfunction(lambda i, j: np.exp(- sum(((X[i, d] - X[j, d])**2)/sigma[d] for d in range(m))), (n, n), dtype=int)


def W_common_sigma(X, sigma):
    """Compute the matrix W for a common sigma using broadcasting"""
    X_norm = np.sum(X ** 2, axis=-1)
    return np.exp(-(1/sigma**2) * (X_norm[:, None] + X_norm[None, :] - 2 * np.dot(X, X.T)))


def P_fast(W):
    """Compute the matrix P without solving a LGS or using an inverse matrix"""
    W_row_sums = W.sum(axis=1)
    return W / W_row_sums[:, np.newaxis]
