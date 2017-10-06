import numpy as np


def horizontal_flips_static(X, y):

    print("Horizontal flips of data set...")

    flipped = X[:, :, :, ::-1]
    X = np.concatenate((X, flipped))
    y = np.concatenate((y, y))

    # random permutation of indices
    perm = np.random.permutation(len(y))

    # shuffle data with permutation
    X = X[perm]
    y = y[perm]

    print("Done.")
