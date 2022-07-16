import numpy as np


def discretegrid(xy, w, nt):
    """
    Convert spatial observations to a discrete intensity grid
    :param xy: observed spatial locations as a two-column vector
    :param w: observation window, i.e. discrete grid to be mapped to, [xmin xmax ymin ymax]
    :param nt: two-element vector defining number of bins in both directions
    """
    # Make grid
    x = np.linspace(w[0], w[1], nt[0] + 1)
    y = np.linspace(w[2], w[3], nt[1] + 1)
    X, Y = np.meshgrid(x, y)

    # Count points
    N = np.zeros([nt[1], nt[0]])
    for i in range(nt[0]):
        for j in range(nt[1]):
            ind = (
                (xy[:, 0] >= x[i])
                & (xy[:, 0] < x[i + 1])
                & (xy[:, 1] >= y[j])
                & (xy[:, 1] < y[j + 1])
            )
            N[j, i] = np.sum(ind)
    return X[:-1, :-1].T, Y[:-1, :-1].T, N.T


def create_spatiotemporal_grid(X, Y):
    """
    create a grid of data sized [T, R1, R2]
    note that this function removes full duplicates (i.e. where all dimensions match)
    TODO: generalise to >5D
    """
    if Y.ndim < 2:
        Y = Y[:, None]
    num_spatial_dims = X.shape[1] - 1
    if num_spatial_dims == 4:
        sort_ind = np.lexsort(
            (X[:, 4], X[:, 3], X[:, 2], X[:, 1], X[:, 0])
        )  # sort by 0, 1, 2, 4
    elif num_spatial_dims == 3:
        sort_ind = np.lexsort(
            (X[:, 3], X[:, 2], X[:, 1], X[:, 0])
        )  # sort by 0, 1, 2, 3
    elif num_spatial_dims == 2:
        sort_ind = np.lexsort((X[:, 2], X[:, 1], X[:, 0]))  # sort by 0, 1, 2
    elif num_spatial_dims == 1:
        sort_ind = np.lexsort((X[:, 1], X[:, 0]))  # sort by 0, 1
    else:
        raise NotImplementedError
    X = X[sort_ind]
    Y = Y[sort_ind]
    unique_time = np.unique(X[:, 0])
    unique_space = np.unique(X[:, 1:], axis=0)
    N_t = unique_time.shape[0]
    N_r = unique_space.shape[0]
    if num_spatial_dims == 4:
        R = np.tile(unique_space, [N_t, 1, 1, 1, 1])
    elif num_spatial_dims == 3:
        R = np.tile(unique_space, [N_t, 1, 1, 1])
    elif num_spatial_dims == 2:
        R = np.tile(unique_space, [N_t, 1, 1])
    elif num_spatial_dims == 1:
        R = np.tile(unique_space, [N_t, 1])
    else:
        raise NotImplementedError
    R_flat = R.reshape(-1, num_spatial_dims)
    Y_dummy = np.nan * np.zeros([N_t * N_r, 1])
    time_duplicate = np.tile(unique_time, [N_r, 1]).T.flatten()
    X_dummy = np.block([time_duplicate[:, None], R_flat])
    X_all = np.vstack([X, X_dummy])
    Y_all = np.vstack([Y, Y_dummy])
    X_unique, ind = np.unique(X_all, axis=0, return_index=True)
    Y_unique = Y_all[ind]
    grid_shape = (unique_time.shape[0],) + unique_space.shape
    R_grid = X_unique[:, 1:].reshape(grid_shape)
    Y_grid = Y_unique.reshape(grid_shape[:-1] + (1,))
    return unique_time[:, None], R_grid, Y_grid
