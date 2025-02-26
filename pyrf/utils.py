import numpy as np


def build_mesh(dx, n_div_x, n_div_y):

    x = np.zeros([n_div_x * n_div_y, 2])
    counter = 0

    for i_y in range(n_div_y):  # depth
        for i_x in range(n_div_x):  # length
            coord_x = dx * i_x
            coord_y = dx * i_y
            x[counter, 0] = coord_x
            x[counter, 1] = coord_y
            counter += 1

    print("Number of nodes:", np.shape(x)[0])

    return x
