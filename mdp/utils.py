import numpy as np
import random

def randomArr(arr):
    if (sum(arr) != 1.0):
        raise ValueError('array distribution does not sum to 1')

    cum_prob = np.cumsum(arr)
    rand = random.random()

    for i in range(len(cum_prob)):
        if (rand < cum_prob[i]):
            return i

    raise ValueError('malformed array distribution')