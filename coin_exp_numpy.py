import numpy as np
%%time

def flip(maxint: np.int64 = 11) -> np.int64:
    return np.int64(np.random.randint(maxint))


if __name__ == "__main__":
    [x for i in range(flip())]
