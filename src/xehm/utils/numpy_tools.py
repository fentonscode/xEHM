import numpy as np


# Project a large array into a smaller array with a random selection
def random_sample(arr: np.array, size: int = 1) -> np.array:
    return arr[np.random.choice(len(arr), size=size, replace=False)]
