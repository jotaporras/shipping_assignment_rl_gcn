import numpy as np
import random

# Run this before every experiment to reset the seeds and ensure same sampling.
def set_seeds(seed):
    random.seed(seed)  # not sure if actually used
    np.random.seed(seed)
    tf.set_random_seed(seed)
