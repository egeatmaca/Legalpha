import os
from numpy.random import seed as set_np_seed
from tensorflow.keras.utils import set_random_seed as set_tf_seed

RANDOM_SEED = os.environ.get('RANDOM_SEED', 42)

def set_random_seeds(seed):
    set_np_seed(seed)
    set_tf_seed(seed)
