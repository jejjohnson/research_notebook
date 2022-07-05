from sklearn.utils import gen_batches
import tqdm
import numpy as np
import tensorflow.data as tfd
from tqdm import tqdm

def batch_predict(data, fn, batch_size: int=100, dtype=None):
    
    ds = tfd.Dataset.from_tensor_slices(data).batch(batch_size)
    
    predictions = []
    
    with tqdm(ds) as pbar:
        for ix in pbar:

            # predict using GP
            if dtype is not None:
                ix = dtype(ix)
            ipred = fn(ix)

            # add stuff
            predictions.append(ipred)
    
    predictions = np.vstack(predictions)
    return predictions
    