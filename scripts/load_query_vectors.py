import numpy as np

def read_fvecs(path):
    """
    Reads .fvecs file into a NumPy array of shape (N, 128)
    Each vector is stored as: [dim(int32), float32 * dim]
    """
    data = np.fromfile(path, dtype='int32')
    dim = data[0]
    assert dim == 128, f"Expected 128D, got {dim}"
    data = data.reshape(-1, dim + 1)  
    return data[:, 1:].view('float32')  


queries = read_fvecs("data/sift/sift_query.fvecs")   
print("Loaded:", queries.shape)

import pandas as pd

df_queries = pd.DataFrame({
    "id": range(len(queries)),
    "vec": list(queries)
})

df_queries.to_parquet("sift_query.parquet")
print("Saved sift_query.parquet")