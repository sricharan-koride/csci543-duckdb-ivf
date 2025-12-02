import numpy as np
import pandas as pd
import os

def read_fvecs(filename):
    """
    Reads an .fvecs file into a 2D NumPy array.
    The .fvecs format is [dim (int32), vector (float32 * dim)]
    """
    # Define the structured datatype for one vector record
    # SIFT is 128 dimensions, so 'i4' (4-byte int) + 'f4' (4-byte float) x 128
    dt = np.dtype([('dim', 'i4'), ('vec', 'f4', 128)])
    
    with open(filename, "rb") as f:
        # Read the entire file using the structured datatype
        data = np.fromfile(f, dtype=dt)
        
    # The 'vec' field of the structured array is our 2D array of vectors
    vectors = data['vec']
    return vectors

# Get the script's directory to build relative paths
SCRIPT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data', 'sift')

BASE_FILE = os.path.join(DATA_DIR, 'sift_base.fvecs')
PARQUET_FILE = os.path.join(DATA_DIR, 'sift_base.parquet')

if not os.path.exists(BASE_FILE):
    print(f"Error: File not found at '{BASE_FILE}'")
    print("Please make sure 'sift_base.fvecs' is in the 'data/sift' directory.")
elif os.path.exists(PARQUET_FILE):
    print(f"'{PARQUET_FILE}' already exists. Skipping conversion.")
else:
    print(f"Reading '{BASE_FILE}'...")
    # Load the 1 million 128-dim vectors
    vectors = read_fvecs(BASE_FILE)
    
    print(f"Loaded {vectors.shape[0]} vectors of dim {vectors.shape[1]}")
    
    # Create a DataFrame to save to Parquet
    # We need an 'id' column and a 'vec' column
    df = pd.DataFrame({
        'id': np.arange(vectors.shape[0]),
        'vec': list(vectors)  
    })
    
    print(f"Saving to '{PARQUET_FILE}'...")
    # Save as Parquet
    df.to_parquet(PARQUET_FILE)
    
    print("Done.")