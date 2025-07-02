import numpy as np
import os

# Define the path to the original large file
original_file_path = 'train.npy'
output_dir = 'ML project/seoultech-applied-ai-machine-learning1/chunks'
os.makedirs(output_dir, exist_ok=True)

X = None
y = None

try:
    # Attempt to load as a dictionary (e.g., {'X': ..., 'y': ...})
    loaded_data = np.load(original_file_path, allow_pickle=True).item()
    if isinstance(loaded_data, dict):
        print(f"Loaded train.npy as a dictionary. Keys: {loaded_data.keys()}")
        # Use the correct keys: 'input' for X and 'label' for y
        X = loaded_data.get('input')
        y = loaded_data.get('label')

    else:
        # If .item() returns something else, try direct load
        X = loaded_data
        print("Loaded train.npy as a direct object (after .item()).")

except Exception as e:
    print(f"Attempt 1 (dict with .item()) failed: {e}")
    try:
        # Attempt to load as a direct numpy array
        X = np.load(original_file_path, allow_pickle=True)
        print("Loaded train.npy as a direct numpy array.")
    except Exception as e_direct:
        print(f"Attempt 2 (direct numpy array) failed: {e_direct}")
        print(f"Could not load {original_file_path}. Please ensure it's a valid .npy file.")
        exit()

if X is None:
    print("Error: X could not be loaded from train.npy. Please check the keys in your .npy file.")
    exit()

print(f"Type of X: {type(X)}")
if isinstance(X, np.ndarray):
    print(f"Shape of X: {X.shape}")
    print(f"Dimensions of X: {X.ndim}")
else:
    print("X is not a numpy array. Cannot split.")
    exit()

if X.ndim == 0:
    print("X is a scalar (0-dimensional array). Cannot split.")
    exit()

# Determine the number of chunks
num_chunks = 10 # Arbitrary number, can be adjusted
total_elements_X = X.shape[0]
chunk_size_X = total_elements_X // num_chunks

print(f"Splitting X (shape: {X.shape}) into {num_chunks} chunks of size {chunk_size_X}")
if y is not None:
    print(f"Type of y: {type(y)}")
    if isinstance(y, np.ndarray):
        print(f"Shape of y: {y.shape}")
        print(f"Dimensions of y: {y.ndim}")
        if y.ndim > 0:
            total_elements_y = y.shape[0]
            chunk_size_y = total_elements_y // num_chunks
            print(f"Splitting y (shape: {y.shape}) into {num_chunks} chunks of size {chunk_size_y}")
        else:
            print("y is a scalar (0-dimensional array) or not splittable. Will not split y.")
            y = None # Do not attempt to split y if it's not a proper array
    else:
        print("y is not a numpy array. Will not split y.")
        y = None

# Split and save X chunks
for i in range(num_chunks):
    start_idx_X = i * chunk_size_X
    end_idx_X = (i + 1) * chunk_size_X if i < num_chunks - 1 else total_elements_X
    chunk_X = X[start_idx_X:end_idx_X]
    np.save(os.path.join(output_dir, f'train_X_part_{i}.npy'), chunk_X)
    print(f"Saved train_X_part_{i}.npy (shape: {chunk_X.shape})")

# Split and save y chunks if y exists and is splittable
if y is not None and isinstance(y, np.ndarray) and y.ndim > 0:
    for i in range(num_chunks):
        start_idx_y = i * chunk_size_y
        end_idx_y = (i + 1) * chunk_size_y if i < num_chunks - 1 else total_elements_y
        chunk_y = y[start_idx_y:end_idx_y]
        np.save(os.path.join(output_dir, f'train_y_part_{i}.npy'), chunk_y)
        print(f"Saved train_y_part_{i}.npy (shape: {chunk_y.shape})")

print("Splitting complete.")
