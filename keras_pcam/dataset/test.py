import os
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Suppress TensorFlow warnings about CUDA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Check TensorFlow version and availability of GPU
print(f"TensorFlow version: {tf.__version__}")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def load_h5_file(file_path, dataset_name):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    with h5py.File(file_path, 'r') as f:
        if dataset_name not in f:
            raise KeyError(f"The dataset {dataset_name} is not found in the file {file_path}.")
        data = np.array(f[dataset_name])
        print(f"Loaded {dataset_name} from {file_path}, shape: {data.shape}")
    return data

def load_data():
    # Ensure the files are placed in this directory
    base_dir = ''
    
    x_train_path = os.path.join(base_dir, 'camelyonpatch_level_2_split_train_x.h5')
    y_train_path = os.path.join(base_dir, 'camelyonpatch_level_2_split_train_y.h5')
    x_valid_path = os.path.join(base_dir, 'camelyonpatch_level_2_split_valid_x.h5')
    y_valid_path = os.path.join(base_dir, 'camelyonpatch_level_2_split_valid_y.h5')
    x_test_path = os.path.join(base_dir, 'camelyonpatch_level_2_split_test_x.h5')
    y_test_path = os.path.join(base_dir, 'camelyonpatch_level_2_split_test_y.h5')
    
    meta_train_path = os.path.join(base_dir, 'camelyonpatch_level_2_split_train_meta.csv')
    meta_valid_path = os.path.join(base_dir, 'camelyonpatch_level_2_split_valid_meta.csv')
    meta_test_path = os.path.join(base_dir, 'camelyonpatch_level_2_split_test_meta.csv')

    # Check if files exist before loading
    for file_path in [x_train_path, y_train_path, x_valid_path, y_valid_path, x_test_path, y_test_path]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist. Please ensure it is placed correctly.")
    
    for meta_path in [meta_train_path, meta_valid_path, meta_test_path]:
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file {meta_path} does not exist. Please ensure it is placed correctly.")

    # Load HDF5 files
    x_train = load_h5_file(x_train_path, 'x')
    y_train = load_h5_file(y_train_path, 'y').reshape(-1, 1)
    x_valid = load_h5_file(x_valid_path, 'x')
    y_valid = load_h5_file(y_valid_path, 'y').reshape(-1, 1)
    x_test = load_h5_file(x_test_path, 'x')
    y_test = load_h5_file(y_test_path, 'y').reshape(-1, 1)

    # Load metadata
    meta_train = pd.read_csv(meta_train_path)
    meta_valid = pd.read_csv(meta_valid_path)
    meta_test = pd.read_csv(meta_test_path)

    return (x_train, y_train, meta_train), (x_valid, y_valid, meta_valid), (x_test, y_test, meta_test)

# Load the dataset
try:
    (x_train, y_train, meta_train), (x_valid, y_valid, meta_valid), (x_test, y_test, meta_test) = load_data()
    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_valid shape: {x_valid.shape}, y_valid shape: {y_valid.shape}")
    print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")
except Exception as e:
    print("An error occurred while loading the data:", e)
    exit(1)

# Define the model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_valid, y_valid))
