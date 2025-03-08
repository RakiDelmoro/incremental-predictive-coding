import numpy as np

def one_hot_label(label_arr):
    one_hot_expected = np.zeros(shape=(label_arr.shape[0], 10))
    one_hot_expected[np.arange(len(label_arr)), label_arr] = 1
    return one_hot_expected

def image_data_batching(img_arr, label_arr, batch_size, shuffle):
    num_train_samples = img_arr.shape[0]    
    # Total samples
    train_indices = np.arange(num_train_samples)
    if shuffle: np.random.shuffle(train_indices)

    for start_idx in range(0, num_train_samples, batch_size):
        end_idx = start_idx + batch_size
        yield img_arr[train_indices[start_idx:end_idx]], one_hot_label(label_arr[train_indices[start_idx:end_idx]])
