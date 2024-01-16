from .traindataloader import load_data as load_train_data
from .valdataloader import load_data as load_val_data

def load_data(dataname, batch_size, val_batch_size, data_root, num_workers, is_train, **kwargs):
        if is_train:
                return load_train_data(batch_size, val_batch_size, data_root, num_workers)
        else:
                return load_val_data(batch_size, val_batch_size, data_root, num_workers)
