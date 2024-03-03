
import numpy as np
import random
import pickle
import torchvision.transforms.functional as TF
from torchvision import transforms
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from collections import Counter
import os


def preprocess_rgb(rgb_patch):
    """Function to preprocess RGB image patch."""
    return (rgb_patch / 255)

def preprocess_nir(nir_patch):
    """Function to preprocess NIR image patch."""
    return (nir_patch / (2**14 - 1))

def preprocess_dw_cloudy(dw_patch):
    """Function to preprocess land cover patch."""
    dw_patch = np.where(dw_patch == 0, 9, dw_patch) # 00 (=water) to 09
    dw_patch = np.where(dw_patch == 98, 10, dw_patch) # 98 (=clouds) to 10
    dw_patch = np.where(dw_patch == 99, 0, dw_patch) # 99 (=nodata) to 0 (other)
    return dw_patch

def preprocess_pop(pop_patch, log=False):
    """Function to preprocess population patch."""
    # get regression values between 0 and 1 (min and max of population)
    pop_patch = np.where(pop_patch < 0, 0, pop_patch)
    if log:
        return np.log(pop_patch + 1) / np.log(519.5 + 1)
    else:
        return pop_patch / 519.5


class H5Dataset(Dataset):
    """Dataset class to load timeslices of h5 files."""
    def __init__(self, h5_files, dw_files, file_name = {'CLD': 'CLD', 'RBG': 'RGB', 'NIR': 'NIR'}, dw_file_name = 'DW_cloudy', patch_size=(128, 128), cloud_threshold = 1.0, 
                 return_rgb_nir_separately=False, augment=False, num_augment=1):
        
        self.h5_files = h5_files
        self.dw_files = dw_files
        self.file_name = file_name
        self.dw_file_name = dw_file_name
        self.patch_size = patch_size
        self.cloud_threshold = cloud_threshold
        self.return_rgb_nir_separately = return_rgb_nir_separately
        self.tile_lengths = [36, 26, 16, 22]

        self.augment = augment
        self.num_augment = num_augment

        self.h5_file_handles = [h5py.File(h5_file, 'r') for h5_file in h5_files]
        self.dw_file_handles = [h5py.File(dw_file, 'r') for dw_file in dw_files]

        self.patch_starts = []
        self.cloud_perc = []
        self.patches_not_relevant = 0


        # Loop over all h5 files and calculate starting position of patches
        for idx, h5_handle in enumerate(self.h5_file_handles):
            img_shape = h5_handle[self.file_name['RGB']].shape
            for i in range(0, img_shape[0] - self.patch_size[0] + 1, self.patch_size[0]):
                for j in range(0, img_shape[1] - self.patch_size[1] + 1, self.patch_size[1]):

                    # Calculate percentage of cloud cover
                    cld_patch = h5_handle['CLD'][i:i+self.patch_size[0], j:j+self.patch_size[1]] if 'CLD' in h5_handle else np.zeros((self.patch_size[0], self.patch_size[1]))
                    cld_perc = self.percentage_cloud(cld_patch)

                    # if check first for cloud coverage
                    if cld_perc <= self.cloud_threshold:
                        self.patch_starts.append((idx, i, j))
                        self.cloud_perc.append(cld_perc)
    
    
    def percentage_cloud(self, cld_patch):
        """Function to compute percentage of cloud cover."""
        cloud_perc = np.sum(cld_patch == 1) / (self.patch_size[0] * self.patch_size[1])
        return cloud_perc

    def get_indices_based_on_patch_starts(self, patch_starts):
        return [self.patch_starts.index(ps) for ps in patch_starts if ps in self.patch_starts]

    def __len__(self):
        return len(self.patch_starts) * self.num_augment
        
    def __getitem__(self, idx):

        # adjust idx for augmentation
        orig_idx = idx // self.num_augment
        augment_idx = idx % self.num_augment

        # Get start indices of patch
        idx, start_i, start_j = self.patch_starts[orig_idx]
        end_i = start_i + self.patch_size[0]
        end_j = start_j + self.patch_size[1]

        h5_handle = self.h5_file_handles[idx]
        dw_handle = self.dw_file_handles[idx]
    
        # Check if patch is relevant (i.e. built-up area excesses 10% of the patch) 
        dw_patch = dw_handle[self.dw_file_name][start_i:end_i, start_j:end_j]
        rgb_patch = h5_handle[self.file_name['RGB']][start_i:end_i, start_j:end_j, :]
        nir_patch = h5_handle[self.file_name['NIR']][start_i:end_i, start_j:end_j]
        
        rgb_patch = preprocess_rgb(np.array(rgb_patch))
        nir_patch = preprocess_nir(np.array(nir_patch))
        dw_patch = preprocess_dw_cloudy(np.array(dw_patch))
        
        # to tensor
        rgb_tensor = torch.tensor(rgb_patch).permute(2, 0, 1)
        nir_tensor = torch.tensor(nir_patch).unsqueeze(0)
        dw_tensor = torch.tensor(dw_patch).unsqueeze(0)

        if self.return_rgb_nir_separately:
            return rgb_tensor, nir_tensor, dw_tensor
        else:
            combined = torch.cat((rgb_tensor, nir_tensor), dim=0).float()
            return combined, dw_tensor
    
    def get_patch_starts(self):
        return self.patch_starts
    
    def get_patches_not_relevant(self):
        return self.patches_not_relevant


def save_preprocessed_data(dataset, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for idx in range(len(dataset)):
        data = dataset[idx]  # Get preprocessed data from dataset
        torch.save(data, os.path.join(save_dir, f'data_{idx}.pt'))


class PreprocessedValDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith('data_')]

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data_file = self.data_files[idx]
        data = torch.load(data_file)
        return data


def load_model(model, model_path, device, NUM_CHANNELS, NUM_CLASSES):
    """Function to load a model from a file."""
    model = model(NUM_CHANNELS, NUM_CLASSES)
    model.to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path))
    return model


def predict_on_patch(model, dataset, idx, device):
    """Function to predict on a single patch."""
    patch = dataset[idx][0].unsqueeze(0)
    patch = patch.to(device)
    pred = model(patch)
    pred = torch.softmax(pred, dim=1)
    pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
    return pred


def count_labels_in_batches(dataset):
    label_counter = Counter()
    for _, labels in dataset:
        flattened_labels = labels.flatten().numpy()
    return label_counter


class H5Dataset_early(Dataset):
    """Dataset class to load timeslices of h5 files."""
    def __init__(self, h5_files, dw_files, file_name = {'CLD': 'CLD', 'RBG': 'RGB', 'NIR': 'NIR'}, dw_file_name = 'DW_cloudy', patch_size=(128, 128), cloud_threshold = 1.0, 
                 return_rgb_nir_separately=False, calculate_time_slice_indices=False, timesteps = 3, time_file_name = 'time_slice_indices_precomputed_256_early3.pkl'):
        
        self.h5_files = h5_files
        self.dw_files = dw_files
        self.file_name = file_name
        self.dw_file_name = dw_file_name
        self.patch_size = patch_size
        self.cloud_threshold = cloud_threshold
        self.return_rgb_nir_separately = return_rgb_nir_separately
        self.tile_lengths = [36, 26, 16, 22]
        self.timesteps = timesteps

        self.h5_file_handles = [h5py.File(h5_file, 'r') for h5_file in h5_files]
        self.dw_file_handles = [h5py.File(dw_file, 'r') for dw_file in dw_files]

        self.patch_starts = []
        self.cloud_perc = []
        self.patches_not_relevant = 0

        # Loop over all h5 files and calculate starting position of patches
        for idx, h5_handle in enumerate(self.h5_file_handles):
            img_shape = h5_handle[self.file_name['RGB']].shape
            for i in range(0, img_shape[0] - self.patch_size[0] + 1, self.patch_size[0]):
                for j in range(0, img_shape[1] - self.patch_size[1] + 1, self.patch_size[1]):

                    # Calculate percentage of cloud cover
                    cld_patch = h5_handle['CLD'][i:i+self.patch_size[0], j:j+self.patch_size[1]] if 'CLD' in h5_handle else np.zeros((self.patch_size[0], self.patch_size[1]))
                    cld_perc = self.percentage_cloud(cld_patch)

                    # if check first for cloud coverage
                    if cld_perc <= self.cloud_threshold:
                        self.patch_starts.append((idx, i, j))
                        self.cloud_perc.append(cld_perc)
        
        # precompute time slice indices
        if calculate_time_slice_indices:
            self.time_slice_indices_precomputed = self.precompute_time_slice_indices()
            self.save_time_slice_indices(time_file_name)
        else:
            self.load_time_slice_indices(time_file_name)

    
    def get_time_slice_indices(self, patch_idx):
        """
        Get indices for the current and previous two time slices for the given patch index.
        Apply padding if there are not enough previous time slices.
        """
        
        # Get patch start based on patch index
        tile_idx, start_i, start_j = self.patch_starts[patch_idx]

        # handle cases 3 and 5
        if self.timesteps == 3:
            if tile_idx in [0, 36, 62, 78]:
                return [patch_idx, patch_idx, patch_idx]
            if tile_idx in [1, 37, 63, 79]:
                patch_idx_1 = self.get_indices_based_on_patch_starts([(tile_idx - 1, start_i, start_j)])[0]
                return [patch_idx_1, patch_idx_1, patch_idx]
            else:
                patch_idx_1 = self.get_indices_based_on_patch_starts([(tile_idx - 1, start_i, start_j)])[0]
                patch_idx_2 = self.get_indices_based_on_patch_starts([(tile_idx - 2, start_i, start_j)])[0]
                return [patch_idx_2, patch_idx_1, patch_idx]
        
        if self.timesteps == 5:
            if tile_idx in [0, 36, 62, 78]:
                return [patch_idx, patch_idx, patch_idx, patch_idx, patch_idx]
            if tile_idx in [1, 37, 63, 79]:
                patch_idx_1 = self.get_indices_based_on_patch_starts([(tile_idx - 1, start_i, start_j)])[0]
                return [patch_idx_1, patch_idx_1, patch_idx_1, patch_idx_1, patch_idx]
            if tile_idx in [2, 38, 64, 80]:
                patch_idx_1 = self.get_indices_based_on_patch_starts([(tile_idx - 1, start_i, start_j)])[0]
                patch_idx_2 = self.get_indices_based_on_patch_starts([(tile_idx - 2, start_i, start_j)])[0]
                return [patch_idx_2, patch_idx_2, patch_idx_1, patch_idx_1, patch_idx]
            if tile_idx in [3, 39, 65, 81]:
                patch_idx_1 = self.get_indices_based_on_patch_starts([(tile_idx - 1, start_i, start_j)])[0]
                patch_idx_2 = self.get_indices_based_on_patch_starts([(tile_idx - 2, start_i, start_j)])[0]
                patch_idx_3 = self.get_indices_based_on_patch_starts([(tile_idx - 3, start_i, start_j)])[0]
                return [patch_idx_3, patch_idx_2, patch_idx_1, patch_idx_1, patch_idx]
            else:
                patch_idx_1 = self.get_indices_based_on_patch_starts([(tile_idx - 1, start_i, start_j)])[0]
                patch_idx_2 = self.get_indices_based_on_patch_starts([(tile_idx - 2, start_i, start_j)])[0]
                patch_idx_3 = self.get_indices_based_on_patch_starts([(tile_idx - 3, start_i, start_j)])[0]
                patch_idx_4 = self.get_indices_based_on_patch_starts([(tile_idx - 4, start_i, start_j)])[0]
                return [patch_idx_4, patch_idx_3, patch_idx_2, patch_idx_1, patch_idx]

    def precompute_time_slice_indices(self):
        """
        Precompute time slice indices for all patches.
        """
        precomputed_indices = []
        for idx in range(len(self.patch_starts)):
            precomputed_indices.append(self.get_time_slice_indices(idx))
        return precomputed_indices
    
    def save_time_slice_indices(self, file_name):
        """
        Save precomputed time slice indices to file.
        """
        with open(file_name, 'wb') as f:
            pickle.dump(self.time_slice_indices_precomputed, f)

    def load_time_slice_indices(self, file_name):
        """
        Load precomputed time slice indices from file.
        """
        with open(file_name, 'rb') as f:
            self.time_slice_indices_precomputed = pickle.load(f)
    
    def percentage_cloud(self, cld_patch):
        """Function to compute percentage of cloud cover."""
        cloud_perc = np.sum(cld_patch == 1) / (self.patch_size[0] * self.patch_size[1])
        return cloud_perc

    def get_indices_based_on_patch_starts(self, patch_starts):
        return [self.patch_starts.index(ps) for ps in patch_starts if ps in self.patch_starts]

    def __len__(self):
        return len(self.patch_starts) 
    
    def __getitem__(self, idx):

        #time_slice_indices = self.get_time_slice_indices(idx)
        time_slice_indices = self.time_slice_indices_precomputed[idx]

        combined_data = []
        for ts_idx in time_slice_indices:
            # Extract patch index and coordinates
            tile_idx, start_i, start_j = self.patch_starts[ts_idx]
            end_i = start_i + self.patch_size[0]
            end_j = start_j + self.patch_size[1]

            # Load data for the specified time slice
            h5_handle = self.h5_file_handles[tile_idx]
            rgb_patch = h5_handle[self.file_name['RGB']][start_i:end_i, start_j:end_j, :]
            nir_patch = h5_handle[self.file_name['NIR']][start_i:end_i, start_j:end_j]

            # Preprocess and concatenate RGB and NIR patches
            rgb_patch_processed = preprocess_rgb(np.array(rgb_patch))
            nir_patch_processed = preprocess_nir(np.array(nir_patch))
            combined_patch = torch.cat((torch.tensor(rgb_patch_processed).permute(2, 0, 1), torch.tensor(nir_patch_processed).unsqueeze(0)), dim=0)
            combined_data.append(combined_patch)

        # Concatenate data from all time slices
        combined_data_tensor = torch.cat(combined_data, dim=0).float()

        # Assuming dw_tensor is the label or target tensor
        # Load dw_tensor for the current patch (idx) as label
        dw_tensor = self.load_dw_tensor(idx)

        return combined_data_tensor, dw_tensor
    
    def load_dw_tensor(self, idx):

        tile_idx, start_i, start_j = self.patch_starts[idx]
        end_i = start_i + self.patch_size[0]
        end_j = start_j + self.patch_size[1]
        dw_handle = self.dw_file_handles[tile_idx]
        dw_patch = dw_handle[self.dw_file_name][start_i:end_i, start_j:end_j]
        dw_patch_processed = preprocess_dw_cloudy(np.array(dw_patch))
        dw_tensor = torch.tensor(dw_patch_processed).unsqueeze(0)
        return dw_tensor

    def get_patch_starts(self):
        return self.patch_starts
    
    def get_patches_not_relevant(self):
        return self.patches_not_relevant


def save_preprocessed_data(dataset, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for idx in range(len(dataset)):
        data = dataset[idx]  # Get preprocessed data from dataset
        torch.save(data, os.path.join(save_dir, f'data_{idx}.pt'))


class PreprocessedValDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith('data_')]

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data_file = self.data_files[idx]
        data = torch.load(data_file)
        return data


class H5Dataset_score(Dataset):
    """Dataset class to load timeslices of h5 files."""
    def __init__(self, h5_files, dw_files, file_name = {'CLD': 'CLD', 'RBG': 'RGB', 'NIR': 'NIR'}, dw_file_name = 'DW_cloudy', patch_size=(128, 128), cloud_threshold = 1.0, 
                 return_rgb_nir_separately=False, calculate_time_slice_indices=False, timesteps = 3, time_file_name = 'time_slice_indices_precomputed_256_score3.pkl'):
        
        self.h5_files = h5_files
        self.dw_files = dw_files
        self.file_name = file_name
        self.dw_file_name = dw_file_name
        self.patch_size = patch_size
        self.cloud_threshold = cloud_threshold
        self.return_rgb_nir_separately = return_rgb_nir_separately
        self.tile_lengths = [36, 26, 16, 22]
        self.timesteps = timesteps

        self.h5_file_handles = [h5py.File(h5_file, 'r') for h5_file in h5_files]
        self.dw_file_handles = [h5py.File(dw_file, 'r') for dw_file in dw_files]

        self.patch_starts = []
        self.cloud_perc = []
        self.patches_not_relevant = 0

        # Loop over all h5 files and calculate starting position of patches
        for idx, h5_handle in enumerate(self.h5_file_handles):
            img_shape = h5_handle[self.file_name['RGB']].shape
            for i in range(0, img_shape[0] - self.patch_size[0] + 1, self.patch_size[0]):
                for j in range(0, img_shape[1] - self.patch_size[1] + 1, self.patch_size[1]):

                    # Calculate percentage of cloud cover
                    cld_patch = h5_handle['CLD'][i:i+self.patch_size[0], j:j+self.patch_size[1]] if 'CLD' in h5_handle else np.zeros((self.patch_size[0], self.patch_size[1]))
                    cld_perc = self.percentage_cloud(cld_patch)

                    # if check first for cloud coverage
                    if cld_perc <= self.cloud_threshold:
                        self.patch_starts.append((idx, i, j))
                        self.cloud_perc.append(cld_perc)
        
        # precompute time slice indices
        if calculate_time_slice_indices:
            self.time_slice_indices_precomputed = self.precompute_time_slice_indices()
            self.save_time_slice_indices(time_file_name)
        else:
            self.load_time_slice_indices(time_file_name)

    
    def get_time_slice_indices(self, patch_idx):
        """
        Get indices for the current and previous two time slices for the given patch index.
        Apply padding if there are not enough previous time slices.
        """
        
        # Get patch start based on patch index
        tile_idx, start_i, start_j = self.patch_starts[patch_idx]

        # handle cases 3 and 5
        if self.timesteps == 3:
            if tile_idx in [0, 36, 62, 78]:
                return [patch_idx, patch_idx, patch_idx]
            if tile_idx in [1, 37, 63, 79]:
                patch_idx_1 = self.get_indices_based_on_patch_starts([(tile_idx - 1, start_i, start_j)])[0]
                return [patch_idx_1, patch_idx_1, patch_idx]
            else:
                patch_idx_1 = self.get_indices_based_on_patch_starts([(tile_idx - 1, start_i, start_j)])[0]
                patch_idx_2 = self.get_indices_based_on_patch_starts([(tile_idx - 2, start_i, start_j)])[0]
                return [patch_idx_2, patch_idx_1, patch_idx]
        
        if self.timesteps == 5:
            if tile_idx in [0, 36, 62, 78]:
                return [patch_idx, patch_idx, patch_idx, patch_idx, patch_idx]
            if tile_idx in [1, 37, 63, 79]:
                patch_idx_1 = self.get_indices_based_on_patch_starts([(tile_idx - 1, start_i, start_j)])[0]
                return [patch_idx_1, patch_idx_1, patch_idx_1, patch_idx_1, patch_idx]
            if tile_idx in [2, 38, 64, 80]:
                patch_idx_1 = self.get_indices_based_on_patch_starts([(tile_idx - 1, start_i, start_j)])[0]
                patch_idx_2 = self.get_indices_based_on_patch_starts([(tile_idx - 2, start_i, start_j)])[0]
                return [patch_idx_2, patch_idx_2, patch_idx_1, patch_idx_1, patch_idx]
            if tile_idx in [3, 39, 65, 81]:
                patch_idx_1 = self.get_indices_based_on_patch_starts([(tile_idx - 1, start_i, start_j)])[0]
                patch_idx_2 = self.get_indices_based_on_patch_starts([(tile_idx - 2, start_i, start_j)])[0]
                patch_idx_3 = self.get_indices_based_on_patch_starts([(tile_idx - 3, start_i, start_j)])[0]
                return [patch_idx_3, patch_idx_2, patch_idx_1, patch_idx_1, patch_idx]
            else:
                patch_idx_1 = self.get_indices_based_on_patch_starts([(tile_idx - 1, start_i, start_j)])[0]
                patch_idx_2 = self.get_indices_based_on_patch_starts([(tile_idx - 2, start_i, start_j)])[0]
                patch_idx_3 = self.get_indices_based_on_patch_starts([(tile_idx - 3, start_i, start_j)])[0]
                patch_idx_4 = self.get_indices_based_on_patch_starts([(tile_idx - 4, start_i, start_j)])[0]
                return [patch_idx_4, patch_idx_3, patch_idx_2, patch_idx_1, patch_idx]

    def precompute_time_slice_indices(self):
        """
        Precompute time slice indices for all patches.
        """
        precomputed_indices = []
        for idx in range(len(self.patch_starts)):
            precomputed_indices.append(self.get_time_slice_indices(idx))
        return precomputed_indices
    
    def save_time_slice_indices(self, file_name):
        """
        Save precomputed time slice indices to file.
        """
        with open(file_name, 'wb') as f:
            pickle.dump(self.time_slice_indices_precomputed, f)

    def load_time_slice_indices(self, file_name):
        """
        Load precomputed time slice indices from file.
        """
        with open(file_name, 'rb') as f:
            self.time_slice_indices_precomputed = pickle.load(f)
    
    def percentage_cloud(self, cld_patch):
        """Function to compute percentage of cloud cover."""
        cloud_perc = np.sum(cld_patch == 1) / (self.patch_size[0] * self.patch_size[1])
        return cloud_perc

    def get_indices_based_on_patch_starts(self, patch_starts):
        return [self.patch_starts.index(ps) for ps in patch_starts if ps in self.patch_starts]

    def __len__(self):
        return len(self.patch_starts) 
    
    def __getitem__(self, idx):

        #time_slice_indices = self.get_time_slice_indices(idx)
        time_slice_indices = self.time_slice_indices_precomputed[idx]

        combined_data = []
        for ts_idx in time_slice_indices:
            # Extract patch index and coordinates
            tile_idx, start_i, start_j = self.patch_starts[ts_idx]
            end_i = start_i + self.patch_size[0]
            end_j = start_j + self.patch_size[1]

            # Load data for the specified time slice
            h5_handle = self.h5_file_handles[tile_idx]
            rgb_patch = h5_handle[self.file_name['RGB']][start_i:end_i, start_j:end_j, :]
            nir_patch = h5_handle[self.file_name['NIR']][start_i:end_i, start_j:end_j]

            # Preprocess and concatenate RGB and NIR patches
            rgb_patch_processed = preprocess_rgb(np.array(rgb_patch))
            nir_patch_processed = preprocess_nir(np.array(nir_patch))
            combined_patch = torch.cat((torch.tensor(rgb_patch_processed).permute(2, 0, 1), torch.tensor(nir_patch_processed).unsqueeze(0)), dim=0)
            combined_data.append(combined_patch)

        # load dw data
        dw_tensor = self.load_dw_tensor(idx)

        # delist combined_data
        if self.timesteps == 3:
            return combined_data[0], combined_data[1], combined_data[2], dw_tensor
        if self.timesteps == 5:
            return combined_data[0], combined_data[1], combined_data[2], combined_data[3], combined_data[4], dw_tensor

    def load_dw_tensor(self, idx):

        tile_idx, start_i, start_j = self.patch_starts[idx]
        end_i = start_i + self.patch_size[0]
        end_j = start_j + self.patch_size[1]
        dw_handle = self.dw_file_handles[tile_idx]
        dw_patch = dw_handle[self.dw_file_name][start_i:end_i, start_j:end_j]
        dw_patch_processed = preprocess_dw_cloudy(np.array(dw_patch))
        dw_tensor = torch.tensor(dw_patch_processed).unsqueeze(0)
        return dw_tensor

    def get_patch_starts(self):
        return self.patch_starts
    
    def get_patches_not_relevant(self):
        return self.patches_not_relevant