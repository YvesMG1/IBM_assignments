
import numpy as np
import random
import torchvision.transforms.functional as TF
from torchvision import transforms
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset



def preprocess_rgb(rgb_patch):
    """Function to preprocess RGB image patch."""
    return (rgb_patch / 255)

def preprocess_nir(nir_patch):
    """Function to preprocess NIR image patch."""
    return (nir_patch / (2**14 - 1))

def preprocess_label(label_patch):
    """Function to preprocess label patch."""
    return np.where(label_patch == 99, 0, label_patch)


class CustomRandomTransform(object):
    """Class to perform random transformations on the image and label."""
    def __init__(self, angle = 10, scale = (0.9, 1.1)):
        self.angle = angle
        self.scale = scale
    
    def __call__(self, rgb_img, nir_img, mask):

        # Random rotation
        angle = random.uniform(-self.angle, self.angle)
        rgb_img = TF.rotate(rgb_img, angle)
        nir_img = TF.rotate(nir_img, angle)
        mask = TF.rotate(mask, angle)

        # Random horizontal flip
        if random.random() > 0.5:
            rgb_img = TF.hflip(rgb_img)
            nir_img = TF.hflip(nir_img)
            mask = TF.hflip(mask)

        # Random vertical flip
        if random.random() > 0.5:
            rgb_img = TF.vflip(rgb_img)
            nir_img = TF.vflip(nir_img)
            mask = TF.vflip(mask)

        # Random scale
        scale = random.uniform(self.scale[0], self.scale[1])
        rgb_img = TF.affine(rgb_img, 0, (0, 0), scale, 0)
        nir_img = TF.affine(nir_img, 0, (0, 0), scale, 0)
        mask = TF.affine(mask, 0, (0, 0), scale, 0)

        # Random crop
        if random.random() > 0.5:
            i, j, h, w = transforms.RandomCrop.get_params(rgb_img, output_size=(128, 128))
            rgb_img = TF.crop(rgb_img, i, j, h, w)
            nir_img = TF.crop(nir_img, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)

        return rgb_img, nir_img, mask
    
augment_transform = CustomRandomTransform()
    
class Smoothed_H5Dataset(Dataset):
    """Dataset class to load smoothed h5 files."""
    def __init__(self, h5_files, BLTh5_files, patch_size=(128, 128), relevant_threshold=0.1, return_rgb_nir_separately=False, augment=False, num_augment=1, 
    augment_transform=None):
        self.h5_files = h5_files
        self.BLTh5_files = BLTh5_files
        self.patch_size = patch_size
        self.relevant_threshold = relevant_threshold
        self.return_rgb_nir_separately = return_rgb_nir_separately
        self.augment = augment
        self.num_augment = num_augment

        self.h5_file_handles = [h5py.File(h5_file, 'r') for h5_file in h5_files]
        self.BLT_file_handles = [h5py.File(BLTh5_file, 'r') for BLTh5_file in BLTh5_files]

        # Calulate starting position of patches
        self.patch_starts = []
        self.patch_perc = []
        self.patches_not_relevant = 0
        for idx, h5_handle in enumerate(self.h5_file_handles):
            blt_handle = self.BLT_file_handles[idx]
            img_shape = h5_handle['RGB_cloudfree_averaged'].shape
            for i in range(0, img_shape[0] - self.patch_size[0] + 1, self.patch_size[0]):
                for j in range(0, img_shape[1] - self.patch_size[1] + 1, self.patch_size[1]):
                    blt_patch = blt_handle['BLT'][i:i+self.patch_size[0], j:j+self.patch_size[1]]
                    blt_perc = self.percentage_patch_relevance(blt_patch)
                    if blt_perc >= self.relevant_threshold:
                        self.patch_starts.append((idx, i, j))
                        self.patch_perc.append(blt_perc)
                    else:
                        self.patches_not_relevant += 1
                        
    def percentage_patch_relevance(self, blt_patch):
        """Function to compute percentage of built-up area."""
        blt_perc = blt_patch.sum() / (self.patch_size[0] * self.patch_size[1])
        return blt_perc
    
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
        blt_handle = self.BLT_file_handles[idx]

        # Check if patch is relevant (i.e. built-up area excesses 10% of the patch) 
        blt_patch = blt_handle['BLT'][start_i:end_i, start_j:end_j]
        rgb_patch = h5_handle['RGB_cloudfree_averaged'][start_i:end_i, start_j:end_j, :]
        nir_patch = h5_handle['NIR_cloudfree_averaged'][start_i:end_i, start_j:end_j]

        assert not np.isnan(rgb_patch).any(), "RGB patch contains NaNs"
        assert not np.isinf(nir_patch).any(), "RGB patch contains Infs"

        assert not np.isnan(rgb_patch).any(), "NIR patch contains NaNs"
        assert not np.isinf(nir_patch).any(), "NIR patch contains Infs"

        if self.augment and augment_idx > 0:
            # to PIL image
            rgb_patch = TF.to_pil_image(rgb_patch.astype(np.uint8))
            nir_patch = TF.to_pil_image(nir_patch.astype(np.uint8))
            blt_patch = TF.to_pil_image(blt_patch)

            # augment
            rgb_patch, nir_patch, blt_patch = augment_transform(rgb_patch, nir_patch, blt_patch)

            # to numpy array
        rgb_patch = preprocess_rgb(np.array(rgb_patch))
        nir_patch = preprocess_nir(np.array(nir_patch))
        blt_patch = preprocess_label(np.array(blt_patch))

        # to tensor
        rgb_tensor = torch.tensor(rgb_patch).permute(2, 0, 1)
        nir_tensor = torch.tensor(nir_patch).unsqueeze(0)
        blt_tensor = torch.tensor(blt_patch).unsqueeze(0)

        if self.return_rgb_nir_separately:
            return rgb_tensor, nir_tensor, blt_tensor
        else:
            combined = torch.cat((rgb_tensor, nir_tensor), dim=0).float()
            return combined, blt_tensor
    
    def get_patch_starts(self):
        return self.patch_starts
    
    def get_patch_perc(self):
        return self.patch_perc
    
    def get_patches_not_relevant(self):
        return self.patches_not_relevant
    

class Custom_Testing_Dataset(Dataset):
    """Dataset class for smoothed h5 files."""
    def __init__(self, h5_file, BLTh5_file, patch_size=(128, 128), return_rgb_nir_separately=False, relevant_threshold = 0.0):
        self.h5_file = h5_file
        self.BLTh5_file = BLTh5_file
        self.patch_size = patch_size
        self.return_rgb_nir_separately = return_rgb_nir_separately
        self.relevant_threshold = relevant_threshold

        self.h5_file_handles = [h5py.File(h5_file, 'r')]
        self.BLT_file_handles = [h5py.File(BLTh5_file, 'r')]
        
        # Calulate starting position of patches
        self.patch_starts = []
        self.patch_perc = []
        self.patches_not_relevant = 0

        for idx, h5_handle in enumerate(self.h5_file_handles):
            blt_handle = self.BLT_file_handles[idx]
            img_shape = h5_handle['RGB_cloudfree_averaged'].shape
            for i in range(0, img_shape[0] - self.patch_size[0] + 1, self.patch_size[0]):
                for j in range(0, img_shape[1] - self.patch_size[1] + 1, self.patch_size[1]):
                    blt_patch = blt_handle['BLT'][i:i+self.patch_size[0], j:j+self.patch_size[1]]
                    blt_perc = self.percentage_patch_relevance(blt_patch)
                    if blt_perc >= self.relevant_threshold:
                        self.patch_starts.append((idx, i, j))
                        self.patch_perc.append(blt_perc)
                    else:
                        self.patches_not_relevant += 1
                        
    def percentage_patch_relevance(self, blt_patch):
        """Function to compute percentage of built-up area."""
        blt_perc = blt_patch.sum() / (self.patch_size[0] * self.patch_size[1])
        return blt_perc         
                        

    def __len__(self):
        return len(self.patch_starts)
        
    def __getitem__(self, idx):
        #Index
        idx, start_i, start_j = self.patch_starts[idx]

        h5_handle = self.h5_file_handles[idx]
        blt_handle = self.BLT_file_handles[idx]

        end_i = start_i + self.patch_size[0]
        end_j = start_j + self.patch_size[1]

        #Patches 
        blt_patch = preprocess_label(blt_handle['BLT'][start_i:end_i, start_j:end_j])
        rgb_patch = preprocess_rgb(h5_handle['RGB_cloudfree_averaged'][start_i:end_i, start_j:end_j, :])
        nir_patch = preprocess_nir(h5_handle['NIR_cloudfree_averaged'][start_i:end_i, start_j:end_j])

        #Tensors
        rgb_tensor = torch.tensor(rgb_patch).permute(2, 0, 1)
        nir_tensor = torch.tensor(nir_patch).unsqueeze(0)            
        blt_tensor = torch.tensor(blt_patch).unsqueeze(0)

        if self.return_rgb_nir_separately:
            return rgb_tensor, nir_tensor, blt_tensor
        else:
            combined = torch.cat((rgb_tensor, nir_tensor), dim=0).float()
            return combined, blt_tensor

    def get_patch_starts(self):
        return self.patch_starts