"""
code adopted from:
https://github.com/dgilton/deep_equilibrium_inverse/tree/main/utils
"""

import torch
import numpy as np
import os, h5py
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torchvision


import utils.forward_models_mri as forward_models_mri
import fastmri
from fastmri.data import transforms as T


def complex_to_chan_dim(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4:
        b, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 3, 1, 2)  # .reshape(b, 2 * c, h, w)
    else:
        # assumes 3
        h, w, two = x.shape
        assert two == 2
        return x.permute(2, 0, 1)  # .reshape(b, 2 * c, h, w)


def chan_complex_to_last_dim(x: torch.Tensor) -> torch.Tensor:
    b, two, h, w = x.shape
    assert two == 2
    return x.permute(0, 2, 3, 1).contiguous()


def directory_filelist(target_directory):
    file_list = [f for f in os.listdir(target_directory)
                 if os.path.isfile(os.path.join(target_directory, f))]
    file_list = list(file_list)
    file_list = [f for f in file_list if not f.startswith('.')]
    return file_list


def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)
    return torch.from_numpy(data)


def center_crop_slice(data, shape):
    """
    Apply a center crop to the input real image or batch of real images.
    Args:
        data (torch.Tensor): The input tensor to be center cropped. It should have at
            least 2 dimensions and the cropping is applied along the last two dimensions.
        shape (int, int): The output shape. The shape should be smaller than the
            corresponding dimensions of data.
    Returns:
        torch.Tensor: The center cropped image
    """
    assert 0 < shape[0] <= data.shape[0]
    assert 0 < shape[1] <= data.shape[1]
    w_from = (data.shape[0] - shape[0]) // 2
    h_from = (data.shape[1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[w_from:w_to, h_from:h_to, ...]


def complex_abs(data):
    """
    Compute the absolute value of a complex valued input tensor.
    Args:
        data (torch.Tensor): A complex valued tensor, where the size of the final dimension
            should be 2.
    Returns:
        torch.Tensor: Absolute value of data
    """
    assert data.size(-1) == 2
    return (data ** 2).sum(dim=-1).sqrt()


def normalize(
        data,
        mean,
        stddev,
        eps=0.0,
):
    """
    Normalize the given tensor.
    Applies the formula (data - mean) / (stddev + eps).
    Args:
        data: Input data to be normalized.
        mean: Mean value.
        stddev: Standard deviation.
        eps: Added to stddev to prevent dividing by zero.
    Returns:
        Normalized tensor.
    """
    return (data - mean) / (stddev + eps)


def normalize_instance(
        data, eps=0.0):
    """
    Normalize the given tensor  with instance norm/
    Applies the formula (data - mean) / (stddev + eps), where mean and stddev
    are computed from the data itself.
    Args:
        data: Input data to be normalized
        eps: Added to stddev to prevent dividing by zero.
    Returns:
        torch.Tensor: Normalized tensor
    """
    mean = data.mean()
    std = data.std()

    return normalize(data, mean, std, eps), mean, std


class singleCoilFastMRIDataloader(Dataset):
    def __init__(self, dataset_location, transform=None, data_indices=None, sketchynormalize=True):
        """
        Args:
            mask_func (common.subsample.MaskFunc): A function that can create a mask of
                appropriate shape.
            resolution (int): Resolution of the image.
            use_seed (bool): If true, this class computes a pseudo random number generator seed
                from the filename. This ensures that the same mask is used for all the slices of
                a given volume every time.
        """
        self.transform = transform
        if data_indices is not None:
            filelist = directory_filelist(dataset_location)
            print(len(filelist))
            try:
                self.filelist = [filelist[x] for x in data_indices]
            except IndexError:
                print(data_indices)
                exit()
        else:
            self.filelist = directory_filelist(dataset_location)
        self.data_directory = dataset_location
        # self.fft = forward_models_mri.toKspace()
        # self.ifft = forward_models_mri.fromKspace()
        # self.sketchynormalize = sketchynormalize

    def __len__(self):
        return len(self.filelist) * 5  # Select 5 slices per patient

    def __getitem__(self, item):
        index = item % 5
        filename = self.filelist[item // 5]
        hf = h5py.File(self.data_directory + filename)
        volume_kspace = hf['kspace'][()]
        slice_idx = volume_kspace.shape[0] // 2 + index - 2
        slice_kspace = volume_kspace[slice_idx]  # (640, 368)
        X_rss = hf['reconstruction_rss']
        X_esc = hf['reconstruction_esc']
        kspace_cropped = center_crop_slice(slice_kspace, shape=[320, 320])

        slice_kspace2 = T.to_tensor(kspace_cropped)  # Convert from numpy array to pytorch tensor

        # slice_image = fastmri.ifft2c(slice_kspace2)  # Apply Inverse Fourier Transform to get the complex image
        # slice_image_abs = fastmri.complex_abs(slice_image)  # Compute absolute value to get a real image
        # show_coils(slice_image_abs, [0], singlecoil=True, cmap='gray')
        # print(X_esc.shape)
        # print(slice_kspace2.shape)
        return X_esc[slice_idx], slice_kspace2


class singleCoilFastMRIMultiSliceDataset(Dataset):
    def __init__(self, dataset_location, transform=None, data_indices=None, sketchynormalize=True):
        """
        Args:
            mask_func (common.subsample.MaskFunc): A function that can create a mask of
                appropriate shape.
            resolution (int): Resolution of the image.
            use_seed (bool): If true, this class computes a pseudo random number generator seed
                from the filename. This ensures that the same mask is used for all the slices of
                a given volume every time.
        """
        self.transform = transform
        if data_indices is not None:
            filelist = directory_filelist(dataset_location)
            print(len(filelist))
            try:
                self.filelist = [filelist[x] for x in data_indices]
            except IndexError:
                print(data_indices)
                exit()
        else:
            self.filelist = directory_filelist(dataset_location)
        self.data_directory = dataset_location
        # self.fft = forward_models_mri.toKspace()
        # self.ifft = forward_models_mri.fromKspace()
        # self.sketchynormalize = sketchynormalize

    def __len__(self):
        return len(self.filelist) * 5  # Select 5 slices per patient

    def __getitem__(self, item):
        index = item % 5
        filename = self.filelist[item // 5]
        with h5py.File(self.data_directory + filename, 'r') as hf:
            X_esc = hf['reconstruction_esc']
            slice_idx = X_esc.shape[0] // 2 + index * 2 - 4
            X_esc = X_esc[slice_idx][()]
        return X_esc, 0  # slice_kspace2


def show_coils(data, slice_nums, singlecoil, cmap=None):
    fig = plt.figure()
    if singlecoil:
        plt.imshow(data, cmap=cmap)
    else:
        for i, num in enumerate(slice_nums):
            plt.subplot(1, len(slice_nums), i + 1)
            plt.imshow(data[num], cmap=cmap)


class fastmriTransform:
    def __init__(
            self,
            center_fractions=[0.04],
            accelerations=[4],
            use_seed=True
    ):
        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.mask_func = RandomMaskFunc(
            center_fractions=self.center_fractions,
            accelerations=self.accelerations)

        self.use_seed = use_seed

    def __call__(
            self,
            kspace,
            mask,
            target,
            attrs,
            fname,
            slice_num,
    ):
        """
        Converts raw kspace to image, center crops, and then does a masked measurement

        Returns a tuple containing zero filled complex image,
        reconstruction target, max value
        """
        kspace_torch = T.to_tensor(kspace)
        # crop input to correct size
        if target is not None:
            crop_size = (target.shape[-2], target.shape[-1])
        else:
            crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # Convert to image domain, crop
        image = fastmri.ifft2c(kspace_torch)
        # check for FLAIR 203
        if image.shape[-2] < crop_size[1]:
            crop_size = (image.shape[-2], image.shape[-2])
        image = T.complex_center_crop(image, crop_size)

        # map back to kspace
        kspace_cropped = fastmri.fft2c(image)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        # apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            # we only need first element, which is k-space after masking
            masked_kspace = T.apply_mask(kspace_cropped, self.mask_func, seed=seed)[0]
        else:
            masked_kspace = kspace_cropped

        # inverse Fourier transform to get zero filled solution
        image = fastmri.ifft2c(masked_kspace)

        #         image = T.complex_center_crop(image, crop_size)
        #         masked_kspace = T.complex_center_crop(masked_kspace, crop_size)

        # absolute value
        #         image = fastmri.complex_abs(image)

        # apply Root-Sum-of-Squares if multicoil data
        #         if self.which_challenge == "multicoil":
        #             image = fastmri.rss(image)

        # normalize input with magnitude
        #         image_abs = fastmri.complex_abs(image)
        #         mean, std = image_abs.mean(), image_abs.std()
        image, mean, std = T.normalize_instance(image, eps=1e-11)
        #         image = image.clamp(-6, 6)

        # normalize target
        if target is not None:
            target_torch = T.to_tensor(target)
            target_torch = T.center_crop(target_torch, crop_size)
        #             target_torch = T.normalize(target_torch, mean, std, eps=1e-11)
        #             target_torch = target_torch.clamp(-6, 6)
        else:
            target_torch = torch.Tensor([0])

        return DnCNNSample(
            image=image,
            target=target_torch,
            mean=mean,
            std=std,
            fname=fname,
            slice_num=slice_num,
            max_value=max_value,
            #             masked_kspace=masked_kspace
        )