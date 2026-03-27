import os
from torch.utils.data import Dataset
import torch
import h5py
import odl
import torch.nn.functional as F
import numpy as np

class LoDoPaB_Dataset(Dataset):
    """
    PyTorch Dataset for the LoDoPaB-CT (Low-Dose Parallel Beam CT) dataset.

    Loads paired sinogram (observation) and ground truth image data from HDF5
    files. Each HDF5 file is expected to contain up to 128 slices stored under
    the key 'data'. Files are matched by sort order, so naming conventions must
    ensure that sinogram and ground truth files correspond 1-to-1.

    Dataset structure expected on disk:
        sino_dir/
            observation_train_000.hdf5   # sinogram files
            observation_train_001.hdf5
            ...
        gt_images_dir/
            ground_truth_train_000.hdf5  # ground truth files
            ground_truth_train_001.hdf5
            ...

    Args:
        sino_dir (str):
            Path to the directory containing sinogram HDF5 files.
            Only files whose names contain 'observation' are loaded.
        gt_images_dir (str):
            Path to the directory containing ground truth HDF5 files.
            Only files whose names contain 'ground_truth' are loaded.
        transform (callable, optional):
            Transform applied to the sinogram tensor after loading.
            Receives a tensor of shape (1, 1000, 513) and must return
            a tensor. Default: None.
        target_transform (callable, optional):
            Transform applied to the ground truth image tensor after loading.
            Receives a tensor of shape (1, 362, 362) and must return
            a tensor. Default: None.
        suffix (str, optional):
            If provided, only files containing this substring in their name
            are included. Useful to restrict loading to a specific split,
            e.g. suffix='train' or suffix='validation'. Default: None.
        amount_images (int, optional):
            Maximum total number of slices to include in the dataset.
            Slices are counted globally across all files. If None, all
            available slices are used. Default: None.

    Attributes:
        index_map (list of tuple): Maps a flat dataset index to
            (file_idx, slice_idx) for efficient lazy loading.

    Returns (per __getitem__):
        tuple[torch.Tensor, torch.Tensor]:
            - sino:     Sinogram tensor of shape (1, 1000, 513)
            - gt_image: Ground truth image tensor of shape (1, 362, 362)

    Example:
        >>> dataset = LoDoPaB_Dataset(
        ...     sino_dir='data/sinograms',
        ...     gt_images_dir='data/ground_truth',
        ...     suffix='train',
        ...     amount_images=1000
        ... )
        >>> sino, gt = dataset[0]
        >>> print(sino.shape, gt.shape)
        torch.Size([1, 1000, 513]) torch.Size([1, 362, 362])

        >>> loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    """

    def __init__(self, sino_dir, gt_images_dir, transform=None, target_transform=None, suffix=None, amount_images=None):
        self.gt_image_names = sorted([x for x in os.listdir(gt_images_dir) if 'ground_truth' in x])
        self.sino_names = sorted([x for x in os.listdir(sino_dir) if 'observation' in x])

        if suffix:
            self.gt_image_names = [x for x in self.gt_image_names if suffix in x]
            self.sino_names = [x for x in self.sino_names if suffix in x]

        self.gt_image_files = [os.path.join(gt_images_dir, x) for x in self.gt_image_names]
        self.sino_files = [os.path.join(sino_dir, x) for x in self.sino_names]

        # Important check: len(observations) == len(targets)?
        assert len(self.gt_image_files) == len(self.sino_files)
        
        # Assume each file contains 128 slices except maybe the last
        if isinstance(amount_images, int) and amount_images > 0:
            max_file_idx = amount_images // 128
            max_number = amount_images % 128
            self.index_map = [
                (file_idx, i)
                for file_idx in range(max_file_idx + (1 if max_number else 0))
                for i in range(max_number if file_idx == max_file_idx else 128)
            ]
        else:
            self.index_map = []
            for file_idx, gt_path in enumerate(self.gt_image_files):
                with h5py.File(gt_path, 'r') as gt_file:
                    self.index_map.extend((file_idx, i) for i in range(len(gt_file['data'])))

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, data_idx = self.index_map[idx]

        with h5py.File(self.sino_files[file_idx], 'r') as sino_file:
            sino = torch.from_numpy(sino_file['data'][data_idx])[None, :, :]

        with h5py.File(self.gt_image_files[file_idx], 'r') as gt_file:
            gt_image = torch.from_numpy(gt_file['data'][data_idx])[None, :, :]

        if self.transform:
            sino = self.transform(sino)
        if self.target_transform:
            gt_image = self.target_transform(gt_image)

        return sino, gt_image


##############################################################
def crop_zoom_top_left(image: torch.Tensor, x: int, y: int, width: int, height: int):
    """
    Crops a zoomed-in region from a grayscale image tensor using top-left coordinates.
    Args:
        image (torch.Tensor): 2D tensor (H, W) or 3D tensor (C, H, W)
        x (int): x-coordinate (column) of the top-left corner
        y (int): y-coordinate (row) of the top-left corner
        width (int): width of the crop
        height (int): height of the crop
    Returns:
        torch.Tensor: Cropped image tensor
    """
    if image.dim() == 2:
        return image[y:y+height, x:x+width]
    elif image.dim() == 3:
        return image[:, y:y+height, x:x+width]
    else:
        raise ValueError("Input image must be 2D or 3D tensor.")
##############################################################

def min_max_normalize(x, eps=1e-8):
    # x: (B, 1, H, W) or (B, H, W)
    x_min = x.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
    x_max = x.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
    x_norm = (x - x_min) / (x_max - x_min + eps)
    return x_norm


def gt_to_coeffs(Y, max=81.35858):
    return Y * max

def X_to_minuspostlog(X, max=81.35858):
    return (X * max)

def minuspostlog_to_proj(X, N_0=4096):
    return torch.exp(-X)*N_0

#################################################################
class RadonTransform():
    """
    Radon Transform class , bridging ODL and PyTorch.

    The core problem this class solves: the deep learning pipeline operates on
    PyTorch tensors, but ODL — the standard library for the Radon transform —
    works exclusively with NumPy arrays.

    The CT geometry is hardcoded to match the LoDoPaB-CT dataset specification:
        - Image space:     [-0.13, 0.13] x [-0.13, 0.13] m, 1000x1000 pixels
        - Projection angles: 1000 uniformly spaced angles over [0, π]
        - Detector:        513 bins, spanning the diagonal of the image space
        - Geometry:        Parallel beam (2D)
        - Backend:         ASTRA, switched between CPU and CUDA based on device

    Args:
        device (torch.device): Target device for the output tensor.
            Determines both the ASTRA backend (astra_cpu / astra_cuda) and
            where the result tensor is placed. Default: torch.device('cpu').

    Methods:
        radon(Y_coeffs): Applies the Radon transform to a batch of images.

    Example:
        >>> rt = RadonTransform(device=torch.device('cuda'))
        >>> sinogram = rt.radon(gt_batch)  # gt_batch: (B, 1, H, W) on any device
        >>> print(sinogram.shape)
        torch.Size([B, 1, 1000, 513])
    """

    def __init__(self, device=torch.device('cpu')):
        self.device = device
        space = odl.uniform_discr(min_pt=[-0.13, -0.13], max_pt=[+0.13, +0.13], shape=(1000, 1000), dtype='float32')
        angle_partition = odl.uniform_partition(0, np.pi, 1000)  # 0 to pi radians
        detector_length = (0.26**2 + 0.26**2)**(1/2)
        detector_partition = odl.uniform_partition(-detector_length/2, +detector_length/2, 513)  # detector length in meters

        geometry = odl.tomo.Parallel2dGeometry(
            apart=angle_partition,
            dpart=detector_partition
        )
        self.ray_trafo = odl.tomo.RayTransform(space, geometry, impl='astra_cpu' if device == torch.device('cpu') else 'astra_cuda')

    def radon(self, Y_coeffs):
        """
        Applies the Radon transform to a batch of images.

        Converts the input PyTorch tensor to NumPy (required by ODL), applies
        the transform sample-by-sample, then converts the result back to a
        PyTorch tensor on self.device.

        Args:
            Y_coeffs (torch.Tensor): Batch of images, shape (B, 1, H, W).
                Interpolated to (1000, 1000) before transformation.

        Returns:
            torch.Tensor: Sinogram batch of shape (B, 1, 1000, 513) on self.device.
        """
        Y_inter = F.interpolate(Y_coeffs, size=(1000, 1000)).detach().cpu().numpy() # have to move it to cpu before calling numpy
        X_radon = torch.from_numpy(np.stack([self.ray_trafo(Y_inter[i][0]) for i in range(Y_inter.shape[0])])).unsqueeze(1).to(self.device) # apply [i][0], since [i] returns a (1, 1000, 513) shaped object, but the library works with (H, W) objects ´
        return X_radon