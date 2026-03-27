import torch
import torch.nn as nn
import torch.nn.functional as F

#####################################
# Configuration and Precomputations
#####################################

n_angles, n_detectors = 1000, 513
s_range = 0.13
img_size = 512
crop_size = 362
#--------------------------------------------------------------------------------------------#
# Precomputations for geometry and filtering

# Spacing between adjacent detectors
Delta_s = 2 * s_range / (n_detectors - 1)

# Frequency components for FFT (used in filtering)
freq_s = torch.fft.fftfreq(n_detectors, d=Delta_s)  # [513]

#---------------------------------------------------------------------------------------------#



#Filter
#Backprojection




class LearnableFBP(nn.Module):
    """
    Learnable Filtered Backprojection (FBP) pipeline.

    Implements the classical FBP algorithm as a differentiable nn.Module,
    making the filter and post-processing steps trainable end-to-end.

    The forward pass follows the standard FBP steps:
        1. FFT along the detector axis
        2. Frequency-domain filtering (learnable filter * learnable window)
        3. IFFT back to sinogram domain
        4. Backprojection onto the image grid
        5. Optional post-processing (e.g. UNet refinement)

    Args:
        filtering_module (nn.Module):   Applies filter and window in frequency domain.
        backprojection_module (nn.Module): Differentiable backprojection (e.g. Vanilla_Backproj).
        post_processing_module (nn.Module): Applied after backprojection (e.g. UNet, identity).

    Input:
        x (torch.Tensor): Sinogram batch of shape (B, 1, 1000, 513).

    Output:
        torch.Tensor: Reconstructed image batch of shape (B, 1, 362, 362).
    """
    def __init__(self, filtering_module, backprojection_module, post_processing_module):
        super(LearnableFBP, self).__init__()
        self.filtering_module = filtering_module
        self.backprojection_module = backprojection_module
        self.post_processing_module = post_processing_module

    def forward(self, x):
        # Step 1: FFT of Radon
        x = torch.fft.fft(x, dim=3)  # (B x 1 x 1000 x 513)

        # Step 2: Filtering
        x = self.filtering_module(x)  # (B x 1 x 1000 x 513)

        # Step 3: IFFT
        x = torch.fft.ifft(x, dim=3).real  # (B x 1 x 1000 x 513)

        # Step 4: Backprojection
        x = self.backprojection_module(x)  # (B x 1 x 362 x 362)

        # Step 5: Post-Processing
        x = self.post_processing_module(x)  # (B x 1 x 362 x 362)

        return x


class Filtering_Module(nn.Module):
    """
    Frequency-domain filtering step of the FBP pipeline.

    Applies a pointwise multiplication of the FFT-transformed sinogram with
    a filter and a window, both of which can be learnable:

        output = window(x) * filter(x) * x

    Separating filter and window allows independent control over the
    frequency shaping (filter) and the roll-off / apodization (window).

    Args:
        filter_model (nn.Module): Produces a frequency-domain filter (e.g. Ramp_Filter).
        window_model (nn.Module): Produces an apodization window (e.g. LearnableWindow).

    Input:
        x (torch.Tensor): FFT of sinogram, shape (B, 1, 1000, 513), complex.

    Output:
        torch.Tensor: Filtered FFT, same shape as input.
    """
    def __init__(self, filter_model, window_model):
        super(Filtering_Module, self).__init__()
        self.filter_model, self.window_model = filter_model, window_model

    def forward(self, x):
        filtering = self.filter_model(x)
        window = self.window_model(x)

        return window * filtering * x

class Vanilla_Backproj(nn.Module):
    """
    Differentiable backprojection module.

    Implements the backprojection step of FBP using PyTorch's grid_sample,
    making it fully differentiable and GPU-compatible. All projection geometry
    is precomputed in __init__ and stored as buffers (moved with the module
    to the correct device automatically).

    For each projection angle θ, every image pixel (x, y) is mapped to its
    corresponding detector position via the dot product x·cos(θ) + y·sin(θ).
    The sinogram value at that position is then interpolated (bilinear) and
    accumulated across all angles — which is exactly the continuous backprojection
    integral, discretized.

    The geometry is hardcoded to the LoDoPaB-CT dataset:
        - 1000 projection angles over [0, π]
        - 513 detectors over [-0.13√2, +0.13√2] m
        - Output cropped from 512×512 to 362×362 (center crop)
        - Scaled by π / n_angles to approximate the continuous integral

    Args:
        n_angles (int):    Number of projection angles. Default: 1000.
        n_detectors (int): Number of detector bins. Default: 513.
        s_range (float):   Half-width of the detector in meters. Default: 0.13.
        img_size (int):    Intermediate image size before cropping. Default: 512.
        crop_size (int):   Final output image size. Default: 362.

    Input:
        x (torch.Tensor): Filtered sinogram, shape (B, 1, 1000, 513).

    Output:
        torch.Tensor: Reconstructed image, shape (B, 1, 362, 362).
    """
    def __init__(self, n_angles=n_angles, n_detectors=n_detectors, s_range=s_range, img_size=img_size, crop_size=crop_size):
        super(Vanilla_Backproj, self).__init__()

        Delta_s = 2 * s_range / (n_detectors - 1)
        x = torch.linspace(-s_range, +s_range, img_size)
        y = torch.linspace(-s_range, +s_range, img_size)

        # Create meshgrid
        Y, X = torch.meshgrid(x, y, indexing='ij')

        # Uniformly spaced projection angles from 0 to pi
        theta = torch.linspace(0, torch.pi, n_angles + 1)[:-1]  # [1000]

        # Dot products of image points with projection directions (Radon transform geometry)
        dot_prods = X[None, :, :] * torch.cos(theta)[:, None, None] + \
                    Y[None, :, :] * torch.sin(theta)[:, None, None]  # (1000, 512, 512)

        # Map coordinates to detector positions
        interested_s_positions = (dot_prods + s_range) / Delta_s  # (1000, 512, 512)

        # Normalize to [-1, 1] for grid_sample
        grid_norm = (interested_s_positions / (n_detectors - 1)) * 2 - 1  # (1000, 512, 512)

        self.register_buffer('grid_norm', torch.stack([grid_norm, torch.zeros_like(grid_norm)], dim=-1))  # (1000, 512, 512, 2)
        self.crop_size = crop_size

    def forward(self, x):
        # Manual differentiable backprojection
        B = x.shape[0]
        grid_norm = self.grid_norm
        slices = []

        for k in range(n_angles):
            grid_k = grid_norm[k].unsqueeze(0).repeat(B, 1, 1, 1)  # (B, H, W, 2)
            slice_k = x[:, :, k, :].unsqueeze(2)  # (B, 1, 1, W)

            # Bilinear interpolation along projection ray
            interpolated = F.grid_sample(slice_k, grid_k, mode='bilinear', align_corners=True)  # (B, 1, H, W)
            slices.append(interpolated)

        # Sum all projections
        x = torch.stack(slices, dim=1).sum(dim=1)  # (B, 1, H, W)

        # Crop to desired region
        x = self._differentiable_center_crop(x, self.crop_size)  # (B, 1, 362, 362)

        # Scale appropriately
        x *= torch.pi / n_angles

        #  transpose for final orientation
        return torch.transpose(x, -1, -2)

    def _differentiable_center_crop(self, x, crop_size):
        """
        Applies a center crop on a 4D tensor in a differentiable manner.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W)
            crop_size (int): Size of the square crop.

        Returns:
            Tensor: Cropped tensor of shape (B, C, crop_size, crop_size)
        """
        _, _, h, w = x.shape
        top = (h - crop_size) // 2
        left = (w - crop_size) // 2
        return x[:, :, top:top + crop_size, left:left + crop_size]


class Ramp_Filter(nn.Module):
    """
    Classical ramp filter (Ram-Lak) for FBP.

    Returns the absolute value of the frequency axis |ω|, which is the
    theoretically correct filter for exact FBP reconstruction. Registered
    as a buffer so it is automatically moved to the correct device.

    This is the non-learnable baseline filter. It can be combined with a
    learnable window in Filtering_Module to add trainable apodization on top.

    Args:
        freqs (torch.Tensor): Frequency axis, precomputed from detector spacing.
            Default: module-level freq_s (shape [513]).

    Input:
        x: Ignored — the filter is independent of the input.

    Output:
        torch.Tensor: Ramp filter of shape (1, 1, 1, 513), broadcastable over (B, 1, n_angles, n_detectors).
    """
    def __init__(self, freqs=freq_s):
        super().__init__()
        self.register_buffer('ramp', torch.abs(freqs))

    def forward(self, x):
        return self.ramp[None, None, None, :]

#####
class CompleteReconstruct(nn.Module):
    """
    Two-stage reconstruction pipeline: sinogram preprocessing + LearnableFBP.

    Combines a sinogram-domain network with the full FBP reconstruction,
    allowing the model to first clean or enhance the raw sinogram before
    the reconstruction step. Used in experiments where sinogram-domain
    processing is beneficial (e.g. noise reduction prior to backprojection).

    Args:
        learnablefbp (LearnableFBP):  The FBP reconstruction module.
        preprocess_net (nn.Module):   Network applied to the sinogram before FBP
                                      (e.g. UNet operating in sinogram domain).

    Input:
        X (torch.Tensor): Raw sinogram batch, shape (B, 1, 1000, 513).

    Output:
        torch.Tensor: Reconstructed image batch, shape (B, 1, 362, 362).
    """
    def __init__(self, learnablefbp, preprocess_net):
        super(CompleteReconstruct, self).__init__()
        self.learnablefbp = learnablefbp
        self.preprocess_net = preprocess_net

    def forward(self, X):
        X = self.preprocess_net(X)
        X = self.learnablefbp(X)
        return X