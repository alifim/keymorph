import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size,
                                                                  window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def _ssim_3D(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv3d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv3d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


class SSIM3D(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window_3D(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window_3D(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim_3D(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def ssim3D(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim_3D(img1, img2, window, window_size, channel, size_average)

#############################
def _ssim_3D_with_cs(img1, img2, window, window_size, channel):
    """
    Compute *full 3D* SSIM and Contrast Sensitivity (CS) maps (shape: N,C,D,H,W).
    We'll use these maps to build Multi-Scale SSIM.
    """
    # Local means (mu1, mu2)
    mu1 = F.conv3d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv3d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # Local variances and covariance
    sigma1_sq = F.conv3d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    # Constants (same style as your code: 0.01^2 and 0.03^2)
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # SSIM map
    ssim_map = ((2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    # Contrast Sensitivity (CS) map
    cs_map = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)

    return ssim_map, cs_map


##########################################
# MULTI-SCALE SSIM FOR 3D
##########################################

class MS_SSIM3D(torch.nn.Module):
    """
    Multi-scale SSIM (MS-SSIM) for 3D volumes, using a 3D Gaussian window.
    Follows standard MS-SSIM approach:
      1) Compute SSIM & CS at each scale
      2) For scale i < n_scales: use CS
      3) For scale i = n_scales-1: use SSIM
      4) Combine via weighted product
    """

    def __init__(
            self,
            window_size=11,
            size_average=True,
            n_scales=5,
            scale_weights=None
    ):
        """
        Args:
            window_size (int): Size of the 3D window
            size_average (bool): Whether to average the final result over the batch
            n_scales (int): Number of scales in MS-SSIM
            scale_weights (list or None): Weights for each scale. If None, use defaults.
        """
        super(MS_SSIM3D, self).__init__()

        self.window_size = window_size
        self.size_average = size_average
        self.n_scales = n_scales

        # Default MS-SSIM scale weights (Wang et al. typical values)
        if scale_weights is None:
            scale_weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        if len(scale_weights) != n_scales:
            raise ValueError(f"Expected {n_scales} scale weights, got {len(scale_weights)}")

        self.scale_weights = torch.FloatTensor(scale_weights)

        # We'll create the window for channel=1 initially;
        # if the input has more channels, we'll recreate it on the fly
        self.channel = 1
        self.window = create_window_3D(window_size, self.channel)

    def forward(self, img1, img2):
        """
        Compute MS-SSIM over a 5D input (N, C, D, H, W).
        Returns:
            A single scalar if size_average=True, else (N, C) shape with per-batch results.
        """
        if img1.shape != img2.shape:
            raise ValueError(f"Input shapes must match! Got {img1.shape} vs {img2.shape}")
        if len(img1.shape) != 5:
            raise ValueError(f"Expected 5D inputs (N,C,D,H,W), got {img1.shape}")

        # Check if we need to rebuild the window for the correct channel & device
        n, channel, d, h, w = img1.shape
        if (channel != self.channel or
                self.window.device != img1.device or
                self.window.dtype != img1.dtype):
            # Create a 3D window for this channel
            window = create_window_3D(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        else:
            window = self.window

        # Move scale weights to device
        self.scale_weights = self.scale_weights.to(img1.device)

        # We'll store the mean CS (first scales) or SSIM (final scale)
        # in a list, each entry has shape (N, C).
        mcs = []

        current_img1 = img1
        current_img2 = img2

        for scale_idx in range(self.n_scales):
            # ssim_map, cs_map are shape (N, C, D, H, W)
            ssim_map, cs_map = _ssim_3D_with_cs(current_img1, current_img2,
                                                window, self.window_size, channel)

            # Average each map across spatial dimensions -> shape (N, C)
            ssim_val = ssim_map.mean(dim=[2, 3, 4])  # average over (D,H,W)
            cs_val = cs_map.mean(dim=[2, 3, 4])

            if scale_idx < self.n_scales - 1:
                # For scales except the last, store CS
                mcs.append(cs_val)
                # Downsample for next scale
                current_img1 = F.avg_pool3d(current_img1, kernel_size=2, stride=2)
                current_img2 = F.avg_pool3d(current_img2, kernel_size=2, stride=2)
            else:
                # For the last scale, store SSIM
                mcs.append(ssim_val)

        # Stack mcs: shape = (n_scales, N, C)
        mcs = torch.stack(mcs, dim=0)
        # Ensure non-negative (avoid numerical issues)
        mcs = torch.relu(mcs)

        # We want to do a product of each scale ^ weight
        # scale_weights shape: (n_scales,) -> reshape to (n_scales, 1, 1)
        weights = self.scale_weights.view(self.n_scales, 1, 1)

        # Weighted product across the scale dimension (dim=0)
        # After exponentiation, shape remains (n_scales, N, C), then product -> (N, C)
        msssim_per_batch = torch.prod(mcs ** weights, dim=0)  # shape: (N, C)

        if self.size_average:
            # Return a single scalar averaged over (N, C)
            return msssim_per_batch.mean()
        else:
            # Return the per-sample, per-channel values
            return msssim_per_batch