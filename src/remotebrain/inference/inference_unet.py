from typing import Tuple

import torch
from membrain_seg.segmentation.networks.unet import SemanticSegmentationUnet
from membrain_seg.segmentation.networks.inference_unet import rescale_tensor


def fourier_cropping_torch(
    data: torch.Tensor, new_shape: tuple, device: torch.device = None
) -> torch.Tensor:
    """
    Fourier cropping adapted for PyTorch and GPU, without smoothing functionality.

    Parameters
    ----------
    data : torch.Tensor
        The input data as a 3D torch tensor on GPU.
    new_shape : tuple
        The target shape for the cropped data as a tuple (x, y, z).
    device : torch.device, optional
        The device to use for the computation. If None, the device is
        automatically set to "cuda" if a GPU is available, otherwise "cpu".

    Returns
    -------
    torch.Tensor
        The resized data as a 3D torch tensor.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    data = data.to(device)

    # Calculate the FFT of the input data
    data_fft = torch.fft.fftn(data)
    data_fft = torch.fft.fftshift(data_fft)

    # Calculate the cropping indices
    original_shape = torch.tensor(data.shape, device=device)
    new_shape = torch.tensor(new_shape, device=device)
    start_indices = (original_shape - new_shape) // 2
    end_indices = start_indices + new_shape

    # Crop the filtered FFT data
    cropped_fft = data_fft[
        start_indices[0] : end_indices[0],
        start_indices[1] : end_indices[1],
        start_indices[2] : end_indices[2],
    ]

    unshifted_cropped_fft = torch.fft.ifftshift(cropped_fft)

    # Calculate the inverse FFT of the cropped data
    resized_data = torch.real(torch.fft.ifftn(unshifted_cropped_fft))

    return resized_data


def fourier_extend_torch(
    data: torch.Tensor, new_shape: tuple, device: torch.device = None
) -> torch.Tensor:
    """
    Fourier padding adapted for PyTorch and GPU, without smoothing functionality.

    Parameters
    ----------
    data : torch.Tensor
        The input data as a 3D torch tensor on GPU.
    new_shape : tuple
        The target shape for the extended data as a tuple (x, y, z).
    device : torch.device, optional
        The device to use for the computation. If None, the device is
        automatically set to "cuda" if a GPU is available, otherwise "cpu".

    Returns
    -------
    torch.Tensor
        The resized data as a 3D torch tensor.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    data = data.to(device)

    data_fft = torch.fft.fftn(data)
    data_fft = torch.fft.fftshift(data_fft)

    padding = [
        (new_dim - old_dim) // 2 for old_dim, new_dim in zip(data.shape, new_shape)
    ]
    padded_fft = torch.nn.functional.pad(
        data_fft,
        pad=[pad for pair in zip(padding, padding) for pad in pair],
        mode="constant",
    )

    unshifted_padded_fft = torch.fft.ifftshift(padded_fft)

    # Calculate the inverse FFT of the cropped data
    resized_data = torch.real(torch.fft.ifftn(unshifted_padded_fft))

    return resized_data


class PreprocessedSemanticSegmentationUnet(SemanticSegmentationUnet):
    """U-Net with rescaling preprocessing.

    This class extends the SemanticSegmentationUnet class by adding
    preprocessing and postprocessing steps. The preprocessing step
    rescales the input to the target shape, and the postprocessing
    step rescales the output to the original shape.
    All of this is done on the GPU if available.
    """

    def __init__(
        self,
        *args,
        rescale_patches: bool = False,  # Should patches be rescaled?
        target_shape: Tuple[int, int, int] = (160, 160, 160),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Store the preprocessing parameters
        self.rescale_patches = rescale_patches
        self.target_shape = target_shape

    def preprocess(self, x):
        """Preprocess the input to the network.

        In this case, we rescale the input to the target shape.
        """
        rescaled_samples = []
        for sample in x:
            sample = sample[0]  # only use the first channel
            if self.rescale_patches:
                if sample.shape[0] > self.target_shape[0]:
                    sample = fourier_cropping_torch(
                        sample, self.target_shape, self.device
                    )
                elif sample.shape[0] < self.target_shape[0]:
                    sample = fourier_extend_torch(
                        sample, self.target_shape, self.device
                    )
            rescaled_samples.append(sample.unsqueeze(0))
        rescaled_samples = torch.stack(rescaled_samples, dim=0)
        return rescaled_samples

    def postprocess(self, x, orig_shape):
        """Postprocess the output of the network.

        In this case, we rescale the output to the original shape.
        """
        rescaled_samples = []
        for sample in x:
            sample = sample[0]  # only use first channel
            if self.rescale_patches:
                sample = rescale_tensor(sample, orig_shape, mode="trilinear")
            rescaled_samples.append(sample.unsqueeze(0))
        rescaled_samples = torch.stack(rescaled_samples, dim=0)
        return rescaled_samples

    def forward(self, x):
        """Forward pass through the network."""
        orig_shape = x.shape[2:]
        preprocessed_x = self.preprocess(x)
        predicted = super().forward(preprocessed_x)
        postprocessed_predicted = self.postprocess(predicted[0], orig_shape)
        # Return list to be compatible with deep supervision outputs
        return [postprocessed_predicted]
