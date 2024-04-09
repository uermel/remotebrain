import torch
from monai.inferers import SlidingWindowInferer
from membrain_seg.segmentation.networks.unet import SemanticSegmentationUnet


from membrain_seg.segmentation.dataloading.memseg_augmentation import (
    get_mirrored_img,
)


def infer(
    sw_roi_size: int,
    test_time_augmentation: bool,
    new_data: torch.Tensor,
    pl_model: SemanticSegmentationUnet,
    device: torch.device,
) -> torch.Tensor:

    # Perform sliding window inference on the new data
    if sw_roi_size % 32 != 0:
        raise OSError("Sliding window size must be multiple of 32°!")
    roi_size = (sw_roi_size, sw_roi_size, sw_roi_size)
    sw_batch_size = 1
    inferer = SlidingWindowInferer(
        roi_size,
        sw_batch_size,
        overlap=0.5,
        progress=True,
        mode="gaussian",
        device=device,
        cache_roi_weight_map=True,
    )

    # Perform test time augmentation (8-fold mirroring)
    gpu_data = new_data.to(device)
    predictions = torch.zeros_like(gpu_data)
    print("Performing 8-fold test-time augmentation.")
    for m in range(8 if test_time_augmentation else 1):
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                predictions += (
                    get_mirrored_img(
                        inferer(
                            get_mirrored_img(new_data.clone(), m).to(device), pl_model
                        )[0],
                        m,
                    )
                    # .detach()
                    # .cpu()
                )
    if test_time_augmentation:
        predictions /= 8.0

    return predictions.cpu()
