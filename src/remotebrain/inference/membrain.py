import torch
from monai.inferers import SlidingWindowInferer

# from membrain_seg.segmentation.networks.unet import SemanticSegmentationUnet
from remotebrain.inference.inference_unet import PreprocessedSemanticSegmentationUnet

from membrain_seg.tomo_preprocessing.matching_utils.px_matching_utils import (
    determine_output_shape,
)

from membrain_seg.segmentation.dataloading.memseg_augmentation import (
    get_mirrored_img,
)


def infer(
    sw_roi_size: int,
    in_pixel_size: float,
    eval_pixel_size: float,
    rescale_patches: bool,
    test_time_augmentation: bool,
    new_data: torch.Tensor,
    pl_model: PreprocessedSemanticSegmentationUnet,
    device: torch.device,
) -> torch.Tensor:

    # Perform sliding window inference on the new data
    if sw_roi_size % 32 != 0:
        raise OSError("Sliding window size must be multiple of 32°!")

    if rescale_patches:
        # Determine the sliding window size according to the input and output pixel size
        sw_roi_size = determine_output_shape(
            # switch in and out pixel size to get SW shape
            pixel_size_in=eval_pixel_size,
            pixel_size_out=in_pixel_size,
            orig_shape=(sw_roi_size, sw_roi_size, sw_roi_size),
        )
        sw_roi_size = sw_roi_size[0]

    #pl_model.rescale_patches = in_pixel_size != eval_pixel_size

    roi_size = (sw_roi_size, sw_roi_size, sw_roi_size)
    print(roi_size)

    sw_batch_size = 1
    inferer = SlidingWindowInferer(
        roi_size,
        sw_batch_size,
        overlap=0.5,
        progress=True,
        mode="gaussian",
        sw_device=device,
        device=device,
        cache_roi_weight_map=True,
    )

    # Perform test time augmentation (8-fold mirroring)
    gpu_data = new_data.to(device)
    predictions = torch.zeros_like(gpu_data, device=device)

    if test_time_augmentation:
        print(
            "Performing 8-fold test-time augmentation.",
            "I.e. the following bar will run 8 times.",
        )
    for m in range(8 if test_time_augmentation else 1):
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                mirrored_input = get_mirrored_img(gpu_data.clone(), m)
                mirrored_pred = inferer(mirrored_input, pl_model)
                if not isinstance(mirrored_pred, list):
                    mirrored_pred = [mirrored_pred]
                correct_pred = get_mirrored_img(mirrored_pred[0], m)
                predictions += correct_pred
    if test_time_augmentation:
        predictions /= 8.0

    return predictions.cpu()
