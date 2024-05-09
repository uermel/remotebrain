import torch
from typing import List, Tuple

import cryoet_data_portal as cdp

from remotebrain.inference.inference_unet import (
    PreprocessedSemanticSegmentationUnet,
)
from remotebrain.data.loader import TomogramLoader


def prep_model(
    ckpt_path: str,
    device: torch.device,
    target_shape: Tuple[int, int, int],
    rescale_patches: bool,
) -> PreprocessedSemanticSegmentationUnet:
    # ------ Model prep part of segment ------ #
    # Load the trained PyTorch Lightning model
    model_checkpoint = ckpt_path

    # Initialize the model and load trained weights from checkpoint
    pl_model = PreprocessedSemanticSegmentationUnet.load_from_checkpoint(
        model_checkpoint, map_location=device, strict=False
    )
    pl_model.to(device)

    # Put the model into evaluation mode
    pl_model.eval()
    pl_model.target_shape = target_shape
    pl_model.rescale_patches = rescale_patches
    pl_model = torch.compile(pl_model, mode="reduce-overhead")

    # ------ Model prep part of segment ------ #
    return pl_model


class DistributedWorker:

    def __init__(
        self,
        gpu_id: int,
        model_ckpt_path: str,
        target_shape: Tuple[int, int, int],
        rescale_patches: bool,
    ):
        self.gpu_id = gpu_id
        self.device = torch.device(f"cuda:{gpu_id}")
        self.model = prep_model(
            model_ckpt_path, self.device, target_shape, rescale_patches
        )
