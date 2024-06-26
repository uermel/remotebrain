import torch
from typing import List

import cryoet_data_portal as cdp

from membrain_seg.segmentation.networks.unet import SemanticSegmentationUnet
from remotebrain.data.loader import TomogramLoader


def prep_model(ckpt_path: str, device: torch.device) -> SemanticSegmentationUnet:
    # ------ Model prep part of segment ------ #
    # Load the trained PyTorch Lightning model
    model_checkpoint = ckpt_path

    # Initialize the model and load trained weights from checkpoint
    pl_model = SemanticSegmentationUnet.load_from_checkpoint(
        model_checkpoint, map_location=device, strict=False
    )
    pl_model.to(device)

    # Put the model into evaluation mode
    pl_model.eval()
    # ------ Model prep part of segment ------ #
    return pl_model


class DistributedWorker:

    def __init__(self, gpu_id: int, model_ckpt_path: str):
        self.gpu_id = gpu_id
        self.device = torch.device(f"cuda:{gpu_id}")
        self.model = prep_model(model_ckpt_path, self.device)
