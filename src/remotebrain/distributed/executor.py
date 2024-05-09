import os
import torch
import queue
import concurrent.futures
from typing import List
import numpy as np
from fsspec import AbstractFileSystem
from s3fs import S3FileSystem
import time

import cryoet_data_portal as cdp

from remotebrain.distributed.worker import DistributedWorker
from remotebrain.data.loader import TomogramLoader
from remotebrain.inference.membrain import infer
from remotebrain.data.io import AbstractTomogram, post_scale_put


class DistributedInference:
    def __init__(
        self,
        gpu_ids: list[int],
        tomograms: List[AbstractTomogram],
        model_ckpt_path: str,
        input_fs: AbstractFileSystem,
        output_fs: AbstractFileSystem,
        output_bucket: str,
        sw_roi_size: int = 160,
        test_time_augmentation: bool = True,
        worker_per_gpu: int = 1,
        eval_pixel_size: float = 10.0,
        rescale_patches: bool = True,
    ):
        self.num_workers = len(gpu_ids) * worker_per_gpu

        # Buffered fetching of tomograms from the portal
        self.loader = TomogramLoader(input_fs, tomograms, self.num_workers * 2)

        # Set up the GPU queue, models loaded on each GPU
        self.gpu_queue = queue.Queue()
        for i in range(worker_per_gpu):
            for idx, gpu_id in enumerate(gpu_ids):
                self.gpu_queue.put(
                    DistributedWorker(
                        gpu_id,
                        model_ckpt_path=model_ckpt_path,
                        target_shape=(sw_roi_size, sw_roi_size, sw_roi_size),
                    )
                )

        # other setup
        self.input_fs = input_fs
        self.output_fs = output_fs
        self.output_bucket = output_bucket
        self.ckpt_token = os.path.basename(model_ckpt_path).split("-val_loss")[0]
        self.sw_roi_size = sw_roi_size
        self.test_time_augmentation = test_time_augmentation
        self.eval_pixel_size = eval_pixel_size
        self.rescale_patches = rescale_patches

    def run(self) -> str:
        worker = self.gpu_queue.get()

        data, header, tomo = next(self.loader)

        # if not data:
        #     self.gpu_queue.put(worker)
        #     return f"No data found for {tomo.s3_mrc_scale0}"

        print(f"Processing {tomo.s3_mrc_scale0} on GPU {worker.gpu_id}\n", flush=True)

        msg = f"Successfully processed {tomo.s3_mrc_scale0} on GPU {worker.gpu_id}"

        try:
            data = data.to(worker.device)

            predictions = infer(
                sw_roi_size=self.sw_roi_size,
                in_pixel_size=tomo.voxel_spacing,
                eval_pixel_size=self.eval_pixel_size,
                rescale_patches=self.rescale_patches,
                test_time_augmentation=self.test_time_augmentation,
                new_data=data,
                pl_model=worker.model,
                device=worker.device,
            )

            RUN_NAME = os.getenv("RUN_NAME")
            post_scale_put(
                self.output_fs,
                tomo,
                RUN_NAME,
                predictions,
                "/tmp/",
                self.output_bucket,
                self.ckpt_token,
                voxel_size=tomo.voxel_spacing,
                voxel_size_probs=tomo.voxel_spacing,
                store_probabilities=True,
                store_connected_components=False,
                mrc_header=header,
                write_zarr=True,
                write_mrc=False,
                write_npy=False,
            )
        except Exception as e:
            msg = f"Error processing {tomo.s3_mrc_scale0} on GPU {worker.gpu_id}\n"
            msg += str(e)

        self.gpu_queue.put(worker)

        return msg

    def execute(self):
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_workers
        ) as executor:
            # Submit jobs
            futures = [executor.submit(self.run) for tomo in range(len(self.loader))]

            # Collect jobs
            for future in concurrent.futures.as_completed(futures):
                print(future.result(), flush=True)

    def finalize(self):
        self.loader.end()
