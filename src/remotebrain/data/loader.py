from cryoet_data_portal import Client, Tomogram
from typing import Optional
import re
from s3fs import S3FileSystem
from fsspec import AbstractFileSystem
from mrcfile.mrcinterpreter import MrcInterpreter
import numpy as np
import torch

# import cryoet_data_portal as cdp
from multiprocessing.pool import ThreadPool


from remotebrain.data.io import AbstractTomogram, fetch_scale_prep


class TomogramLoader:
    def __init__(
        self,
        fs: AbstractFileSystem,
        tomos: list[AbstractTomogram],
        buffer_size: int = 8,
    ):

        self.tomos = tomos  # tomos_filtered
        self._len = len(tomos)

        self.fs = fs

        launch_size = min(buffer_size, len(self.tomos))
        self._pool = ThreadPool(launch_size)
        self._buffer = []

        for i in range(launch_size):
            tom = self.tomos.pop(0)
            self._buffer.append(self._pool.apply_async(self.load, (tom,)))

    def load(
        self, tomogram: AbstractTomogram
    ) -> tuple[torch.Tensor, np.recarray, AbstractTomogram]:
        print(f"Loading {tomogram.s3_mrc_scale0}\n", flush=True)

        # ret = (None, None, None)
        # try:
        ret = fetch_scale_prep(self.fs, tomogram)
        # except Exception as e:
        #    print(f"Error loading {tomogram.s3_mrc_scale0}: {e}\n")

        return ret

    def __len__(self) -> int:
        return self._len

    def __iter__(self):
        return self

    def __next__(self) -> tuple[torch.Tensor, np.recarray, AbstractTomogram]:
        if len(self._buffer) > 0:
            ret = self._buffer.pop(0).get()
        else:
            raise StopIteration

        if len(self.tomos) > 0:
            tom = self.tomos.pop(0)
            self._buffer.append(self._pool.apply_async(self.load, (tom,)))

        return ret

    def end(self):
        self._pool.close()
        self._pool.join()
        print("TomogramLoader ended gracefully.", flush=True)
