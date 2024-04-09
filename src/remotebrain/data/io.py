import os
import torch
import cryoet_data_portal as cdp
import numpy as np
from mrcfile.mrcinterpreter import MrcInterpreter
from fsspec import AbstractFileSystem
from s3fs import S3FileSystem
from pathlib import Path
from scipy import ndimage
from typing import Union, Optional, List, Dict, Any
from dataclasses import dataclass

import ome_zarr.io
import ome_zarr.writer
import zarr


from membrain_seg.segmentation.dataloading.data_utils import (
    store_tomogram,
    normalize_tomogram,
    Tomogram as MSTomogram,
    convert_dtype,
)

from membrain_seg.segmentation.dataloading.memseg_augmentation import (
    get_prediction_transforms,
)

from membrain_seg.tomo_preprocessing.matching_utils.px_matching_utils import (
    determine_output_shape,
    fourier_cropping,
    fourier_extend,
)

from membrain_seg.segmentation.connected_components import connected_components


@dataclass
class AbstractTomogram:
    s3_mrc_scale0: str
    voxel_spacing: float
    size_x: int
    size_y: int
    size_z: int
    id: int

    @classmethod
    def from_portal(cls, tomogram: cdp.Tomogram):
        return cls(
            s3_mrc_scale0=tomogram.s3_mrc_scale0,
            voxel_spacing=tomogram.voxel_spacing,
            size_x=tomogram.size_x,
            size_y=tomogram.size_y,
            size_z=tomogram.size_z,
            id=tomogram.id,
        )

    @classmethod
    def from_mrc_path(cls, fs: AbstractFileSystem, file: str):
        with fs.open(file, "rb") as f:
            interpreter = MrcInterpreter(iostream=f, permissive=True, header_only=True)
            header = interpreter.header
            voxel_spacing = round(float(interpreter.voxel_size.y), 3)
            size_x = int(header.nx)
            size_y = int(header.ny)
            size_z = int(header.nz)
        return cls(
            s3_mrc_scale0=file,
            voxel_spacing=voxel_spacing,
            size_x=size_x,
            size_y=size_y,
            size_z=size_z,
            id=0,
        )


def match_pixel_size(
    tomogram: AbstractTomogram,
    data: np.ndarray,
    pixel_size_in: float,
    pixel_size_out: float,
    disable_smooth: bool,
) -> np.ndarray:

    smoothing = not disable_smooth

    print(
        "Matching input tomogram",
        tomogram.id,
        "from pixel size",
        pixel_size_in,
        "to pixel size",
        pixel_size_out,
        ".",
    )

    # Calculate the output shape after pixel size matching
    output_shape = determine_output_shape(pixel_size_in, pixel_size_out, data.shape)

    # Perform Fourier-based resizing (cropping or extending) using the determined
    # output shape
    if (pixel_size_in / pixel_size_out) < 1.0:
        resized_data = fourier_cropping(data, output_shape, smoothing)
    elif (pixel_size_in / pixel_size_out) > 1.0:
        resized_data = fourier_extend(data, output_shape, smoothing)
    else:
        resized_data = data

    resized_data = normalize_tomogram(resized_data)
    return resized_data


def fetch_scale_prep(
    fs: S3FileSystem,
    tomogram: AbstractTomogram,
    match_voxel_size: bool = False,
    target_voxel_size: float = 10.0,
) -> tuple[torch.Tensor, np.recarray, AbstractTomogram]:

    # ------ Equivalent to load_tomogram ------ #
    # Fetch from S3
    with fs.open(tomogram.s3_mrc_scale0, "rb") as f:
        interpreter = MrcInterpreter(iostream=f, permissive=True)
        data = interpreter.data.copy()
        data = np.transpose(data, (2, 1, 0))
        header = interpreter.header
        voxel_size = tomogram.voxel_spacing

    # Normalize
    data = normalize_tomogram(data)
    # ------ Equivalent to load_tomogram ------ #

    # ------ Equivalent to match_pixel_size ------ #
    # Match pixel size
    if match_voxel_size:
        if voxel_size != target_voxel_size:
            data = match_pixel_size(
                tomogram, data, voxel_size, target_voxel_size, False
            )
    # ------ Equivalent to match_pixel_size ------ #

    # ------ Data loading part of segment ------ #
    # Prepare for segmentation
    transforms = get_prediction_transforms()

    new_data = np.expand_dims(data, 0)
    new_data = transforms(new_data)
    new_data = new_data.unsqueeze(0)  # Add batch dimension
    new_data = new_data.to(torch.float32)
    # ------ Data loading part of segment ------ #

    return new_data, header, tomogram


def ome_zarr_axes() -> List[Dict[str, str]]:
    return [
        {
            "name": "z",
            "type": "space",
            "unit": "angstrom",
        },
        {
            "name": "y",
            "type": "space",
            "unit": "angstrom",
        },
        {
            "name": "x",
            "type": "space",
            "unit": "angstrom",
        },
    ]


def ome_zarr_transforms(voxel_size: float) -> List[Dict[str, Any]]:
    return [{"scale": [voxel_size, voxel_size, voxel_size], "type": "scale"}]


def array_info(data):
    print(
        f"Shape: {data.shape} | Min: {np.min(data)} | Max: {np.max(data)} | Mean: {np.mean(data)} | STD: {np.std(data)}\n"
    )


def write_multiscale_zarr_3D(
    filename: str,
    tomogram: Union[np.ndarray, List[np.ndarray]],
    voxel_size: Optional[Union[float, List[float]]] = None,
    fs: Optional[S3FileSystem] = None,
    reorder_axis: bool = True,
) -> List:

    if fs:
        loc = zarr.storage.FSStore(
            filename, key_separator="/", mode="w", dimension_separator="/", fs=fs
        )
    else:
        loc = filename
        os.makedirs(filename, exist_ok=True)

    root_group = zarr.group(loc, overwrite=True)

    pyramid = []
    axes = ome_zarr_axes()
    scales = []
    if isinstance(tomogram, list):
        if not voxel_size:
            voxel_size = [idx * 2 for idx in range(1, len(tomogram) + 1)]

        if len(voxel_size) != len(tomogram):
            raise ValueError("Length of voxel_size must match length of tomogram")

        for data, vs in zip(tomogram, voxel_size):
            if reorder_axis:
                pyramid.append(np.transpose(data, (2, 1, 0)))
            else:
                pyramid.append(data)

            scales.append(ome_zarr_transforms(vs))
    else:
        if not voxel_size:
            voxel_size = 1.0

        if reorder_axis:
            pyramid.append(np.transpose(tomogram, (2, 1, 0)))
        else:
            pyramid.append(tomogram)

        scales.append(ome_zarr_transforms(voxel_size))

    # for p in pyramid:
    #     array_info(p)

    return ome_zarr.writer.write_multiscale(
        pyramid,
        group=root_group,
        axes=axes,
        coordinate_transformations=scales,
        storage_options=dict(chunks=(256, 256, 256), overwrite=True),
        compute=True,
    )


def portal_s3_to_loc(
    s3_uri: str,
    temp_out: str,
    s3_out: str,
) -> tuple[str, str, str]:

    location = "/".join(Path(s3_uri).parts[2:-1])
    name = Path(s3_uri).parts[-1]

    s3_out_loc = f"s3://{s3_out.rstrip('/')}/{location}"
    temp_out_loc = f"{temp_out.rstrip('/')}/{location}"

    return temp_out_loc, s3_out_loc, name


def post_scale_put(
    fs: S3FileSystem,
    tomogram: AbstractTomogram,
    run_tag: str,
    network_output: torch.Tensor,
    temp_folder: str,
    out_folder: str,
    ckpt_token: str,
    voxel_size: float,
    voxel_size_probs: float,
    store_probabilities: bool = False,
    store_connected_components: bool = False,
    connected_component_thres: int = None,
    mrc_header: np.recarray = None,
    segmentation_threshold: float = 0.0,
    write_zarr: bool = True,
    write_mrc: bool = True,
    write_npy: bool = False,
    determine_dtype: bool = True,
) -> None:

    predictions = network_output[0]

    # Output locations
    temp_out_loc, s3_out_loc, name = portal_s3_to_loc(
        tomogram.s3_mrc_scale0, temp_folder, out_folder
    )
    name = name.rstrip(".mrc")
    os.makedirs(temp_out_loc, exist_ok=True)

    # Store the probabilities
    if store_probabilities:
        predictions_np = predictions.squeeze(0).squeeze(0).cpu().numpy()

        if write_npy:
            pto = f"{temp_out_loc}/{name}_{ckpt_token}_probabilities_{run_tag}.npy"
            pso = f"{s3_out_loc}/{name}_{ckpt_token}_probabilities_{run_tag}.npy"

            np.save(pto, predictions_np)
            fs.put(pto, pso)
            os.remove(pto)

        if write_zarr:
            pso = f"{s3_out_loc}/{name}_{ckpt_token}_probabilities_{run_tag}.zarr"
            write_multiscale_zarr_3D(
                pso,
                convert_dtype(predictions_np) if determine_dtype else predictions_np,
                voxel_size_probs,
                fs=fs,
            )

    # Threshold
    predictions_np_thres = (
        predictions.squeeze(0).squeeze(0).cpu().numpy() > segmentation_threshold
    )

    # Rescale
    output_shape = (tomogram.size_x, tomogram.size_y, tomogram.size_z)
    rescale_factors = [
        target_dim / original_dim
        for target_dim, original_dim in zip(output_shape, predictions_np_thres.shape)
    ]
    resized_data = ndimage.zoom(
        predictions_np_thres, rescale_factors, order=0, prefilter=False
    )

    # Store
    if write_mrc:
        sto = f"{temp_out_loc}/{name}_{ckpt_token}_segmented_{run_tag}.mrc"
        sso = f"{s3_out_loc}/{name}_{ckpt_token}_segmented_{run_tag}.mrc"

        tomo = MSTomogram(resized_data, mrc_header, tomogram.voxel_spacing)
        store_tomogram(sto, tomo)
        fs.put(sto, sso)
        os.remove(sto)

    if write_zarr:
        sso = f"{s3_out_loc}/{name}_{ckpt_token}_segmented_{run_tag}.zarr"
        write_multiscale_zarr_3D(
            sso,
            convert_dtype(resized_data) if determine_dtype else resized_data,
            voxel_size,
            fs=fs,
        )

    # Connected components
    if store_connected_components:

        resized_data = connected_components(
            resized_data, size_thres=connected_component_thres
        )

        if write_mrc:
            ccto = f"{temp_out_loc}/{name}_{ckpt_token}_conncomp_{run_tag}.mrc"
            ccso = f"{s3_out_loc}/{name}_{ckpt_token}_conncomp_{run_tag}.mrc"

            out_tomo = MSTomogram(
                data=resized_data, header=mrc_header, voxel_size=voxel_size
            )
            store_tomogram(ccto, out_tomo)
            fs.put(ccto, ccso)
            os.remove(ccto)

        if write_zarr:
            ccso = f"{s3_out_loc}/{name}_{ckpt_token}_conncomp_{run_tag}.zarr"
            write_multiscale_zarr_3D(
                ccso,
                convert_dtype(resized_data) if determine_dtype else resized_data,
                voxel_size,
                fs=fs,
            )
