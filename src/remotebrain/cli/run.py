from typing import List
import aiobotocore
from s3fs import S3FileSystem
import re
import os
import click
from typing import Union, Literal

from cryoet_data_portal import Client, Tomogram

from remotebrain.distributed.executor import DistributedInference
from remotebrain.data.io import AbstractTomogram


@click.group()
@click.pass_context
def cli(ctx):
    pass


@cli.command()
@click.option(
    "--input-profile",
    type=str,
    default=None,
    help="AWS profile to use for input bucket.",
)
@click.option(
    "--input-list",
    type=str,
    required=True,
    help="""S3 URI to a text file containing a list of cryoET data portal dataset ids, 
            tomogram ids or paths to MRC-files. One per line.""",
)
@click.option(
    "--input-type",
    type=click.Choice(["dataset_ids", "tomogram_ids", "files"]),
    default="dataset_ids",
    help="""Type of input list. 
             - dataset_ids: list of cryoET data portal dataset ids
             - tomogram_ids: list of cryoET data portal tomogram ids
             - files: list of S3 URIs to MRC-files""",
)
@click.option(
    "--output-bucket",
    type=str,
    required=True,
    help="S3 URI to the location on the output bucket.",
)
@click.option(
    "--output-profile",
    type=str,
    default=None,
    help="AWS profile to use for output bucket.",
)
@click.option(
    "--model-ckpt-path",
    type=str,
    required=True,
    help="S3 URI to the model checkpoint (assumed to exist on output bucket).",
)
@click.option("--num-gpu", type=int, default=8, help="Number of GPUs to use.")
@click.option(
    "--worker-per-gpu", type=int, default=1, help="Number of workers per GPU."
)
@click.option("--rescale-patches", type=bool, default=True, help="Rescale patches.")
@click.option(
    "--eval-pixel-size", type=float, default=10.0, help="Evaluation pixel size."
)
@click.pass_context
def run(
    ctx,
    input_profile: str,
    input_list: str,
    input_type: Union[
        Literal["dataset_ids"], Literal["tomogram_ids"], Literal["files"]
    ],
    output_bucket: str,
    output_profile: str,
    model_ckpt_path: str,
    #    filter_run: str,
    num_gpu: int,
    worker_per_gpu: int,
    rescale_patches: bool,
    eval_pixel_size: float,
):

    # Input FS
    if input_profile:
        input_session = aiobotocore.session.AioSession(profile=input_profile)
        input_fs = S3FileSystem(session=input_session)
    else:
        input_fs = S3FileSystem()

    # Read input list
    with input_fs.open(input_list, "r") as f:
        input_objects = f.readlines()

    tomos = []

    # Get tomolist for input dataset id list
    if input_type == "dataset_ids":
        portal_client = Client()
        tomos = Tomogram.find(
            portal_client,
            [Tomogram.tomogram_voxel_spacing.run.dataset_id._in(input_objects)],
        )
        tomos = [AbstractTomogram.from_portal(t) for t in tomos]
        print(f"Found {len(tomos)} tomos.", flush=True)

    # Get tomolist for input tomogram id list
    if input_type == "tomogram_ids":
        portal_client = Client()
        tomos = [Tomogram.get_by_id(portal_client, int(t)) for t in input_objects]
        tomos = [AbstractTomogram.from_portal(t) for t in tomos]
        print(f"Found {len(tomos)} tomos.", flush=True)

    # Get tomolist for input file list
    if input_type == "files":
        for t in input_objects:
            try:
                tomos.append(AbstractTomogram.from_mrc_path(input_fs, t.strip()))
            except Exception as e:
                print(f"Error processing {t.strip()}: {e}", flush=True)
                continue
        print(f"Found {len(tomos)} tomos.", flush=True)

    # Setup output fs and fetch model checkpoint
    if output_profile:
        output_session = aiobotocore.session.AioSession(profile=output_profile)
        output_fs = S3FileSystem(session=output_session)
    else:
        output_fs = S3FileSystem()

    # Fetch model checkpoint
    model_base = os.path.basename(model_ckpt_path)
    local_model = f"/tmp/{model_base}"
    output_fs.get(model_ckpt_path, local_model)
    print(f"Model fetched to {local_model}.", flush=True)

    # Setup distributed inference
    # num_gpu = 8  # int(os.getenv("WORLD_SIZE"))
    runner = DistributedInference(
        gpu_ids=list(range(num_gpu)),
        tomograms=tomos,
        model_ckpt_path=local_model,
        input_fs=input_fs,
        output_fs=output_fs,
        output_bucket=output_bucket,
        sw_roi_size=160,
        test_time_augmentation=True,
        worker_per_gpu=worker_per_gpu,
        eval_pixel_size=eval_pixel_size,
        rescale_patches=rescale_patches,
    )

    print("Runner setup.", flush=True)

    # Run inference
    runner.execute()

    runner.finalize()


if __name__ == "__main__":
    cli()
