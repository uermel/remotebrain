# membrain-seg for use with data from AWS S3

Run [membrain-seg](https://github.com/teamtomo/membrain-seg) on clusters with data from AWS S3. Possible input data are lists of either:

* cryoET data portal dataset IDs
* cryoET data portal tomogram IDs
* S3 file paths to mrc files

## Sample CLI commands


### File list input

**Example File list**
```
s3://BUCKET/Users/first.last/sample/file_1.mrc
s3://BUCKET/Users/first.last/sample/file_2.mrc
s3://BUCKET/Users/first.last/sample/file_3.mrc
```

**Command Line Interface**

```bash
remotebrain run --input-profile $YOUR_AWS_PROFILE$ \
    --input-list s3://$BUCKET$/$PATH$ \
    --input-type files \
    --output-bucket s3://$BUCKET$/User_outputs \
    --output-profile $YOUR_AWS_PROFILE$ \
    --model-ckpt-path s3://$BUCKET$/checkpoints/MemBrain_seg_v10_alpha.ckpt \
    --num-gpu 8 \
    --worker-per-gpu 2
```
