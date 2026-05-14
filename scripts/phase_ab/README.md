# Phase A/B Execution Scripts

This directory contains scripts for:
- creating the isolated conda environment for vanilla nnUNet + centerline feature generation,
- running resource prechecks (gpu monitor),
- building nnUNet datasets directly from annotation directory data,
- submitting plan/preprocess jobs,
- submitting 3-fold Phase A baseline jobs,
- submitting feature-generation jobs,
- submitting Phase B fusion matrix jobs.

## 1) One-time environment setup

Run:

bash scripts/phase_ab/create_conda_env_sten_nnunet.sh sten-nnunet

## 2) Shared runtime environment

The shared loader is:

scripts/phase_ab/common_env.sh

It exports:
- nnUNet_raw=/export/home3/dazhou/centerline/nnUNet_raw
- nnUNet_preprocessed=/export/home3/dazhou/centerline/nnUNet_preprocessed
- nnUNet_results=/export/home3/dazhou/centerline/nnUNet_results

## 3) GPU precheck

Run:

bash scripts/phase_ab/precheck_gpu.sh logs/precheck

This calls gpu alias if available (fallback to gpu_monitor.py) and stores logs.

## 4) Submit Phase A 3-fold prescreen

Run:

bash scripts/phase_ab/submit_phase_a_3fold.sh DATASET_ID 2d

Before first training, run plan+preprocess:

bash scripts/phase_ab/submit_plan_preprocess.sh DATASET_ID 2d

Optional environment overrides:
- PARTITION (default RTXA6Kq)
- GPUS (default 1)
- CPUS (default 8)
- MEM (default 64G)
- TIME_LIMIT (default 24:00:00)
- CONDA_ENV (default sten-nnunet)

## 5) Submit centerline feature generation

Run:

bash scripts/phase_ab/submit_feature_generation.sh

Optional 4th positional argument: LIMIT.

Default paths now target OR masks and partial centerline annotations:
- mask dir: /export/home3/dazhou/centerline/annotation/labelsTr_OR
- annotation json: /export/home3/dazhou/centerline/annotation/annotations_OR.json
- output dir: /export/home3/dazhou/centerline/centerline/output_OR

Only annotated cases are processed by centerline_pipeline.

## 6) Build nnUNet datasets from annotation directory

Phase A (image only, full set):

python scripts/phase_ab/build_nnunet_dataset_from_png.py \
  --dataset-id 901 \
  --dataset-name Vessel_A_full \
  --images-dir /export/home3/dazhou/centerline/annotation/imagesTr \
  --labels-dir /export/home3/dazhou/centerline/annotation/labelsTr \
  --mode image_only \
  --overwrite

Phase A/B fair comparison on annotated subset only:

python scripts/phase_ab/build_nnunet_dataset_from_png.py \
  --dataset-id 902 \
  --dataset-name Vessel_A_annotated57 \
  --images-dir /export/home3/dazhou/centerline/annotation/imagesTr \
  --labels-dir /export/home3/dazhou/centerline/annotation/labelsTr_OR \
  --case-list-json /export/home3/dazhou/centerline/annotation/annotations_OR.json \
  --mode image_only \
  --overwrite

Phase B (distance+radius, annotated subset):

python scripts/phase_ab/build_nnunet_dataset_from_png.py \
  --dataset-id 903 \
  --dataset-name Vessel_B_dist_radius_annotated57 \
  --images-dir /export/home3/dazhou/centerline/annotation/imagesTr \
  --labels-dir /export/home3/dazhou/centerline/annotation/labelsTr_OR \
  --case-list-json /export/home3/dazhou/centerline/annotation/annotations_OR.json \
  --features-root /export/home3/dazhou/centerline/centerline/output_OR \
  --mode image_dist_radius \
  --overwrite

## 7) Submit Phase B matrix (B1/B2/B3)

Run:

bash scripts/phase_ab/submit_phase_b_matrix.sh DATASET_DIST DATASET_RADIUS DATASET_DIST_RADIUS 2d

If feature generation must finish first, set:

FEATURE_DEP_JOB_ID=<job_id> bash scripts/phase_ab/submit_phase_b_matrix.sh ...

## Notes

- A and B scripts call precheck_gpu.sh before training starts.
- If gpu monitor fails or Slurm submission fails, stop and discuss before continuing.
- Logs are written under logs/slurm, logs/phase_a, logs/phase_b, logs/feature_generation.
