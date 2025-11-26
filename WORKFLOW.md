# Centerline-to-StenUNet Workflow Notes

## AGENTS.md locations and scope
- `/workspace/centerline/AGENTS.md` governs the entire repository unless a deeper `AGENTS.md` overrides it.
- An additional `AGENTS.md` under `centerline/centerline_original/` applies only within that legacy MATLAB subtree.

## Where this document lives
- Path: `/workspace/centerline/WORKFLOW.md` inside the local working copy of the repository.
- Changes are applied locally and committed to the repo on this machine; nothing is pushed to a remote GitHub repository unless an explicit push command is executed.

## Four-step flow
1. **Centerline generation**  \
   Run the Python centerline pipeline against the annotated vessel masks to populate `centerline/output/<case_id>/`:
   ```bash
   conda activate centerline
   python -m centerline_pipeline --mask-dir annotation/labelsTr \
       --annotations annotation/annotations.json \
       --out centerline/output
   ```
   The outputs should include centerline JSON, stenosis metrics, and pseudo-color overlays for each case.

2. **Feature-map (pseudo-color) preparation**  \
   Convert or collect the pseudo-color feature maps produced in `centerline/output` and sync them into the StenUNet training tree. The expected data copy looks like:
   ```bash
   # Feature maps (e.g., HSV pseudo-color targets)
   rsync -av centerline/output/pseudo_color/ StenUNet/pseudo_color/train_data/targets_train/

   # Corresponding binary masks if they are regenerated alongside the features
   rsync -av centerline/output/masks/ StenUNet/pseudo_color/train_data/masks_train/
   ```
   Adjust the source subfolders to match the actual exporters (per-case folders are acceptable as long as the filenames align).

3. **Model training**  \
   Launch pseudo-color regression or downstream StenUNet training from within `StenUNet/` after the feature maps are in place:
   ```bash
   cd StenUNet
   python -m pseudo_color.train_scalar_field \
       --train-mask-dir pseudo_color/train_data/masks_train \
       --train-target-dir pseudo_color/train_data/targets_train \
       --target-mode hsv \
       --include-distance --include-coords \
       --output-dir ./pseudo_color/runs
   ```
   Document the new `target_mode=HSV` expectation alongside the dataset copy step above; the training CLI currently validates only `scalar` and `rgb`, so either extend it to accept `hsv` or pre-convert the targets to RGB before running.

4. **Model inference**  \
   After training, perform stenosis inference using the StenUNet entrypoint (weights from the run directory):
   ```bash
   python inference.py -chk pseudo_color/runs/best_model.pt -i dataset_test/raw
   ```
   Maintain consistent preprocessing between training and inference.
