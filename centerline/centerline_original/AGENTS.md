# Agent Playbook

## Mission Snapshot
Port the MATLAB-based vessel stenosis workflow to Python and adapt it for the 2-D PNG masks under `annotation/labelsTr/` using the endpoints recorded in `annotation/annotations.json`. Generate numeric centerlines and rendered overlays for each annotated mask while preserving the motion-balloon logic described in `algorithm.md` and `血管狭窄分析程序说明ReadMe.docx`.

## Reference Mining
1. Parse the DOCX (XML if needed) and `algorithm.md` to restate every preprocessing, distance-transform, and path-search step.  
2. Walk through `AutomaticAnalysis.m`, `GeneMask.m`, `GenePath.m`, `ShortestPath_inuse.m`, and `GeneStenosis.m` to understand data contracts (intensity scaling, neighborhood rules, output formats).  
3. Document assumptions in a scratchpad before coding to avoid silent behavior drift.

## Environment & Tooling
- Target environment: conda env named `centerline` rooted in `/root/vessel/conda_envs/centerline`; set `CONDA_PKGS_DIRS` and `CONDA_OVERRIDE_CUDA=""` when invoking conda to bypass semaphore and GPU detection issues.  
- If the sandbox blocks package downloads, capture the failure details and request a pre-populated package cache; avoid falling back to ad-hoc system installs without approval.  
- Baseline Python stack: `numpy`, `scipy`, `scikit-image`, `networkx`, `opencv-python` (for overlays), and `matplotlib`/`Pillow` for visualization. Prefer conda packages; if unavailable, coordinate on alternatives.

## Python Conversion Roadmap
1. Build a `centerline_pipeline/` package with modules for IO (`io.py`), preprocessing (`preprocess.py`), distance/cost volume (`distance.py`), shortest-path search (`path.py`), and stenosis metrics (`metrics.py`).  
2. Recreate the MATLAB thresholding and endpoint snapping behavior; note MATLAB’s 1-indexing vs. Python’s 0-indexing, and convert coordinates carefully.  
3. Implement distance transforms via `scipy.ndimage.distance_transform_edt`, exponentiate costs with the same K factor (`K=4`), and solve paths using weighted adjacency graphs (8-neighborhood for 2-D).  
4. Sample radii along the path, compute reference diameters, and export stenosis summaries plus overlay images to `centerline/output/<case_id>/`.

## Validation & Deliverables
- Develop a CLI entry point `python -m centerline_pipeline.process --mask-dir annotation/labelsTr --annotations annotation/annotations.json --out centerline/output`.  
- For each case, save: `centerline.json` (path coordinates & diameters), `stenosis.csv`, and an overlay PNG combining mask + centerline.  
- Add unit tests (e.g., `pytest`) covering distance transform invariants and minimal path correctness on toy masks; include at least one regression test against a MATLAB output once available.  
- Keep a migration log noting parity gaps, data quirks, and pending MATLAB features (e.g., 3-D WHT) required for future expansion.

## Communication Protocol
Report blockers immediately (environment creation, missing packages, data inconsistencies). When assumptions are made about image orientation or pixel spacing, confirm with sample visualizations and share findings before broad batch processing.*** End Patch*** End Patch
