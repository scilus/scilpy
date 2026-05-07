# Implementation Plan - Directional Orientation Management

## Phase 1: Core Implementation (StatefulImage) [checkpoint: 87f3afb]
- [x] Task: Implement `rotate_sh` utility in `scilpy.reconst.sh` (or verify existing one) to handle coefficient rotation.
- [x] Task: Add `to_voxel_direction()` to `StatefulImage`.
    - [x] Write Tests: Verify RAS-to-Voxel rotation for a known 90-degree rotation.
    - [x] Implement: Use rotation component of affine to rotate directions/coefficients.
- [x] Task: Add `to_world_direction()` to `StatefulImage`.
    - [x] Write Tests: Verify Voxel-to-RAS rotation.
    - [x] Implement: Use inverse rotation of affine.
- [x] Task: Update `StatefulImage.load()` with `is_direction_image` and `is_world_space` parameters.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Core Implementation' (Protocol in workflow.md)

## Phase 2: Tracking Integration
- [ ] Task: Analyze `scil_tracking_local.py` for fODF/Peak loading.
- [ ] Task: Integrate `to_voxel_direction()` call after loading directional images.
    - [ ] Write Tests: Regression test for tracking through an oblique affine.
    - [ ] Implement: Apply transformation to loaded `StatefulImage`.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Tracking Integration' (Protocol in workflow.md)

## Phase 3: Visualization Integration
- [ ] Task: Analyze `scilpy/viz/backends/fury.py` and `scil_viz_bundle.py`.
- [ ] Task: Integrate `to_voxel_direction()` in ODF/Peak actor creation.
    - [ ] Write Tests: Visual verification script (save screenshot or manual check).
    - [ ] Implement: Apply transformation before passing data to Fury actors.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Visualization Integration' (Protocol in workflow.md)
