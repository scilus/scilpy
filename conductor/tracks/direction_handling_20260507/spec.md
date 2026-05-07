# Specification: Directional Orientation Management in StatefulImage

## Background
DIPY's tracking and visualization tools (Fury) often assume directional data (SH coefficients, Peaks) is in voxel space or applies its own rotation based on the image affine. If the input data is already in world space (RAS), this leads to a "double-rotation" error.

## Objective
Enhance `StatefulImage` to handle the transformation of directional data between world space and voxel space, providing a centralized API to solve orientation issues in tracking and visualization scripts.

## Requirements

### 1. StatefulImage Enhancements
- **New Method: `to_voxel_direction()`**
  - Transforms directional data from world space to the current in-memory voxel space.
  - Must handle SH coefficients (l=0, 2, 4...) and Peaks (N, 3).
  - Must respect the current image stride and voxel order.
- **New Method: `to_world_direction()`**
  - Transforms directional data from voxel space to world space (RAS).
- **Legacy Support in `load()`**:
  - Add an argument (e.g., `is_direction_image=False`, `is_world_space=True`) to specify if the loaded image contains directional data and its current space.

### 2. Integration
- **Tracking:** Update `scil_tracking_local.py` to ensure fODFs/Peaks are moved to voxel space before being passed to the direction getter.
- **Visualization:** Update visualization backends (Fury) to handle directional data transformations consistently.

### 3. Verification
- Verify that a non-canonical affine (oblique) results in correct ODF/Peak orientation when moved to voxel space.
- Compare against manual `apply_affine` translations used in previous attempts.
