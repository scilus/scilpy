# Initial Concept
Ok revert that, for both viz and tracking we will have the same solution: A new function in the statefulImage to revert direction image (peaks, sh, sf) to image space (but respect the stride/voxel_order) which should be called after loading in viz or tracking. And a matching revert to world space (which has no use for now). And just in case a option to mention if the loaded fodf are already in image space so they can be modified to go to world space (facilitate backcompatibility).

# Scilpy: Directional Orientation Management

## Vision
To provide a robust and consistent framework for handling directional dMRI data (fODFs/SH, Peaks, SF) within the `StatefulImage` ecosystem, ensuring that data is always correctly oriented for tracking and visualization regardless of its storage space (World or Voxel).

## Target Audience
- **Researchers:** Who need reliable orientation for their tractography and visualization pipelines.
- **Developers:** Who want a clean, centralized API for orientation transformations.
- **Data Scientists:** Working with complex dMRI datasets with varying orientation conventions.

## Key Features
- **Directional Space Transformation:** New `StatefulImage` methods to transform direction-based images between World Space (RAS) and Voxel Space (respecting stride/voxel order).
- **Tracking & Viz Integration:** Centralized call point after loading images in visualization and tracking scripts to prevent "double-rotation" issues.
- **Legacy Compatibility:** Options to flag loaded data as already being in Voxel Space, enabling a seamless transition to the new World-Space-by-default standard.
- **Stride Awareness:** Explicit handling of voxel strides and axis orders during rotation to maintain spatial integrity.
