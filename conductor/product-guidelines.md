# Product Guidelines

## Prose Style
- **Technical & Scientific:** Documentation and messages should be direct, objective, and precise.
- **Explicit Terminology:** Always distinguish clearly between "World Space" (RAS mm) and "Voxel Space" (Indices/Stride).

## Code Style & Documentation
- **PEP8:** All Python code must adhere to PEP8 standards.
- **NumPy Style Docstrings:** Follow the established project convention for all new functions and classes.
- **Type Hinting:** Use type hints for all public API methods to improve maintainability and IDE support.

## UX & Interaction
- **Concise Logging:** Prefer high-signal, low-noise logging. Only output essential progress and critical warnings/errors.
- **CLI Consistency:** Maintain consistent parameter naming and behavior across tracking and visualization scripts.

## Scientific Integrity
- **Orientation Safety:** Transformations affecting orientation must be verified against reference datasets (e.g., identity vs. non-canonical affines).
- **Non-Destructive Operations:** Transformations within `StatefulImage` should avoid modifying the raw data on disk unless explicitly requested.
