# Cell Infiltrations Project - AI Coding Instructions

## Project Overview
A desktop image analysis tool (Tkinter GUI) for quantifying cell infiltration patterns in microscopy images. The application processes TIFF/PNG/JPG images through computer vision algorithms to detect infiltration areas and calculate statistics.

**Key Purpose**: Analyze histology samples to compute the percentage of white (bright) infiltration areas within gray (darker) tissue regions, supporting medical research workflows.

## Architecture

### Single-File Monolith (`main.py`)
- **Structure**: All code in one file (~600 lines) organized as:
  1. Image processing functions (gamma correction, mask extraction, visualization)
  2. `ImageViewer` class (Tkinter GUI application with state management)
  3. CLI entry point with `--debug` flag support

### Core Processing Pipeline
```
Load Image → Gamma Correct → Extract Masks → Apply Deletions → Calculate Stats → Visualize
```

**Key Data Structures** (in `ImageViewer`):
- `image_settings`: `{image_path: {'sensitivity': int, 'white_threshold': int, 'deleted_contours': []}}`
- `image_display_stats`: `{image_path: {'gray_area': px, 'white_area': px, 'percentage': float}}`
- State is **per-image**, persisted across navigation using these dictionaries

## Critical Implementation Patterns

### Gamma Correction (`gamma_correct_target_d`)
- Uses **dichotomy (binary search)** to normalize image intensity to target mean (100)
- Handles both brightening (gamma < 1.0) and darkening (gamma > 1.0)
- Applied FIRST before any mask extraction—normalization is essential for algorithm robustness

### Mask Extraction (`extract_cells_infiltrations_masks`)
- **Three-step filtration**:
  1. Light pixel detection (threshold ≥ 100) → morphological cleaning
  2. Gray pixel detection (100 < value < 150) → separate morphological filtering
  3. Overlap validation: only gray contours with gray-pixel interior are kept
- White mask extracted from brightest pixels (default threshold: 175) within validated regions
- Returns `(validated_mask, white_mask_final)` — both used for display and statistics

**Critical Parameter**: `gray_morph_iterations` (slider, 1-15) controls opening iterations on gray mask; directly impacts gray region sensitivity

### Interactive Deletion System
- **Persistent**: Deleted contours stored in `image_settings['deleted_contours']` across image switches
- **Toggle Mechanism**: Click contour to delete; click inside deleted region outline to reinstate
- Uses `cv2.pointPolygonTest()` for point-in-polygon detection
- Display scale factor calculated on resize to map screen clicks → original image coords
- Deleted contours drawn from **base masks** (pre-deletion), not active masks

### Statistics Calculation
- **Per-image**: White area / Gray area × 100 = Infiltration %
- **Aggregate**: Summed across all images with stats calculated (excludes images with processing errors)
- **Updating**: Recalculated in `show_image()` after every mask/deletion change
- Persisted in `image_display_stats` for CSV export

## Developer Workflows

### Running the Application
```bash
python main.py                    # Normal mode
python main.py --debug            # Debug visualization mode (development)
```

### Debug Mode Features
- Replaces mask overlay with 6-panel grid:
  1. Corrected image
  2. Grayscale
  3. Initial threshold mask
  4. Morphological-filtered mask
  5. Gray region filtering (shows iterations parameter effect)
  6. Validated mask (after overlap check)
- Grid resized to ~50% of display label for all-in-view inspection
- Toggling mask icon automatically disables debug mode

### Testing & Validation
- **Jupyter Notebook** (`tests_classic_cv.ipynb`): Prototyping functions before integration into GUI
- **Sample Data** (`data/`): 13 TIFF images for manual testing (AA/CA prefixes indicate sample types)
- **Expected Workflow**: Modify functions in notebook, sync validated versions to `main.py`

### Building Executables for Distribution
- **Local build**: `pyinstaller --onefile --noconsole build.spec`
- **CI/CD (GitHub Actions)**: Automatically builds for Linux, macOS (x86_64 + arm64), Windows (x86 + x64) on:
  - Push to main branch (artifacts retained 30 days)
  - Git tag push `v*` (creates GitHub Release with all artifacts)
- See `BUILD_DISTRIBUTION.md` for detailed release process

## Project-Specific Conventions

### Parameter Tuning
- **Sensitivity Slider**: `gray_morph_iterations` (1-15) — increased values reduce noise but may lose small infiltration zones
- **White Threshold Slider**: Value (100-254) — raised threshold captures only brightest pixels, lowered includes grayer whites
- **Both parameters stored per-image**: Users can fine-tune each image independently

### State Management Pattern
When modifying `show_image()` or callbacks (`_on_sensitivity_change`, etc.):
1. Load settings from `image_settings` dict or initialize with GUI slider values
2. Calculate base masks using settings
3. Apply deletion list to create active masks
4. Calculate stats from active masks
5. Store/update both dictionaries
6. Generate visualization (debug vs. normal)
7. Update status labels (calls `_update_display_info()`)

### Image Coordinate Mapping
- Original image stored as numpy array in OpenCV (BGR)
- PIL conversion for Tkinter display (RGB)
- Display resizing computed respecting aspect ratio with `display_scale_factor`
- Click coordinates mapped: label → display image → original image coords

## Key Files & Their Roles

| File | Purpose |
|------|---------|
| `main.py` | Full application (processing + GUI) |
| `main.spec` | PyInstaller configuration for executable builds |
| `Cell Infiltrations.spec` | Alternative PyInstaller spec (unused but archived) |
| `tests_classic_cv.ipynb` | Algorithm prototyping & validation |
| `requirements.txt` | Dependencies (opencv, pillow, numpy) |
| `data/` | 13 sample TIFF images for testing |
| `result.csv` | Last exported statistics (user-generated) |

## External Dependencies & Integration Points

- **OpenCV (`cv2`)**: Morphological operations, contour detection, coordinate geometry
- **PIL/Pillow**: Image I/O, format conversion, Tkinter display
- **NumPy**: Mask arrays, pixel operations
- **Tkinter**: Built-in GUI framework (no external setup required)

**No external APIs or databases**—fully self-contained.

## Debugging Tips

1. **Contour Issues**: Enable debug mode to visualize intermediate masks; compare "Area Filtered" vs. "Validated Mask"
2. **Missing Infiltrations**: Check white threshold slider (lowering may capture grayed whites); debug grid shows gray detection
3. **State Corruption**: Verify `image_settings` has `deleted_contours` as list of numpy arrays (not lists)
4. **Resize Flickering**: `_on_resize` debounces with 5px tolerance; check `display_scale_factor` calculation if stuck

## Known Limitations & Future Considerations

- **Single-file design**: Growing complexity may benefit from modularization (e.g., `cv_pipeline.py`, `gui.py`)
- **Deleted Contours Reinstatement**: Current UI requires clicking the outline; could add "Clear All" button
- **CSV Export**: Lacks image metadata (date, sample type); could be extended with user-defined fields
