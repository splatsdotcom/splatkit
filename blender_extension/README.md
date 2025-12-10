# Splats - Multi-Frame Gaussian Capture

The Blender extension used by **splatkit** to generate Gaussian Splats from Blender scenes. This extension renders multi-frame sequences from multiple camera positions arranged on a sphere, creating the training data needed for 3D Gaussian Splatting.

## Overview

This extension is the core component of the splatkit workflow, enabling you to create Gaussian Splats directly from your Blender scenes. It automates the process of capturing your 3D scenes from multiple viewpoints by intelligently positioning cameras around your scene using Fibonacci sphere distribution. The extension generates all the necessary data—multi-view images, point clouds, and camera parameters—that splatkit uses to train Gaussian Splatting models.

## Key Features

### Multi-Camera Rendering
- **Fibonacci Sphere Distribution**: Automatically positions cameras evenly around your scene using mathematically optimal Fibonacci sphere distribution
- **Configurable Camera Count**: Set anywhere from 1 to 100 camera positions (default: 60)
- **Automatic Sphere Generation**: One-click creation of camera sphere based on scene bounds or selected objects
- **Radius Variation**: Optional random depth variation (0.8-1.0x) or fixed radius for consistent camera distances

### Animation & Frame Sequences
- **Multi-Frame Support**: Render complete animation sequences from all camera positions
- **Frame Range Control**: Specify start and end frames for your sequence
- **Efficient Batch Rendering**: Optimized rendering pipeline that processes all frames per camera in a single pass

### Live Camera Preview
- **Interactive Preview Mode**: Preview camera positions before rendering
- **Adjustable Radius**: Real-time radius adjustment with visual feedback
- **Camera Navigation**: Cycle through camera positions to preview each viewpoint
- **Apply Preview Settings**: Lock in your preferred radius for the final render

### Smart Rendering Optimizations
- **Automatic Frustum Culling**: Hides objects outside camera views to speed up rendering
- **Union Culling Logic**: Objects visible from at least one camera are kept, optimizing render times
- **4K Resolution**: Automatically renders at 3840×2160 resolution for high-quality captures

### Comprehensive Export
- **Per-Frame PLY Export**: Exports point cloud meshes for each frame with consistent world bounds
- **Camera Parameters**: Generates OpenCV-format YAML files with:
  - Intrinsic matrices (K matrices) for all cameras
  - Extrinsic matrices (world-to-camera transformations)
  - Distortion coefficients
  - Image dimensions
- **Organized Output**: Creates structured directory hierarchy with separate folders for images and point clouds
- **ZIP Archive**: Automatically packages all outputs into a single ZIP file for easy distribution

### Progress Tracking
- **Real-Time Progress**: Visual progress bar and percentage completion
- **Time Estimates**: ETA calculations for current camera and overall render time
- **Detailed Status**: Shows current camera number, frame progress, and timing information

## Installation

### As Extension (Blender 4.2+)
1. Build the extension using `blender --command extension build` (see Building section below)
2. Install from the Extensions platform, or
3. Install from disk: **Edit > Preferences > Extensions > Install from Disk**

### Legacy Installation (Blender < 4.2)
Use the "Install legacy Add-on" button in User Preferences.

## Usage

### Quick Start

1. **Create Camera Sphere**
   - Open the "Multi-Cam" panel in the 3D Viewport sidebar (press `N` if sidebar is hidden)
   - Click "Create Auto Sphere" to automatically generate a camera sphere based on your scene bounds
   - Or select a mesh object and click "Create Auto Sphere" to center the sphere on that object

2. **Preview Camera Positions** (Optional)
   - Click "Enable Preview" to enter preview mode
   - Adjust the radius multiplier slider to change camera distance
   - Use arrow buttons to cycle through camera positions
   - Click "Use This Radius" to lock in your preferred radius for rendering

3. **Configure Render Settings**
   - Set the number of cameras (default: 60)
   - Specify start and end frame numbers
   - Choose output folder and folder name

4. **Render**
   - Click "▶ RENDER ALL CAMERAS" to start the rendering process
   - Monitor progress in the panel with real-time ETA updates

5. **Access Output**
   - Find your rendered data in the specified output folder
   - All outputs are automatically packaged into a ZIP file for easy distribution

### Output Structure

```
output_folder/
└── folder_name.zip
    └── folder_name/
        ├── images/
        │   ├── 00/
        │   │   ├── 000000.png
        │   │   ├── 000001.png
        │   │   └── ...
        │   ├── 01/
        │   └── ...
        ├── pointclouds/
        │   ├── 000000.ply
        │   ├── 000001.ply
        │   └── ...
        ├── intri.yml
        └── extri.yml
```

## Use Cases

- **Gaussian Splatting**: Generate training data for 3D Gaussian Splatting models
- **Neural Radiance Fields (NeRF)**: Create multi-view datasets for NeRF training
- **Photogrammetry**: Capture scenes from multiple angles for reconstruction
- **3D Reconstruction**: Generate comprehensive camera datasets for structure-from-motion pipelines
- **Computer Vision Research**: Create standardized multi-view datasets with known camera parameters

## Technical Details

### Output Format
- **Images**: PNG format (RGBA, transparent background) at 3840×2160 resolution
- **Point Clouds**: PLY format with vertex positions and face indices
- **Camera Parameters**: YAML files in OpenCV format
- **Coordinate System**: OpenCV convention for camera matrices (compatible with most CV libraries)
- **Scene Units**: Automatically sets metric units (meters) for accurate world-space coordinates
- **Modifier Support**: Applies modifiers when exporting geometry for accurate mesh representation

## Building the Extension

### Method 1: Using Blender Command (Recommended)

To build the extension package:

```bash
blender --command extension build
```

This will create a `.zip` file in the current directory that can be installed or uploaded to the Extensions platform.

To validate the manifest before building:

```bash
blender --command extension validate
```

### Method 2: Manual ZIP Creation (Workaround)

If the Blender command-line tool has Python initialization issues, you can manually create the extension zip:

```bash
zip -r splats-1.0.0.zip __init__.py blender_manifest.toml LICENSE README.md DESCRIPTION.md -x "*.pyc" "__pycache__/*" ".DS_Store" "__MACOSX/*" "*.zip"
```

**Note**: Make sure to exclude `__MACOSX/` and `.DS_Store` files to avoid validation errors.

## Extension Format

This add-on has been converted to the Blender 4.2+ Extension format. It includes:
- `blender_manifest.toml` - Extension metadata (replaces `bl_info`)
- `__init__.py` - Main add-on code
- Single-file Python module with no sub-packages

## License

GPL-3.0-or-later

## Author

Splats  
GitHub: https://github.com/splatsdotcom
