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

## Use Cases

- **Gaussian Splatting**: Generate training data for 3D Gaussian Splatting models
- **Neural Radiance Fields (NeRF)**: Create multi-view datasets for NeRF training
- **Photogrammetry**: Capture scenes from multiple angles for reconstruction
- **3D Reconstruction**: Generate comprehensive camera datasets for structure-from-motion pipelines
- **Computer Vision Research**: Create standardized multi-view datasets with known camera parameters

## Technical Details

- **Output Format**: PNG images (RGBA, transparent background), PLY point clouds, YAML camera parameters
- **Coordinate System**: OpenCV convention for camera matrices (compatible with most CV libraries)
- **Scene Units**: Automatically sets metric units (meters) for accurate world-space coordinates
- **Modifier Support**: Applies modifiers when exporting geometry for accurate mesh representation

## Workflow

1. **Setup**: Create or select a camera sphere (empty object) to define camera positions
2. **Preview**: Use preview mode to adjust camera radius and verify viewpoints
3. **Configure**: Set number of cameras, frame range, and output location
4. **Render**: Click "Render All Cameras" and monitor progress
5. **Export**: Receive organized ZIP file with images, point clouds, and camera parameters

Perfect for artists, researchers, and developers working with modern 3D reconstruction techniques.