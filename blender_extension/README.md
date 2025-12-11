# Splats - Multi-Frame Gaussian Capture

The Blender extension used by **splatkit** to generate Gaussian Splats from Blender scenes. This extension renders multi-frame sequences from multiple camera positions arranged on a sphere, creating the training data needed for Dynamic Gaussian Splatting.

## Overview

This extension is the core component of the splatkit workflow, enabling you to create Gaussian Splats directly from your Blender scenes. It automates the process of capturing your 3D scenes from multiple viewpoints by intelligently positioning cameras around your scene using Fibonacci sphere distribution. The extension generates all the necessary data—multi-view images, point clouds, and camera parameters—that splatkit uses to train Gaussian Splatting models.

## Installation

### As Extension (Blender 4.2+)
1. Download Extension here: https://drive.google.com/file/d/1YDTYruVlezDdC1zc-Yei-UCfoRB5gPHo/view?usp=drive_link
2. Install from disk: **Edit > Preferences > Extensions > Install from Disk**
3. Installation from the Extensions platform is coming soon,

### Legacy Installation (Blender < 4.2)
Use the "Install legacy Add-on" button in User Preferences, using the `splats.py` file.

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


## License

GPL-3.0-or-later

## Author

Splats  
GitHub: https://github.com/splatsdotcom
