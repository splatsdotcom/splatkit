"""
Multi-Camera Render Addon for Blender
Renders scene from multiple camera positions on a sphere for 3D reconstruction

Features:
- Multi-frame sequence rendering support
- Single render per camera (optimized for speed)
- Random camera radius variation for diversity
- Detailed progress tracking with time estimates
- Live camera preview with adjustable radius
- Automatic frustum culling to hide objects outside camera views
- Per-frame PLY point cloud export with consistent world bounds

Author: Splats
GitHub: https://github.com/orgs/True3DLabs/repositories

Code Organization:
- Lines 63-780: Utility functions (path validation, geometry, file I/O, camera utilities)
- Lines 780-850: Preview functions (camera preview and updates)
- Lines 850-870: Render callbacks (frame change handlers)
- Lines 870-1500: Operators (main rendering logic, validation, export functions)
- Lines 1500-2180: UI Panel (user interface)
- Lines 2180-2376: Properties and Registration (Blender addon registration)
"""

bl_info = {
    "name": "Multi-Frame Gaussian Capture",
    "author": "Splats",
    "version": (1, 0, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > Multi-Cam",
    "description": "Render multi-frame sequences from multiple camera positions for 3D reconstruction",
    "category": "Render",
    "doc_url": "https://github.com/orgs/True3DLabs/repositories",
    "tracker_url": "https://github.com/orgs/True3DLabs/repositories/issues",
}

import bpy
import math
import os
import re
import time
import zipfile
import json
import random
from datetime import datetime
import numpy as np
from mathutils import Matrix, Vector
from bpy.props import (
    IntProperty,
    StringProperty,
    PointerProperty,
    FloatProperty,
    BoolProperty,
)
from bpy.types import (
    Operator,
    Panel,
    PropertyGroup,
)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def sanitize_filename(name):
    """
    Sanitize a filename/folder name by removing invalid characters.
    
    Removes or replaces characters that are invalid in file/folder names
    on common operating systems (Windows, macOS, Linux).
    
    Args:
        name (str): Original filename/folder name
    
    Returns:
        str: Sanitized filename safe for use in file paths
    """
    # Remove or replace invalid characters
    # Invalid chars: < > : " / \ | ? * and control characters
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)
    # Remove leading/trailing spaces and dots (Windows doesn't allow these)
    sanitized = sanitized.strip(' .')
    # Replace multiple consecutive underscores with single underscore
    sanitized = re.sub(r'_+', '_', sanitized)
    # Ensure it's not empty
    if not sanitized:
        sanitized = "untitled"
    # Limit length to avoid filesystem issues (255 chars is safe for most systems)
    if len(sanitized) > 255:
        sanitized = sanitized[:255]
    return sanitized


def validate_and_resolve_path(path, context):
    """
    Validate and resolve a file path, ensuring it's safe and writable.
    
    Args:
        path (str): Path to validate (can be relative with // prefix)
        context: Blender context for path resolution
    
    Returns:
        tuple: (resolved_path, error_message)
            - resolved_path: Absolute resolved path or None if invalid
            - error_message: Error message string or None if valid
    """
    if not path or path.strip() == "":
        return None, "Path is empty"
    
    try:
        # Resolve Blender relative paths (// prefix)
        resolved = bpy.path.abspath(path)
        
        # Check if path is absolute
        if not os.path.isabs(resolved):
            return None, f"Path could not be resolved to absolute path: {path}"
        
        # Check if parent directory exists or can be created
        parent_dir = os.path.dirname(resolved)
        if parent_dir and not os.path.exists(parent_dir):
            try:
                os.makedirs(parent_dir, exist_ok=True)
            except (OSError, PermissionError) as e:
                return None, f"Cannot create parent directory: {e}"
        
        # Check if we can write to the directory
        if os.path.exists(resolved):
            if not os.access(resolved, os.W_OK):
                return None, f"Path is not writable: {resolved}"
        else:
            # Check if parent is writable
            if parent_dir and not os.access(parent_dir, os.W_OK):
                return None, f"Parent directory is not writable: {parent_dir}"
        
        return resolved, None
    except Exception as e:
        return None, f"Error validating path: {e}"


def fibonacci_sphere_points(n_points):
    """
    Generate n_points evenly distributed on unit sphere surface using Fibonacci spiral.
    
    The Fibonacci spiral method provides excellent distribution of points on a sphere,
    avoiding clustering at poles and ensuring even coverage.
    
    Args:
        n_points (int): Number of points to generate on the sphere
    
    Returns:
        list: List of (x, y, z) tuples representing points on a unit sphere
    """
    points = []
    phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle in radians

    for i in range(n_points):
        # Handle single point case
        if n_points == 1:
            y = 1.0
        else:
            y = 1 - (i / float(n_points - 1)) * 2.0  # y goes from 1 to -1
        radius_at_y = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius_at_y
        z = math.sin(theta) * radius_at_y

        points.append((x, y, z))

    return points


def get_scene_bounds(context):
    """
    Calculate bounding box of all visible mesh objects in the scene.
    
    Args:
        context: Blender context
    
    Returns:
        tuple: (center, max_distance, (min_co, max_co)) or (None, None, None) if no objects
            - center: Vector representing the center point
            - max_distance: Maximum distance from center to any corner
            - min_co, max_co: Minimum and maximum corner coordinates
    """
    min_co = Vector((float('inf'), float('inf'), float('inf')))
    max_co = Vector((float('-inf'), float('-inf'), float('-inf')))

    has_objects = False
    for obj in context.scene.objects:
        if obj.type == 'MESH' and obj.visible_get():
            has_objects = True
            # Get world space bounding box
            bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
            for corner in bbox_corners:
                min_co.x = min(min_co.x, corner.x)
                min_co.y = min(min_co.y, corner.y)
                min_co.z = min(min_co.z, corner.z)
                max_co.x = max(max_co.x, corner.x)
                max_co.y = max(max_co.y, corner.y)
                max_co.z = max(max_co.z, corner.z)

    if not has_objects:
        return None, None, None

    center = (min_co + max_co) / 2
    max_distance = max([(Vector(corner) - center).length
                        for corner in [min_co, max_co,
                                       Vector((min_co.x, min_co.y, max_co.z)),
                                       Vector((min_co.x, max_co.y, min_co.z)),
                                       Vector((max_co.x, min_co.y, min_co.z)),
                                       Vector((min_co.x, max_co.y, max_co.z)),
                                       Vector((max_co.x, min_co.y, max_co.z)),
                                       Vector((max_co.x, max_co.y, min_co.z))]])

    return center, max_distance, (min_co, max_co)


def is_object_in_camera_frustum(obj, camera_obj, scene):
    """
    Check if object's bounding box intersects with camera's view frustum.
    
    Performs frustum culling by checking if any corner of the object's bounding box
    is within the camera's field of view and clip range.
    
    Args:
        obj (bpy.types.Object): Blender mesh object to check
        camera_obj (bpy.types.Object): Blender camera object
        scene (bpy.types.Scene): Blender scene
    
    Returns:
        bool: True if object is potentially visible from camera, False otherwise
    """
    # Get camera data
    cam_data = camera_obj.data
    cam_matrix = camera_obj.matrix_world
    
    # Get camera location and orientation
    cam_loc = cam_matrix.translation
    cam_forward = -cam_matrix.col[2].xyz.normalized()
    cam_right = cam_matrix.col[0].xyz.normalized()
    cam_up = cam_matrix.col[1].xyz.normalized()
    
    # Get camera clip distances
    clip_start = cam_data.clip_start
    clip_end = cam_data.clip_end
    
    # Get camera FOV
    if cam_data.type == 'PERSP':
        sensor_width = cam_data.sensor_width
        focal_length = cam_data.lens
        fov = 2 * math.atan(sensor_width / (2 * focal_length))
        aspect = scene.render.resolution_x / scene.render.resolution_y
    else:
        # Orthographic - use ortho_scale
        ortho_scale = cam_data.ortho_scale
        aspect = scene.render.resolution_x / scene.render.resolution_y
        half_width = ortho_scale / 2
        half_height = half_width / aspect
    
    # Get object bounding box corners in world space
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    
    # Check if any corner is within frustum
    for corner in bbox_corners:
        # Vector from camera to corner
        to_corner = corner - cam_loc
        
        # Check if within clip range
        depth = to_corner.dot(cam_forward)
        if depth < clip_start or depth > clip_end:
            continue
        
        if cam_data.type == 'PERSP':
            # Perspective: check if within FOV
            # Project onto camera's right/up plane
            right_dist = to_corner.dot(cam_right)
            up_dist = to_corner.dot(cam_up)
            
            # Calculate frustum bounds at this depth
            frustum_width = 2 * depth * math.tan(fov / 2)
            frustum_height = frustum_width / aspect
            
            if abs(right_dist) < frustum_width / 2 and abs(up_dist) < frustum_height / 2:
                return True
        else:
            # Orthographic: check if within ortho bounds
            right_dist = to_corner.dot(cam_right)
            up_dist = to_corner.dot(cam_up)
            
            if abs(right_dist) < half_width and abs(up_dist) < half_height:
                return True
    
    # Also check if object center is in frustum (more lenient check)
    obj_center = obj.matrix_world.translation
    to_center = obj_center - cam_loc
    depth = to_center.dot(cam_forward)
    
    if depth >= clip_start and depth <= clip_end:
        if cam_data.type == 'PERSP':
            frustum_width = 2 * depth * math.tan(fov / 2)
            frustum_height = frustum_width / aspect
            right_dist = to_center.dot(cam_right)
            up_dist = to_center.dot(cam_up)
            if abs(right_dist) < frustum_width / 2 and abs(up_dist) < frustum_height / 2:
                return True
        else:
            right_dist = to_center.dot(cam_right)
            up_dist = to_center.dot(cam_up)
            if abs(right_dist) < half_width and abs(up_dist) < half_height:
                return True
    
    return False


def calculate_union_bounds(context, mesh_objects, start_frame, end_frame, apply_modifiers=True):
    """
    Calculate union bounding box across all frames for given mesh objects.
    
    Iterates through all frames in the specified range and computes a single
    bounding box that encompasses all object positions across all frames. This
    ensures consistent world bounds for point cloud normalization.
    
    Args:
        context (bpy.types.Context): Blender context
        mesh_objects (list): List of mesh objects to check
        start_frame (int): First frame to check
        end_frame (int): Last frame to check (inclusive)
        apply_modifiers (bool): Whether to apply modifiers when computing bounds
    
    Returns:
        tuple: (min_co, max_co) or (None, None) if no geometry found
            - min_co: Vector of minimum coordinates (x, y, z)
            - max_co: Vector of maximum coordinates (x, y, z)
    """
    all_mins = []
    all_maxs = []
    
    depsgraph = context.evaluated_depsgraph_get() if apply_modifiers else None
    
    for frame_num in range(start_frame, end_frame + 1):
        context.scene.frame_set(frame_num)
        context.view_layer.update()
        
        for obj in mesh_objects:
            if obj.type != 'MESH':
                continue
            
            try:
                if apply_modifiers and depsgraph:
                    eval_obj = obj.evaluated_get(depsgraph)
                    obj_matrix = eval_obj.matrix_world
                    # Get bounding box corners in world space
                    bbox_corners = [obj_matrix @ Vector(corner) for corner in obj.bound_box]
                else:
                    obj_matrix = obj.matrix_world
                    bbox_corners = [obj_matrix @ Vector(corner) for corner in obj.bound_box]
                
                # Get min/max from this object's bbox
                if bbox_corners:
                    obj_mins = [min(c[i] for c in bbox_corners) for i in range(3)]
                    obj_maxs = [max(c[i] for c in bbox_corners) for i in range(3)]
                    all_mins.append(obj_mins)
                    all_maxs.append(obj_maxs)
            except Exception:
                continue
    
    if not all_mins:
        return None, None
    
    # Union bounding box
    min_co = Vector([min(m[i] for m in all_mins) for i in range(3)])
    max_co = Vector([max(m[i] for m in all_maxs) for i in range(3)])
    
    return min_co, max_co


def export_meshes_to_ply(context, mesh_objects, output_path, apply_modifiers=True):
    """
    Manually export Blender mesh objects to PLY file.
    
    Exports vertices and faces from multiple mesh objects into a single PLY file.
    Transforms vertices to world space and handles triangulation automatically.
    
    Args:
        context (bpy.types.Context): Blender context
        mesh_objects (list): List of Blender mesh objects to export
        output_path (str): Path to output PLY file
        apply_modifiers (bool): Whether to apply modifiers (uses evaluated mesh)
    
    Returns:
        bool: True if export succeeded, False if no geometry to export
    """
    all_vertices = []
    all_faces = []
    vertex_offset = 0
    
    # Get evaluated depsgraph for modifiers if needed
    depsgraph = context.evaluated_depsgraph_get() if apply_modifiers else None
    
    for obj in mesh_objects:
        if obj.type != 'MESH':
            continue
            
        try:
            # Get mesh data (with modifiers applied if requested)
            if apply_modifiers and depsgraph:
                eval_obj = obj.evaluated_get(depsgraph)
                mesh = eval_obj.data
                obj_matrix = eval_obj.matrix_world
            else:
                mesh = obj.data
                obj_matrix = obj.matrix_world
            
            # Update mesh to ensure data is current
            mesh.update()
            
            # Skip if mesh has no vertices
            if len(mesh.vertices) == 0:
                continue
            
            # Get vertices in world space
            for vertex in mesh.vertices:
                world_vertex = obj_matrix @ vertex.co
                all_vertices.append((world_vertex.x, world_vertex.y, world_vertex.z))
            
            # Get faces (triangulate if needed)
            mesh.calc_loop_triangles()
            for triangle in mesh.loop_triangles:
                # Get vertex indices (add offset for this object)
                face_verts = [triangle.vertices[0] + vertex_offset,
                             triangle.vertices[1] + vertex_offset,
                             triangle.vertices[2] + vertex_offset]
                all_faces.append(face_verts)
            
            vertex_offset += len(mesh.vertices)
        except Exception as e:
            # Skip objects that fail to export
            print(f"Warning: Failed to export mesh from object '{obj.name}': {e}")
            continue
    
    if len(all_vertices) == 0:
        return False
    
    # Write PLY file
    num_vertices = len(all_vertices)
    num_faces = len(all_faces)
    
    try:
        with open(output_path, 'w') as f:
            # PLY header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {num_vertices}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write(f"element face {num_faces}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            
            # Write vertices
            for vertex in all_vertices:
                f.write(f"{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
            
            # Write faces
            for face in all_faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
        
        return True
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False


def cull_objects_outside_all_frustums(context, cameras, scene):
    """
    Hide objects that are NOT visible from ANY camera (union logic).
    
    Uses union culling: objects visible from at least one camera are kept,
    objects not visible from any camera are hidden. This optimizes rendering
    by not rendering objects that won't appear in any output image.
    
    Args:
        context (bpy.types.Context): Blender context
        cameras (list): List of camera objects to check against
        scene (bpy.types.Scene): Blender scene
    
    Returns:
        tuple: (hidden_count, visible_count, visibility_map)
            - hidden_count: Number of objects hidden
            - visible_count: Number of objects kept visible
            - visibility_map: Dictionary mapping object names to original hide_render state
    """
    # Build set of objects visible from ANY camera (union)
    visible_from_any = set()
    
    for cam in cameras:
        for obj in scene.objects:
            if obj.type == 'MESH' and obj.visible_get():
                if is_object_in_camera_frustum(obj, cam, scene):
                    visible_from_any.add(obj.name)
    
    # Store original visibility and hide objects not visible from any camera
    visibility_map = {}
    hidden_count = 0
    
    for obj in scene.objects:
        if obj.type == 'MESH':
            visibility_map[obj.name] = obj.hide_render
            if obj.name not in visible_from_any:
                if not obj.hide_render:
                    obj.hide_render = True
                    hidden_count += 1
    
    visible_count = len(visible_from_any)
    return hidden_count, visible_count, visibility_map


def calculate_camera_intrinsics(scene):
    """
    Calculate camera intrinsic matrix (K matrix) from scene camera settings.
    
    Computes the 3x3 intrinsic matrix containing focal length and principal point.
    Uses scene camera if available, otherwise falls back to default values.
    
    Args:
        scene (bpy.types.Scene): Blender scene
    
    Returns:
        tuple: (K, width, height)
            - K: 3x3 intrinsic matrix as nested list
            - width: Image width in pixels
            - height: Image height in pixels
    """
    camera = scene.camera

    if camera and camera.type == 'CAMERA':
        focal_length_mm = camera.data.lens
        sensor_width_mm = camera.data.sensor_width
    else:
        # Default values
        focal_length_mm = 50.0
        sensor_width_mm = 36.0

    width = scene.render.resolution_x
    height = scene.render.resolution_y
    aspect_ratio = width / height
    sensor_height_mm = sensor_width_mm / aspect_ratio

    # Calculate focal length in pixels
    fx = (focal_length_mm / sensor_width_mm) * width
    fy = (focal_length_mm / sensor_height_mm) * height

    # Principal point (image center)
    cx = width / 2.0
    cy = height / 2.0

    # Build K matrix (3x3 intrinsic matrix)
    K = [
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ]

    return K, width, height


def blender_to_opencv_matrix(blender_matrix):
    """
    Convert Blender camera matrix to OpenCV convention (camera-to-world).
    
    Blender uses +X right, +Y forward, +Z up (camera looks along -Z).
    OpenCV uses +X right, +Y down, +Z forward (camera looks along +Z).
    This function applies the necessary coordinate system transformation.
    
    Args:
        blender_matrix (Matrix): 4x4 Blender camera matrix
    
    Returns:
        Matrix: 4x4 camera-to-world matrix in OpenCV convention
    """
    # Blender: +X right, +Y forward, +Z up (camera looks along -Z)
    # OpenCV: +X right, +Y down, +Z forward (camera looks along +Z)
    # Conversion: rotate 180° around X-axis
    opencv_from_blender = Matrix((
        (1, 0, 0, 0),
        (0, -1, 0, 0),
        (0, 0, -1, 0),
        (0, 0, 0, 1)
    ))

    # Camera-to-world in OpenCV convention
    cam_to_world = blender_matrix @ opencv_from_blender
    return cam_to_world


def matrix_to_rodrigues(R):
    """
    Convert 3x3 rotation matrix to Rodrigues vector (axis-angle representation).
    
    The Rodrigues vector represents rotation as axis * angle, where the magnitude
    is the rotation angle and the direction is the rotation axis.
    
    Args:
        R (Matrix): 3x3 rotation matrix
    
    Returns:
        list: [rx, ry, rz] Rodrigues vector
    """
    # Convert to numpy for easier calculation
    R_np = np.array([[R[i][j] for j in range(3)] for i in range(3)])

    # Calculate rotation angle
    trace = np.trace(R_np)
    theta = np.arccos(np.clip((trace - 1) / 2, -1, 1))

    if theta < 1e-6:
        # No rotation
        return [0.0, 0.0, 0.0]

    # Calculate rotation axis
    if abs(theta - np.pi) < 1e-6:
        # 180 degree rotation - special case
        # Find the column with largest diagonal element
        i = np.argmax([R_np[0, 0], R_np[1, 1], R_np[2, 2]])
        axis = np.zeros(3)
        axis[i] = np.sqrt((R_np[i, i] + 1) / 2)
        for j in range(3):
            if i != j:
                axis[j] = R_np[i, j] / (2 * axis[i])
    else:
        # Normal case
        axis = np.array([
            R_np[2, 1] - R_np[1, 2],
            R_np[0, 2] - R_np[2, 0],
            R_np[1, 0] - R_np[0, 1]
        ]) / (2 * np.sin(theta))

    # Rodrigues vector is axis * angle
    rodrigues = axis * theta
    return rodrigues.tolist()


def format_float(value):
    """
    Format float with 10 decimal places for YAML output.
    
    Args:
        value (float): Value to format
    
    Returns:
        str: Formatted string with 10 decimal places
    """
    return f"{float(value):.10f}"


def write_opencv_matrix(f, name, rows, cols, data):
    """
    Write OpenCV matrix format to YAML file.
    
    Writes a matrix in OpenCV's YAML format with proper formatting.
    
    Args:
        f (file): Open file handle to write to
        name (str): Matrix name (e.g., "K_00", "R_00")
        rows (int): Number of rows
        cols (int): Number of columns
        data (list): Flattened matrix data (row-major order)
    """
    f.write(f"{name}: !!opencv-matrix\n")
    f.write(f"  rows: {rows}\n")
    f.write(f"  cols: {cols}\n")
    f.write(f"  dt: d\n")
    f.write(f"  data: [")
    f.write(", ".join([format_float(v) for v in data]))
    f.write("]\n")


def create_temp_camera(name, location, look_at):
    """
    Create a temporary camera object at the specified location, pointing at a target.
    
    Args:
        name (str): Name for the camera object
        location (Vector): World-space location for the camera
        look_at (Vector): World-space target point the camera should face
    
    Returns:
        bpy.types.Object: The created camera object
    """
    # Create camera data
    cam_data = bpy.data.cameras.new(name=name)
    cam_obj = bpy.data.objects.new(name=name, object_data=cam_data)

    # Set clip distances for proper depth range
    cam_data.clip_start = 0.01  # Very close
    cam_data.clip_end = 1000.0  # Far enough for most scenes

    # Link to scene
    bpy.context.scene.collection.objects.link(cam_obj)

    # Set location
    cam_obj.location = location

    # Point camera at target
    direction = look_at - location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam_obj.rotation_euler = rot_quat.to_euler()

    return cam_obj


def format_time(seconds):
    """
    Format seconds as human-readable time string.
    
    Args:
        seconds (float): Time in seconds
    
    Returns:
        str: Formatted time string (e.g., "5m 30s", "2h 15m")
    """
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h {minutes}m"


def create_zip(source_folder, zip_path):
    """
    Create a ZIP archive from a source folder.
    
    Args:
        source_folder (str): Path to the folder to zip
        zip_path (str): Path where the ZIP file should be created
    """
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_folder):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(source_folder))
                zipf.write(file_path, arcname)


def rename_frames_from_zero(directory, start_frame_blender, num_frames, extension):
    """
    Rename frame files to start from 000000 regardless of Blender frame numbers.
    
    This ensures consistent frame numbering in the output, always starting from 000000
    even if the Blender scene uses different frame numbers.
    
    Args:
        directory (str): Directory containing the frame files
        start_frame_blender (int): Starting frame number in Blender
        num_frames (int): Number of frames to rename
        extension (str): File extension (e.g., ".png", ".exr")
    """
    for frame_offset in range(num_frames):
        blender_frame_num = start_frame_blender + frame_offset
        target_frame_num = frame_offset

        # Original filename (Blender's frame number)
        original_name = f"{blender_frame_num:06d}{extension}"
        original_path = os.path.join(directory, original_name)

        # Target filename (starting from 000000)
        target_name = f"{target_frame_num:06d}{extension}"
        target_path = os.path.join(directory, target_name)

        if os.path.exists(original_path) and original_path != target_path:
            os.rename(original_path, target_path)


# ============================================================================
# PREVIEW FUNCTIONS
# ============================================================================

def update_preview_camera(self, context):
    """Update preview camera position when properties change"""
    props = context.scene.multi_cam_props

    if not props.preview_enabled or not props.camera_sphere:
        return

    # Get or create preview camera
    preview_cam = None
    for obj in bpy.data.objects:
        if obj.name == "PreviewCamera_MultiCam":
            preview_cam = obj
            break

    if not preview_cam:
        # Create preview camera
        cam_data = bpy.data.cameras.new("PreviewCamera_MultiCam")
        preview_cam = bpy.data.objects.new("PreviewCamera_MultiCam", cam_data)
        context.scene.collection.objects.link(preview_cam)

    # Calculate position
    sphere_center = props.camera_sphere.location
    sphere_radius = props.camera_sphere.empty_display_size

    # Generate Fibonacci sphere points
    points = fibonacci_sphere_points(props.num_cameras)

    # Get the selected camera position
    cam_index = min(props.preview_camera_index, len(points) - 1)
    x, y, z = points[cam_index]

    # Apply radius multiplier
    actual_radius = sphere_radius * props.preview_radius_multiplier

    # Set position
    cam_location = Vector((
        sphere_center.x + x * actual_radius,
        sphere_center.y + y * actual_radius,
        sphere_center.z + z * actual_radius
    ))

    preview_cam.location = cam_location

    # Point at center
    direction = sphere_center - cam_location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    preview_cam.rotation_euler = rot_quat.to_euler()

    # Make it the active camera
    context.scene.camera = preview_cam

    # Force viewport update
    for area in context.screen.areas:
        if area.type == 'VIEW_3D':
            area.tag_redraw()


# ============================================================================
# RENDER CALLBACKS
# ============================================================================

def frame_change_handler(scene, depsgraph):
    """Called when frame changes during animation rendering"""
    props = scene.multi_cam_props
    if props.is_rendering and props.camera_start_time > 0:
        # Update current frame based on scene's current frame
        frame_offset = scene.frame_current - props.start_frame
        props.current_frame = frame_offset + 1  # 1-indexed for display

        # Calculate time elapsed for current camera
        elapsed = time.time() - props.camera_start_time

        # Estimate remaining time for current camera
        if props.current_frame > 0 and props.total_frames > 0:
            avg_time_per_frame = elapsed / props.current_frame
            remaining_frames = props.total_frames - props.current_frame
            props.current_camera_eta = avg_time_per_frame * remaining_frames


# ============================================================================
# OPERATORS
# ============================================================================

class RENDER_OT_create_sphere(Operator):
    """Create camera sphere based on scene bounds"""
    bl_idname = "render.create_camera_sphere"
    bl_label = "Create Auto Sphere"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        # Check if an object is selected - use its center if available
        center = None
        max_distance = None
        bounds = None
        
        if context.selected_objects:
            # Use the first selected object's center
            selected_obj = context.selected_objects[0]
            if selected_obj.type == 'MESH':
                # Get object's bounding box in world space
                bbox_corners = [selected_obj.matrix_world @ Vector(corner) for corner in selected_obj.bound_box]
                
                # Calculate min/max
                min_co = Vector((float('inf'), float('inf'), float('inf')))
                max_co = Vector((float('-inf'), float('-inf'), float('-inf')))
                
                for corner in bbox_corners:
                    min_co.x = min(min_co.x, corner.x)
                    min_co.y = min(min_co.y, corner.y)
                    min_co.z = min(min_co.z, corner.z)
                    max_co.x = max(max_co.x, corner.x)
                    max_co.y = max(max_co.y, corner.y)
                    max_co.z = max(max_co.z, corner.z)
                
                # Center is midpoint of bounding box
                center = (min_co + max_co) / 2
                
                # Calculate max distance from center to corners
                max_distance = max([(Vector(corner) - center).length for corner in bbox_corners])
                bounds = (min_co, max_co)
        
        # Fall back to scene bounds if no object selected
        if center is None:
            center, max_distance, bounds = get_scene_bounds(context)

        if center is None:
            self.report({'ERROR'}, "No renderable objects in scene or selected object")
            return {'CANCELLED'}

        # Get FOV
        camera = context.scene.camera
        if camera and camera.type == 'CAMERA':
            fov_rad = camera.data.angle
        else:
            fov_rad = math.radians(50)  # Default 50° FOV

        # Calculate radius with 50% safety margin to prevent clipping
        radius = (max_distance / math.tan(fov_rad / 2)) * 2.0

        # Create empty
        bpy.ops.object.empty_add(type='SPHERE', location=center)
        sphere = context.active_object
        sphere.name = "CameraRig_Sphere"
        sphere.empty_display_size = radius

        # Store in scene properties
        context.scene.multi_cam_props.camera_sphere = sphere

        if context.selected_objects:
            self.report({'INFO'}, f"Created sphere at selected object center {center} with radius {radius:.2f}m")
        else:
            self.report({'INFO'}, f"Created sphere at scene center {center} with radius {radius:.2f}m")
        return {'FINISHED'}


class RENDER_OT_toggle_preview(Operator):
    """Toggle camera preview mode"""
    bl_idname = "render.toggle_camera_preview"
    bl_label = "Toggle Preview"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.multi_cam_props

        if not props.camera_sphere:
            self.report({'ERROR'}, "Please create or select a camera sphere first")
            return {'CANCELLED'}

        # Toggle preview
        props.preview_enabled = not props.preview_enabled

        if props.preview_enabled:
            # Store original camera
            props.original_camera = context.scene.camera
            # Update preview camera
            update_preview_camera(None, context)
            self.report({'INFO'}, "Preview enabled - use slider to adjust radius")
        else:
            # Restore original camera
            if props.original_camera:
                context.scene.camera = props.original_camera

            # Clean up preview camera
            for obj in bpy.data.objects:
                if obj.name == "PreviewCamera_MultiCam":
                    bpy.data.objects.remove(obj, do_unlink=True)

            self.report({'INFO'}, "Preview disabled")

        return {'FINISHED'}


class RENDER_OT_preview_next(Operator):
    """Switch to next camera position in preview"""
    bl_idname = "render.preview_next_camera"
    bl_label = "Next Camera"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.multi_cam_props

        if not props.preview_enabled:
            self.report({'WARNING'}, "Preview not enabled")
            return {'CANCELLED'}

        # Cycle to next camera
        props.preview_camera_index = (props.preview_camera_index + 1) % props.num_cameras

        return {'FINISHED'}


class RENDER_OT_preview_previous(Operator):
    """Switch to previous camera position in preview"""
    bl_idname = "render.preview_previous_camera"
    bl_label = "Previous Camera"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.multi_cam_props

        if not props.preview_enabled:
            self.report({'WARNING'}, "Preview not enabled")
            return {'CANCELLED'}

        # Cycle to previous camera
        props.preview_camera_index = (props.preview_camera_index - 1) % props.num_cameras

        return {'FINISHED'}


class RENDER_OT_apply_preview_radius(Operator):
    """Apply the preview radius to be used in final render (sets render variation to this fixed value)"""
    bl_idname = "render.apply_preview_radius"
    bl_label = "Use This Radius"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.multi_cam_props

        # Store the preview radius as the fixed render radius
        props.use_fixed_radius = True
        props.fixed_radius_value = props.preview_radius_multiplier

        self.report({'INFO'}, f"Render will use fixed radius multiplier: {props.fixed_radius_value:.2f}")

        return {'FINISHED'}


class RENDER_OT_render_all(Operator):
    """
    Operator to render all cameras for a frame sequence.
    
    This operator handles the complete rendering pipeline:
    - Creates temporary cameras at specified positions
    - Performs frustum culling to hide objects outside camera views
    - Renders images for each camera across all frames
    - Exports PLY point clouds per frame
    - Generates camera intrinsics and extrinsics YAML files
    - Creates a ZIP archive of all outputs
    """
    bl_idname = "render.render_all_cameras"
    bl_label = "Render All Cameras"
    bl_options = {'REGISTER'}

    _timer = None
    _rendering = False
    _frame_times = []  # Store frame times for current camera
    _camera_times = []  # Store total times for completed cameras

    def modal(self, context, event):
        if event.type == 'ESC':
            self.cancel(context)
            return {'CANCELLED'}

        if event.type == 'TIMER':
            # Update UI
            context.area.tag_redraw()

        return {'PASS_THROUGH'}

    def execute(self, context):
        props = context.scene.multi_cam_props
        scene = context.scene

        # Disable preview if active
        if props.preview_enabled:
            bpy.ops.render.toggle_camera_preview()

        # Validation
        errors = self.validate_settings(context)
        if errors:
            self.report({'ERROR'}, "\n".join(errors))
            return {'CANCELLED'}

        # Ensure scene units are METRIC
        scene.unit_settings.system = 'METRIC'
        scene.unit_settings.length_unit = 'METERS'

        # Store original settings
        original_camera = scene.camera
        original_transparent = scene.render.film_transparent
        original_color_mode = scene.render.image_settings.color_mode
        original_filepath = scene.render.filepath
        original_use_nodes = scene.use_nodes
        original_res_x = scene.render.resolution_x
        original_res_y = scene.render.resolution_y
        original_frame_start = scene.frame_start
        original_frame_end = scene.frame_end
        original_resolution_percentage = scene.render.resolution_percentage

        # Force 4K resolution
        scene.render.resolution_x = 3840
        scene.render.resolution_y = 2160
        scene.render.resolution_percentage = 100

        # Setup
        sphere_center = props.camera_sphere.location
        sphere_radius = props.camera_sphere.empty_display_size
        num_cameras = props.num_cameras
        start_frame = props.start_frame
        end_frame = props.end_frame
        num_frames = end_frame - start_frame + 1

        # Initialize progress tracking
        props.total_frames = num_frames
        props.current_frame = 0
        props.current_camera = 0
        props.render_progress = 0
        props.is_rendering = True
        props.overall_start_time = time.time()
        props.camera_start_time = 0
        props.first_camera_total_time = 0
        props.current_camera_eta = 0
        props.overall_eta = 0
        self._camera_times = []

        # Register frame change handler for animation rendering
        if frame_change_handler not in bpy.app.handlers.frame_change_pre:
            bpy.app.handlers.frame_change_pre.append(frame_change_handler)

        # Resolve and create output directories with validation
        output_folder, folder_error = validate_and_resolve_path(props.output_folder, context)
        if folder_error:
            self.report({'ERROR'}, f"Output folder error: {folder_error}")
            return {'CANCELLED'}
        
        # Sanitize folder name
        safe_folder_name = sanitize_filename(props.folder_name)
        output_base = os.path.join(output_folder, safe_folder_name)
        images_dir = os.path.join(output_base, "images")
        
        try:
            os.makedirs(output_base, exist_ok=True)
        except (OSError, PermissionError) as e:
            self.report({'ERROR'}, f"Cannot create output directory: {e}")
            return {'CANCELLED'}

        # Generate camera positions using Fibonacci sphere distribution
        points = fibonacci_sphere_points(num_cameras)
        temp_cameras = []
        camera_matrices = []

        self.report({'INFO'}, f"Creating {num_cameras} cameras...")

        for i, (x, y, z) in enumerate(points):
            # Determine radius for this camera
            if props.use_fixed_radius:
                # Use fixed radius from preview
                varied_radius = sphere_radius * props.fixed_radius_value
            else:
                # Apply random depth variation for each camera
                radius_variation = random.uniform(0.8, 1.0)
                varied_radius = sphere_radius * radius_variation

            cam_location = Vector((
                sphere_center.x + x * varied_radius,
                sphere_center.y + y * varied_radius,
                sphere_center.z + z * varied_radius
            ))

            cam = create_temp_camera(f"TempCam_{i:02d}", cam_location, sphere_center)
            temp_cameras.append(cam)

        # CRITICAL: Force Blender to update world matrices
        context.view_layer.update()

        # Now capture the updated matrices
        for cam in temp_cameras:
            camera_matrices.append(cam.matrix_world.copy())

        # Set frame range for rendering
        scene.frame_start = start_frame
        scene.frame_end = end_frame

        # Set scene to start frame BEFORE frustum culling
        # This ensures visibility is checked at the correct frame (where objects are positioned)
        scene.frame_set(start_frame)
        context.view_layer.update()

        # FRUSTUM CULLING: Hide objects NOT visible from ANY camera (union)
        # Always enabled - ensures only objects visible from cameras are rendered
        self.report({'INFO'}, "Culling objects not visible from any camera...")
        hidden_count, visible_count, visibility_map = cull_objects_outside_all_frustums(
            context, temp_cameras, scene
        )
        # Store original visibility for restoration
        props.original_object_visibility = json.dumps(visibility_map)
        self.report({'INFO'}, 
            f"Culled {hidden_count} objects. {visible_count} objects visible from at least one camera.")

        # Calculate intrinsics (same for all cameras)
        K, width, height = calculate_camera_intrinsics(scene)

        # Render loop
        try:
            for i in range(num_cameras):
                props.current_camera = i + 1  # 1-indexed for display
                props.camera_start_time = time.time()
                props.current_frame = 0

                # Set active camera
                scene.camera = temp_cameras[i]

                # Create camera directories
                cam_name = f"{i:02d}"
                cam_images_dir = os.path.join(images_dir, cam_name)
                os.makedirs(cam_images_dir, exist_ok=True)

                # Set color output path with frame number token
                color_path = os.path.join(cam_images_dir, "######.png")
                scene.render.filepath = color_path

                # Configure render settings
                scene.render.film_transparent = True
                scene.render.image_settings.file_format = 'PNG'
                scene.render.image_settings.color_mode = 'RGBA'

                # Render animation sequence (efficient batch rendering)
                self.report({'INFO'}, f"Rendering Camera {i + 1}/{num_cameras}...")
                bpy.ops.render.render(animation=True)

                # Rename color frames to start from 000000
                rename_frames_from_zero(cam_images_dir, start_frame, num_frames, ".png")

                # Camera completed - calculate timing
                camera_time = time.time() - props.camera_start_time
                self._camera_times.append(camera_time)

                # Store first camera time as baseline
                if i == 0:
                    props.first_camera_total_time = camera_time

                # Update overall progress
                props.render_progress = (i + 1) / num_cameras

                # Calculate overall ETA based on actual camera completion times
                if len(self._camera_times) > 0:
                    avg_camera_time = sum(self._camera_times) / len(self._camera_times)
                    remaining_cameras = num_cameras - (i + 1)
                    props.overall_eta = avg_camera_time * remaining_cameras

                # Reset for next camera
                props.current_frame = 0
                props.current_camera_eta = 0

                self.report({'INFO'}, f"Camera {i + 1}/{num_cameras} completed in {format_time(camera_time)}")

                # Update UI
                for area in context.screen.areas:
                    if area.type == 'VIEW_3D':
                        area.tag_redraw()

                # Allow Blender to process events
                bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

            # Export PLY files per frame (after all cameras rendered)
            self.report({'INFO'}, "Exporting PLY files per frame...")
            pointclouds_dir = os.path.join(output_base, "pointclouds")
            os.makedirs(pointclouds_dir, exist_ok=True)
            
            # Get ALL mesh objects (not filtered by initial frustum culling)
            # This ensures we can check visibility per frame for all objects
            all_mesh_objects = [obj for obj in scene.objects if obj.type == 'MESH']
            
            union_min = None
            union_max = None
            
            if all_mesh_objects:
                # FIRST PASS: Calculate union bounding box across all frames
                # Use all mesh objects for bounds calculation (will include objects from all frames)
                self.report({'INFO'}, "Calculating union bounding box across all frames...")
                union_min, union_max = calculate_union_bounds(
                    context, all_mesh_objects, start_frame, end_frame, apply_modifiers=True
                )
                
                if union_min and union_max:
                    union_size = union_max - union_min
                    self.report({'INFO'}, 
                        f"Union bounds: min=({union_min.x:.3f}, {union_min.y:.3f}, {union_min.z:.3f}), "
                        f"max=({union_max.x:.3f}, {union_max.y:.3f}, {union_max.z:.3f}), "
                        f"size=({union_size.x:.3f}, {union_size.y:.3f}, {union_size.z:.3f})")
                else:
                    union_min, union_max = None, None
                    self.report({'WARNING'}, "Could not calculate union bounds")
                
                # SECOND PASS: Export PLY for each frame (with animation)
                # Check visibility per frame to match what's actually rendered in images
                for frame_offset in range(num_frames):
                    target_frame_num = frame_offset
                    scene.frame_set(start_frame + frame_offset)
                    
                    # Update scene to current frame (important for animated objects)
                    context.view_layer.update()
                    
                    # Filter objects visible from ANY camera at THIS frame
                    # Check ALL mesh objects, not just those that passed initial culling
                    # This ensures objects visible at this frame are included, regardless of initial culling
                    frame_visible_objects = []
                    for obj in all_mesh_objects:
                        # Skip objects that are user-hidden (not frustum-culled)
                        if not obj.visible_get():
                            continue
                        
                        # Check if object is in frustum of ANY camera at this frame
                        is_visible_from_any = False
                        for cam in temp_cameras:
                            if is_object_in_camera_frustum(obj, cam, scene):
                                is_visible_from_any = True
                                break
                        
                        if is_visible_from_any:
                            frame_visible_objects.append(obj)
                    
                    # Export PLY file with only objects visible at this frame
                    ply_filename = f"frame_{target_frame_num:06d}.ply"
                    ply_path = os.path.join(pointclouds_dir, ply_filename)
                    
                    try:
                        # Use manual PLY export (works in all Blender versions)
                        success = export_meshes_to_ply(
                            context, 
                            frame_visible_objects,  # Filtered per frame
                            ply_path, 
                            apply_modifiers=True
                        )
                        
                        if success:
                            self.report({'INFO'}, 
                                f"Exported frame {target_frame_num} to {ply_filename} "
                                f"({len(frame_visible_objects)} objects visible)")
                        else:
                            self.report({'WARNING'}, f"No geometry exported for frame {target_frame_num}")
                    except Exception as e:
                        self.report({'WARNING'}, f"Failed to export PLY for frame {target_frame_num}: {e}")
                        import traceback
                        traceback.print_exc()
            else:
                self.report({'WARNING'}, "No mesh objects found for PLY export")

            # Export YAML files (ONCE after all cameras)
            self.report({'INFO'}, "Exporting camera parameters...")
            self.export_intrinsics(output_base, K, width, height, num_cameras)
            self.export_extrinsics(output_base, camera_matrices, num_cameras)

            # Create ZIP
            self.report({'INFO'}, "Creating ZIP file...")
            zip_path = output_base + ".zip"
            create_zip(output_base, zip_path)

            # Success message
            total_time = time.time() - props.overall_start_time
            self.report({'INFO'},
                        f"Render complete! {num_cameras} cameras × {num_frames} frames in {format_time(total_time)}")
            self.report({'INFO'}, f"Output: {zip_path}")

        finally:
            # Clean up
            props.is_rendering = False
            props.use_fixed_radius = False  # Reset fixed radius setting
            scene.camera = original_camera
            scene.render.film_transparent = original_transparent
            scene.render.image_settings.color_mode = original_color_mode
            scene.render.filepath = original_filepath
            scene.use_nodes = original_use_nodes
            if scene.use_nodes and scene.node_tree:
                scene.node_tree.nodes.clear()

            scene.render.resolution_x = original_res_x
            scene.render.resolution_y = original_res_y
            scene.render.resolution_percentage = original_resolution_percentage
            scene.frame_start = original_frame_start
            scene.frame_end = original_frame_end

            # Unregister frame handler (with error handling)
            try:
                if frame_change_handler in bpy.app.handlers.frame_change_pre:
                    bpy.app.handlers.frame_change_pre.remove(frame_change_handler)
            except Exception as e:
                self.report({'WARNING'}, f"Could not unregister frame handler: {e}")

            # Restore original object visibility (with error handling)
            if props.original_object_visibility:
                try:
                    visibility_map = json.loads(props.original_object_visibility)
                    for obj_name, was_hidden in visibility_map.items():
                        try:
                            obj = scene.objects.get(obj_name)
                            if obj:
                                obj.hide_render = was_hidden
                        except Exception as e:
                            # Continue with other objects even if one fails
                            self.report({'WARNING'}, f"Could not restore visibility for '{obj_name}': {e}")
                except Exception as e:
                    self.report({'WARNING'}, f"Could not restore object visibility: {e}")

            # Clean up temporary cameras (with error handling)
            for cam in temp_cameras:
                try:
                    if cam and cam.name in bpy.data.objects:
                        bpy.data.objects.remove(cam)
                except Exception as e:
                    # Log but continue cleanup
                    self.report({'WARNING'}, f"Could not remove temporary camera '{cam.name if cam else 'unknown'}': {e}")

        return {'FINISHED'}

    def validate_settings(self, context):
        """Validate render settings before starting"""
        errors = []
        props = context.scene.multi_cam_props

        if not props.camera_sphere or props.camera_sphere.type != 'EMPTY':
            errors.append("Camera sphere not found or invalid type")

        if props.end_frame < props.start_frame:
            errors.append("End frame must be greater than or equal to start frame")

        num_frames = props.end_frame - props.start_frame + 1

        # Validate and resolve output folder path
        output_folder, folder_error = validate_and_resolve_path(props.output_folder, context)
        if folder_error:
            errors.append(f"Output folder: {folder_error}")
        
        # Validate folder name
        if props.folder_name:
            sanitized = sanitize_filename(props.folder_name)
            if sanitized != props.folder_name:
                errors.append(f"Folder name contains invalid characters. Use: {sanitized}")

        # Check scene has renderable objects
        has_objects = any(obj.type == 'MESH' and obj.visible_get()
                          for obj in context.scene.objects)
        if not has_objects:
            errors.append("No renderable objects in scene")

        return errors

    def export_intrinsics(self, output_dir, K, width, height, num_cameras):
        """Export intrinsics YAML file"""
        intri_path = os.path.join(output_dir, "intri.yml")

        with open(intri_path, 'w') as f:
            f.write("%YAML:1.0\n---\n\n")

            # Write camera names
            f.write("names:\n")
            for i in range(num_cameras):
                f.write(f'  - "{i:02d}"\n')
            f.write("\n")

            # Write each camera's intrinsics (all identical)
            for i in range(num_cameras):
                cam_name = f"{i:02d}"

                # K matrix
                K_flat = [K[row][col] for row in range(3) for col in range(3)]
                write_opencv_matrix(f, f"K_{cam_name}", 3, 3, K_flat)

                # Image dimensions
                f.write(f"H_{cam_name}: {format_float(height)}\n")
                f.write(f"W_{cam_name}: {format_float(width)}\n")

                # Distortion coefficients (all zeros)
                write_opencv_matrix(f, f"D_{cam_name}", 5, 1, [0.0] * 5)

                # Color correction matrix (identity)
                ccm = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
                write_opencv_matrix(f, f"ccm_{cam_name}", 3, 3, ccm)

                f.write("\n")

    def export_extrinsics(self, output_dir, camera_matrices, num_cameras):
        """Export extrinsics YAML file in world-to-camera format"""
        extri_path = os.path.join(output_dir, "extri.yml")

        with open(extri_path, 'w') as f:
            f.write("%YAML:1.0\n---\n\n")

            # Write camera names
            f.write("names:\n")
            for i in range(num_cameras):
                f.write(f'  - "{i:02d}"\n')
            f.write("\n")

            # Define flip matrices for world-to-camera conversion
            R_y_flip = Matrix((
                (-1, 0, 0),
                (0, 1, 0),
                (0, 0, -1)
            ))

            R_z_flip = Matrix((
                (-1, 0, 0),
                (0, -1, 0),
                (0, 0, 1)
            ))

            R_combined_flip = R_z_flip @ R_y_flip

            # Write each camera's extrinsics
            for i, blender_matrix in enumerate(camera_matrices):
                cam_name = f"{i:02d}"

                # Convert to OpenCV convention (camera-to-world)
                opencv_matrix = blender_to_opencv_matrix(blender_matrix)

                # Extract rotation and translation (camera-to-world)
                R_cw = opencv_matrix.to_3x3()
                T_cw = opencv_matrix.translation

                # Invert to world-to-camera
                R_wc = R_cw.transposed()
                T_wc = -(R_wc @ T_cw)

                # Apply 180° Y and Z rotation flip
                R_final = R_wc @ R_combined_flip
                T_final = T_wc

                # Rodrigues vector
                rodrigues = matrix_to_rodrigues(R_final)
                write_opencv_matrix(f, f"R_{cam_name}", 3, 1, rodrigues)

                # Rotation matrix
                R_flat = [R_final[row][col] for row in range(3) for col in range(3)]
                write_opencv_matrix(f, f"Rot_{cam_name}", 3, 3, R_flat)

                # Translation vector
                write_opencv_matrix(f, f"T_{cam_name}", 3, 1, [T_final.x, T_final.y, T_final.z])

                # Time (always 0)
                f.write(f"t_{cam_name}: {format_float(0.0)}\n")

                # Near and far planes
                f.write(f"n_{cam_name}: {format_float(0.0001)}\n")
                f.write(f"f_{cam_name}: {format_float(1000000.0)}\n")

                # Bounds
                bounds = [-1000000.0, -1000000.0, -1000000.0,
                          1000000.0, 1000000.0, 1000000.0]
                write_opencv_matrix(f, f"bounds_{cam_name}", 2, 3, bounds)

                f.write("\n")

    def cancel(self, context):
        """Cancel rendering"""
        props = context.scene.multi_cam_props
        props.is_rendering = False
        self.report({'INFO'}, "Render cancelled")


# ============================================================================
# UI PANEL
# ============================================================================

class RENDER_PT_multi_camera(Panel):
    """Multi-Camera Render Panel"""
    bl_label = "Multi-Camera Render"
    bl_idname = "RENDER_PT_multi_camera"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Multi-Cam'

    def draw(self, context):
        layout = self.layout
        props = context.scene.multi_cam_props
        scene = context.scene

        # Camera Sphere Selection
        box = layout.box()
        box.label(text="Camera Sphere:", icon='SPHERE')
        box.prop(props, "camera_sphere", text="")
        box.operator("render.create_camera_sphere", icon='ADD')

        # Sphere info
        if props.camera_sphere:
            sphere = props.camera_sphere
            col = box.column(align=True)
            col.label(text=f"Radius: {sphere.empty_display_size:.2f}m")
            loc = sphere.location
            col.label(text=f"Center: ({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f})")

        layout.separator()

        # Camera Preview Section
        box = layout.box()
        box.label(text="Camera Preview:", icon='CAMERA_DATA')

        if props.camera_sphere:
            # Toggle preview button
            if props.preview_enabled:
                box.operator("render.toggle_camera_preview", text="Disable Preview", icon='HIDE_ON')

                # Preview controls
                col = box.column(align=True)

                # Radius slider
                row = col.row(align=True)
                row.label(text="Radius:")
                row.prop(props, "preview_radius_multiplier", text="", slider=True)

                # Display actual distance
                actual_dist = props.camera_sphere.empty_display_size * props.preview_radius_multiplier
                col.label(text=f"Distance: {actual_dist:.2f}m", icon='DRIVER_DISTANCE')

                # Camera position navigation
                row = col.row(align=True)
                row.label(text=f"Camera: {props.preview_camera_index + 1}/{props.num_cameras}")
                row.operator("render.preview_previous_camera", text="", icon='TRIA_LEFT')
                row.operator("render.preview_next_camera", text="", icon='TRIA_RIGHT')

                # Apply radius button
                col.separator()
                if props.preview_radius_multiplier != 1.0:
                    col.operator("render.apply_preview_radius",
                                 text=f"Use {props.preview_radius_multiplier:.2f}x for Render",
                                 icon='CHECKMARK')

                # Show if fixed radius is set
                if props.use_fixed_radius:
                    col.label(text=f"Render radius: {props.fixed_radius_value:.2f}x", icon='LOCKED')
            else:
                box.operator("render.toggle_camera_preview", text="Enable Preview", icon='HIDE_OFF')
                box.label(text="Preview disabled", icon='INFO')
        else:
            box.label(text="Create sphere first", icon='ERROR')

        layout.separator()

        # Render Settings
        box = layout.box()
        box.label(text="Render Settings:", icon='RENDER_STILL')
        box.prop(props, "num_cameras")

        # Frame range
        row = box.row(align=True)
        row.prop(props, "start_frame")
        row.prop(props, "end_frame")

        # Show frame count
        num_frames = max(0, props.end_frame - props.start_frame + 1)
        frame_count_row = box.row()
        frame_count_row.label(text=f"Total frames: {num_frames}")

        # Show radius setting
        if props.use_fixed_radius:
            box.label(text=f"Using fixed radius: {props.fixed_radius_value:.2f}x", icon='LOCKED')
        else:
            box.label(text="Using random radius variation (0.8-1.0x)", icon='MOD_NOISE')

        layout.separator()

        # Output Settings
        box = layout.box()
        box.label(text="Output:", icon='FILE_FOLDER')
        box.prop(props, "output_folder")
        box.prop(props, "folder_name")

        # Show resolved path
        resolved_path = bpy.path.abspath(props.output_folder)
        if resolved_path:
            col = box.column(align=True)
            col.label(text="Will save to:", icon='DISK_DRIVE')
            if len(resolved_path) > 40:
                col.label(text=resolved_path[:40] + "...")
                col.label(text="..." + resolved_path[-37:])
            else:
                col.label(text=resolved_path)

        layout.separator()

        # Main Render Button or Progress Display
        if not props.is_rendering:
            # Disable render button if preview is active
            row = layout.row()
            row.enabled = not props.preview_enabled
            row.operator("render.render_all_cameras",
                         text="▶ RENDER ALL CAMERAS",
                         icon='RENDER_ANIMATION')
            if props.preview_enabled:
                layout.label(text="Disable preview to render", icon='INFO')
        else:
            # Enhanced Progress Display
            box = layout.box()
            box.label(text="Rendering...", icon='RENDER_ANIMATION')

            # Line 1: Progress bar with percentage
            progress = props.render_progress
            col = box.column(align=True)

            # Create progress bar visualization
            progress_pct = int(progress * 100)
            bar_width = 20
            filled = int(progress * bar_width)
            empty = bar_width - filled
            progress_bar = "=" * filled + (">" if empty > 0 else "") + " " * empty
            col.label(text=f"Progress: [{progress_bar}] {progress_pct}%")

            # Line 2: Camera and frame progress (1-indexed)
            if props.current_frame > 0:
                col.label(
                    text=f"Camera {props.current_camera}/{props.num_cameras} | Frame {props.current_frame}/{props.total_frames}")
            else:
                col.label(text=f"Camera {props.current_camera}/{props.num_cameras} | Processing...")

            # Line 3: Current camera ETA
            if props.current_camera_eta > 0:
                camera_eta_text = f"Current camera ETA: {format_time(props.current_camera_eta)}"
            elif props.current_camera == 1:
                camera_eta_text = "Current camera ETA: Calculating..."
            else:
                camera_eta_text = "Current camera ETA: Processing..."
            col.label(text=camera_eta_text)

            # Line 4: Overall ETA
            if props.overall_eta > 0:
                overall_eta_text = f"TIME REMAINING ETA: {format_time(props.overall_eta)}"
            elif props.first_camera_total_time > 0:
                overall_eta_text = f"TIME REMAINING ETA: {format_time(props.overall_eta)}"
            else:
                overall_eta_text = "TIME REMAINING ETA: Calculating..."
            col.label(text=overall_eta_text)


# ============================================================================
# PROPERTIES
# ============================================================================

class MultiCamProperties(PropertyGroup):
    camera_sphere: PointerProperty(
        name="Camera Sphere",
        type=bpy.types.Object,
        description="Empty sphere defining camera positions"
    )

    num_cameras: IntProperty(
        name="Number of Cameras",
        description="Number of camera positions on sphere",
        default=60,
        min=1,
        max=100
    )

    start_frame: IntProperty(
        name="Start Frame",
        description="First frame to render",
        default=0,
        min=0
    )

    end_frame: IntProperty(
        name="End Frame",
        description="Last frame to render",
        default=1,
        min=0
    )

    output_folder: StringProperty(
        name="Output Folder",
        description="Base folder for output (use // for .blend file location)",
        default=os.path.expanduser("~/blender_renders"),
        subtype='DIR_PATH'
    )

    folder_name: StringProperty(
        name="Folder Name",
        description="Name of output subfolder",
        default="data"
    )

    # Preview properties
    preview_enabled: BoolProperty(
        name="Preview Enabled",
        description="Enable camera preview mode",
        default=False
    )

    preview_radius_multiplier: FloatProperty(
        name="Radius Multiplier",
        description="Multiply sphere radius by this value for preview",
        default=1.0,
        min=0.5,
        max=1.5,
        soft_min=0.7,
        soft_max=1.3,
        step=1,
        precision=2,
        update=update_preview_camera
    )

    preview_camera_index: IntProperty(
        name="Preview Camera Index",
        description="Which camera position to preview",
        default=0,
        min=0,
        update=update_preview_camera
    )

    original_camera: PointerProperty(
        name="Original Camera",
        type=bpy.types.Object,
        description="Store original camera to restore after preview"
    )

    use_fixed_radius: BoolProperty(
        name="Use Fixed Radius",
        description="Use a fixed radius value for all cameras instead of random variation",
        default=False
    )

    fixed_radius_value: FloatProperty(
        name="Fixed Radius Value",
        description="Fixed radius multiplier to use for all cameras",
        default=1.0,
        min=0.5,
        max=1.5
    )

    auto_cull_objects: BoolProperty(
        name="Auto-Hide Objects Outside Cameras",
        description="Hide objects that are NOT visible from ANY camera (union culling). Objects visible from at least one camera are kept.",
        default=True
    )

    original_object_visibility: StringProperty(
        name="Original Object Visibility",
        description="JSON string storing original hide_render state (internal use)",
        default="{}"
    )

    # Rendering state
    is_rendering: BoolProperty(
        name="Is Rendering",
        default=False
    )

    render_progress: FloatProperty(
        name="Render Progress",
        description="Current render progress",
        default=0.0,
        min=0.0,
        max=1.0,
        subtype='FACTOR'
    )

    # Enhanced progress tracking properties
    current_camera: IntProperty(
        name="Current Camera",
        description="Current camera being rendered (1-indexed)",
        default=0
    )

    current_frame: IntProperty(
        name="Current Frame",
        description="Current frame being rendered (1-indexed)",
        default=0
    )

    total_frames: IntProperty(
        name="Total Frames",
        description="Total frames per camera",
        default=0
    )

    frame_start_time: FloatProperty(
        name="Frame Start Time",
        description="Timestamp when rendering started for current camera",
        default=0.0
    )

    last_frame_time: FloatProperty(
        name="Last Frame Time",
        description="Not used with batch animation rendering",
        default=0.0
    )

    camera_start_time: FloatProperty(
        name="Camera Start Time",
        description="Timestamp when current camera started",
        default=0.0
    )

    first_camera_total_time: FloatProperty(
        name="First Camera Total Time",
        description="Total time for first camera (baseline)",
        default=0.0
    )

    current_camera_eta: FloatProperty(
        name="Current Camera ETA",
        description="Estimated time remaining for current camera",
        default=0.0
    )

    overall_eta: FloatProperty(
        name="Overall ETA",
        description="Estimated time remaining for all cameras",
        default=0.0
    )

    overall_start_time: FloatProperty(
        name="Overall Start Time",
        description="Timestamp when rendering started",
        default=0.0
    )



# ============================================================================
# REGISTRATION
# ============================================================================

classes = (
    MultiCamProperties,
    RENDER_OT_create_sphere,
    RENDER_OT_toggle_preview,
    RENDER_OT_preview_next,
    RENDER_OT_preview_previous,
    RENDER_OT_apply_preview_radius,
    RENDER_OT_render_all,
    RENDER_PT_multi_camera,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.multi_cam_props = PointerProperty(type=MultiCamProperties)


def unregister():
    # Clean up preview camera if it exists
    for obj in bpy.data.objects:
        if obj.name == "PreviewCamera_MultiCam":
            bpy.data.objects.remove(obj, do_unlink=True)

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

    del bpy.types.Scene.multi_cam_props


if __name__ == "__main__":
    register()