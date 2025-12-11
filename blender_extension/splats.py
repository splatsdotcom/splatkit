"""
Multi-Camera Render Addon for Blender
Renders scene from multiple camera positions on a sphere for 3D reconstruction

Author: Splats
GitHub: https://github.com/splatsdotcom
"""

bl_info = {
    "name": "Splats",
    "author": "Splats",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > Multi-Cam",
    "description": "Render multi-frame sequences from multiple camera positions",
    "category": "Render",
    "doc_url": "https://github.com/splatsdotcom",
    "tracker_url": "https://github.com/splatsdotcom",
}

import bpy
import math
import os
import re
import time
import zipfile
import json
import random
import traceback
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
# CONSTANTS
# ============================================================================

# Camera settings
DEFAULT_FOV_DEGREES = 50.0
CAMERA_CLIP_START = 0.01
CAMERA_CLIP_END = 1000.0
RADIUS_SAFETY_MARGIN = 2.0
RADIUS_VARIATION_MIN = 0.8
RADIUS_VARIATION_MAX = 1.0

# Render settings
DEFAULT_RESOLUTION_X = 3840
DEFAULT_RESOLUTION_Y = 2160

# Camera intrinsics defaults
DEFAULT_FOCAL_LENGTH_MM = 50.0
DEFAULT_SENSOR_WIDTH_MM = 36.0

# Extrinsics export defaults
NEAR_PLANE = 0.0001
FAR_PLANE = 1000000.0
BOUNDS_MIN = -1000000.0
BOUNDS_MAX = 1000000.0

# Other constants
MAX_FILENAME_LENGTH = 255
PROGRESS_BAR_WIDTH = 20

# Object names
PREVIEW_CAMERA_NAME = "PreviewCamera_MultiCam"
CAMERA_RIG_SPHERE_NAME = "CameraRig_Sphere"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

class SceneSettingsRestore:
    """Context manager to restore Blender scene settings."""
    def __init__(self, scene):
        self.scene = scene
        self.original_camera = scene.camera
        self.original_transparent = scene.render.film_transparent
        self.original_color_mode = scene.render.image_settings.color_mode
        self.original_filepath = scene.render.filepath
        self.original_use_nodes = scene.use_nodes
        self.original_res_x = scene.render.resolution_x
        self.original_res_y = scene.render.resolution_y
        self.original_frame_start = scene.frame_start
        self.original_frame_end = scene.frame_end
        self.original_resolution_percentage = scene.render.resolution_percentage
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.scene.camera = self.original_camera
        self.scene.render.film_transparent = self.original_transparent
        self.scene.render.image_settings.color_mode = self.original_color_mode
        self.scene.render.filepath = self.original_filepath
        self.scene.use_nodes = self.original_use_nodes
        if self.scene.use_nodes and self.scene.node_tree:
            self.scene.node_tree.nodes.clear()
        self.scene.render.resolution_x = self.original_res_x
        self.scene.render.resolution_y = self.original_res_y
        self.scene.render.resolution_percentage = self.original_resolution_percentage
        self.scene.frame_start = self.original_frame_start
        self.scene.frame_end = self.original_frame_end
        return False

def sanitize_filename(name):
    """Sanitize filename by removing invalid characters."""
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)
    sanitized = sanitized.strip(' .')
    sanitized = re.sub(r'_+', '_', sanitized)
    if not sanitized:
        sanitized = "untitled"
    if len(sanitized) > MAX_FILENAME_LENGTH:
        sanitized = sanitized[:MAX_FILENAME_LENGTH]
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
        resolved = bpy.path.abspath(path)
        
        if not os.path.isabs(resolved):
            return None, f"Path could not be resolved to absolute path: {path}"
        
        parent_dir = os.path.dirname(resolved)
        if parent_dir and not os.path.exists(parent_dir):
            try:
                os.makedirs(parent_dir, exist_ok=True)
            except (OSError, PermissionError) as e:
                return None, f"Cannot create parent directory: {e}"
        
        if os.path.exists(resolved):
            if not os.access(resolved, os.W_OK):
                return None, f"Path is not writable: {resolved}"
        else:
            if parent_dir and not os.access(parent_dir, os.W_OK):
                return None, f"Parent directory is not writable: {parent_dir}"
        
        return resolved, None
    except Exception as e:
        return None, f"Error validating path: {e}"


def fibonacci_sphere_points(n_points):
    """Generate evenly distributed points on unit sphere using Fibonacci spiral."""
    points = []
    phi = math.pi * (3.0 - math.sqrt(5.0))

    for i in range(n_points):
        if n_points == 1:
            y = 1.0
        else:
            y = 1 - (i / float(n_points - 1)) * 2.0
        radius_at_y = math.sqrt(1 - y * y)
        theta = phi * i

        x = math.cos(theta) * radius_at_y
        z = math.sin(theta) * radius_at_y
        points.append((x, y, z))

    return points


def _calculate_object_bounds(obj):
    """Calculate bounding box for a single object."""
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    min_co = Vector((float('inf'), float('inf'), float('inf')))
    max_co = Vector((float('-inf'), float('-inf'), float('-inf')))
    
    for corner in bbox_corners:
        min_co.x = min(min_co.x, corner.x)
        min_co.y = min(min_co.y, corner.y)
        min_co.z = min(min_co.z, corner.z)
        max_co.x = max(max_co.x, corner.x)
        max_co.y = max(max_co.y, corner.y)
        max_co.z = max(max_co.z, corner.z)
    
    return min_co, max_co


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
            obj_min, obj_max = _calculate_object_bounds(obj)
            min_co.x = min(min_co.x, obj_min.x)
            min_co.y = min(min_co.y, obj_min.y)
            min_co.z = min(min_co.z, obj_min.z)
            max_co.x = max(max_co.x, obj_max.x)
            max_co.y = max(max_co.y, obj_max.y)
            max_co.z = max(max_co.z, obj_max.z)

    if not has_objects:
        return None, None, None

    center = (min_co + max_co) / 2
    
    bbox_corners = [
        min_co,
        max_co,
        Vector((min_co.x, min_co.y, max_co.z)),
        Vector((min_co.x, max_co.y, min_co.z)),
        Vector((max_co.x, min_co.y, min_co.z)),
        Vector((min_co.x, max_co.y, max_co.z)),
        Vector((max_co.x, min_co.y, max_co.z)),
        Vector((max_co.x, max_co.y, min_co.z))
    ]
    max_distance = max((corner - center).length for corner in bbox_corners)

    return center, max_distance, (min_co, max_co)


def _check_point_in_frustum(point, cam_loc, cam_forward, cam_right, cam_up, 
                            clip_start, clip_end, cam_data, aspect, fov=None, half_width=None, half_height=None):
    """Helper function to check if a point is within camera frustum."""
    to_point = point - cam_loc
    depth = to_point.dot(cam_forward)
    
    if depth < clip_start or depth > clip_end:
        return False
    
    right_dist = to_point.dot(cam_right)
    up_dist = to_point.dot(cam_up)
    
    if cam_data.type == 'PERSP':
        frustum_width = 2 * depth * math.tan(fov / 2)
        frustum_height = frustum_width / aspect
        return abs(right_dist) < frustum_width / 2 and abs(up_dist) < frustum_height / 2
    else:
        return abs(right_dist) < half_width and abs(up_dist) < half_height


def is_object_in_camera_frustum(obj, camera_obj, scene):
    """Check if object's bounding box intersects with camera's view frustum."""
    cam_data = camera_obj.data
    cam_matrix = camera_obj.matrix_world
    
    cam_loc = cam_matrix.translation
    cam_forward = -cam_matrix.col[2].xyz.normalized()
    cam_right = cam_matrix.col[0].xyz.normalized()
    cam_up = cam_matrix.col[1].xyz.normalized()
    
    clip_start = cam_data.clip_start
    clip_end = cam_data.clip_end
    
    aspect = scene.render.resolution_x / scene.render.resolution_y
    
    if cam_data.type == 'PERSP':
        sensor_width = cam_data.sensor_width
        focal_length = cam_data.lens
        fov = 2 * math.atan(sensor_width / (2 * focal_length))
        half_width = None
        half_height = None
    else:
        ortho_scale = cam_data.ortho_scale
        half_width = ortho_scale / 2
        half_height = half_width / aspect
        fov = None
    
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    
    for corner in bbox_corners:
        if _check_point_in_frustum(corner, cam_loc, cam_forward, cam_right, cam_up,
                                   clip_start, clip_end, cam_data, aspect, fov, half_width, half_height):
            return True
    
    obj_center = obj.matrix_world.translation
    return _check_point_in_frustum(obj_center, cam_loc, cam_forward, cam_right, cam_up,
                                   clip_start, clip_end, cam_data, aspect, fov, half_width, half_height)


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
                else:
                    obj_matrix = obj.matrix_world
                
                bbox_corners = [obj_matrix @ Vector(corner) for corner in obj.bound_box]
                
                if bbox_corners:
                    obj_mins = [min(c[i] for c in bbox_corners) for i in range(3)]
                    obj_maxs = [max(c[i] for c in bbox_corners) for i in range(3)]
                    all_mins.append(obj_mins)
                    all_maxs.append(obj_maxs)
            except (AttributeError, RuntimeError, ValueError) as e:
                continue
    
    if not all_mins:
        return None, None
    
    min_co = Vector([min(m[i] for m in all_mins) for i in range(3)])
    max_co = Vector([max(m[i] for m in all_maxs) for i in range(3)])
    
    return min_co, max_co


def export_meshes_to_ply(context, mesh_objects, output_path, apply_modifiers=True):
    """
    Manually export Blender mesh objects to PLY file.
    
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
    depsgraph = context.evaluated_depsgraph_get() if apply_modifiers else None
    
    for obj in mesh_objects:
        if obj.type != 'MESH':
            continue
            
        try:
            if apply_modifiers and depsgraph:
                eval_obj = obj.evaluated_get(depsgraph)
                mesh = eval_obj.data
                obj_matrix = eval_obj.matrix_world
            else:
                mesh = obj.data
                obj_matrix = obj.matrix_world
            
            mesh.update()
            
            if len(mesh.vertices) == 0:
                continue
            
            for vertex in mesh.vertices:
                world_vertex = obj_matrix @ vertex.co
                all_vertices.append((world_vertex.x, world_vertex.y, world_vertex.z))
            
            mesh.calc_loop_triangles()
            for triangle in mesh.loop_triangles:
                face_verts = [triangle.vertices[0] + vertex_offset,
                             triangle.vertices[1] + vertex_offset,
                             triangle.vertices[2] + vertex_offset]
                all_faces.append(face_verts)
            
            vertex_offset += len(mesh.vertices)
        except (AttributeError, RuntimeError, ValueError, KeyError) as e:
            continue
    
    if len(all_vertices) == 0:
        return False
    
    num_vertices = len(all_vertices)
    num_faces = len(all_faces)
    
    try:
        with open(output_path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {num_vertices}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write(f"element face {num_faces}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            
            for vertex in all_vertices:
                f.write(f"{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
            
            for face in all_faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
        
        return True
    except (IOError, OSError, PermissionError) as e:
        traceback.print_exc()
        return False


def cull_objects_outside_all_frustums(context, cameras, scene):
    """
    Hide objects that are NOT visible from ANY camera (union logic).
    
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
    visible_from_any = set()
    mesh_objects = [obj for obj in scene.objects if obj.type == 'MESH' and obj.visible_get()]
    
    for cam in cameras:
        for obj in mesh_objects:
            if is_object_in_camera_frustum(obj, cam, scene):
                visible_from_any.add(obj.name)
    
    visibility_map = {}
    hidden_count = 0
    
    for obj in scene.objects:
        if obj.type == 'MESH':
            visibility_map[obj.name] = obj.hide_render
            if obj.name not in visible_from_any and not obj.hide_render:
                obj.hide_render = True
                hidden_count += 1
    
    return hidden_count, len(visible_from_any), visibility_map


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
        focal_length_mm = DEFAULT_FOCAL_LENGTH_MM
        sensor_width_mm = DEFAULT_SENSOR_WIDTH_MM

    width = scene.render.resolution_x
    height = scene.render.resolution_y
    aspect_ratio = width / height
    sensor_height_mm = sensor_width_mm / aspect_ratio

    fx = (focal_length_mm / sensor_width_mm) * width
    fy = (focal_length_mm / sensor_height_mm) * height

    cx = width / 2.0
    cy = height / 2.0

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
    R_np = np.array([[R[i][j] for j in range(3)] for i in range(3)])

    trace = np.trace(R_np)
    theta = np.arccos(np.clip((trace - 1) / 2, -1, 1))

    if theta < 1e-6:
        return [0.0, 0.0, 0.0]

    if abs(theta - np.pi) < 1e-6:
        i = np.argmax([R_np[0, 0], R_np[1, 1], R_np[2, 2]])
        axis = np.zeros(3)
        axis[i] = np.sqrt((R_np[i, i] + 1) / 2)
        for j in range(3):
            if i != j:
                axis[j] = R_np[i, j] / (2 * axis[i])
    else:
        axis = np.array([
            R_np[2, 1] - R_np[1, 2],
            R_np[0, 2] - R_np[2, 0],
            R_np[1, 0] - R_np[0, 1]
        ]) / (2 * np.sin(theta))

    rodrigues = axis * theta
    return rodrigues.tolist()


def format_float(value):
    """Format float with 10 decimal places for YAML output."""
    return f"{float(value):.10f}"


def format_camera_name(index):
    """Format camera index as zero-padded 2-digit string."""
    return f"{index:02d}"


def write_opencv_matrix(f, name, rows, cols, data):
    """Write OpenCV matrix format to YAML file."""
    f.write(f"{name}: !!opencv-matrix\n")
    f.write(f"  rows: {rows}\n")
    f.write(f"  cols: {cols}\n")
    f.write(f"  dt: d\n")
    f.write(f"  data: [")
    f.write(", ".join([format_float(v) for v in data]))
    f.write("]\n")


def write_yaml_header(f, num_cameras):
    """Write YAML file header with camera names list."""
    f.write("%YAML:1.0\n---\n\n")
    f.write("names:\n")
    for i in range(num_cameras):
        f.write(f'  - "{format_camera_name(i)}"\n')
    f.write("\n")


def create_temp_camera(name, location, look_at, context):
    """Create a temporary camera object at the specified location, pointing at a target."""
    cam_data = bpy.data.cameras.new(name=name)
    cam_obj = bpy.data.objects.new(name=name, object_data=cam_data)

    cam_data.clip_start = CAMERA_CLIP_START
    cam_data.clip_end = CAMERA_CLIP_END

    context.scene.collection.objects.link(cam_obj)

    cam_obj.location = location
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
    
    Args:
        directory (str): Directory containing the frame files
        start_frame_blender (int): Starting frame number in Blender
        num_frames (int): Number of frames to rename
        extension (str): File extension (e.g., ".png", ".exr")
    """
    for frame_offset in range(num_frames):
        blender_frame_num = start_frame_blender + frame_offset
        target_frame_num = frame_offset

        original_name = f"{blender_frame_num:06d}{extension}"
        original_path = os.path.join(directory, original_name)
        target_name = f"{target_frame_num:06d}{extension}"
        target_path = os.path.join(directory, target_name)

        if os.path.exists(original_path) and original_path != target_path:
            try:
                os.rename(original_path, target_path)
            except (OSError, PermissionError):
                pass


# ============================================================================
# PREVIEW FUNCTIONS
# ============================================================================

def update_preview_camera(self, context):
    """Update preview camera position when properties change."""
    if context is None:
        context = bpy.context
    props = context.scene.multi_cam_props

    if not props.preview_enabled or not props.camera_sphere:
        return

    preview_cam = None
    for obj in bpy.data.objects:
        if obj.name == PREVIEW_CAMERA_NAME:
            preview_cam = obj
            break

    if not preview_cam:
        cam_data = bpy.data.cameras.new(PREVIEW_CAMERA_NAME)
        preview_cam = bpy.data.objects.new(PREVIEW_CAMERA_NAME, cam_data)
        context.scene.collection.objects.link(preview_cam)

    sphere_center = props.camera_sphere.location
    sphere_radius = props.camera_sphere.empty_display_size
    points = fibonacci_sphere_points(props.num_cameras)

    cam_index = min(props.preview_camera_index, len(points) - 1)
    x, y, z = points[cam_index]
    actual_radius = sphere_radius * props.preview_radius_multiplier

    cam_location = Vector((
        sphere_center.x + x * actual_radius,
        sphere_center.y + y * actual_radius,
        sphere_center.z + z * actual_radius
    ))

    preview_cam.location = cam_location
    direction = sphere_center - cam_location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    preview_cam.rotation_euler = rot_quat.to_euler()

    context.scene.camera = preview_cam

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
        frame_offset = scene.frame_current - props.start_frame
        props.current_frame = frame_offset + 1

        elapsed = time.time() - props.camera_start_time

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
        center = None
        max_distance = None
        
        if context.selected_objects:
            selected_obj = context.selected_objects[0]
            if selected_obj.type == 'MESH':
                min_co, max_co = _calculate_object_bounds(selected_obj)
                center = (min_co + max_co) / 2
                bbox_corners = [min_co, max_co,
                               Vector((min_co.x, min_co.y, max_co.z)),
                               Vector((min_co.x, max_co.y, min_co.z)),
                               Vector((max_co.x, min_co.y, min_co.z)),
                               Vector((min_co.x, max_co.y, max_co.z)),
                               Vector((max_co.x, min_co.y, max_co.z)),
                               Vector((max_co.x, max_co.y, min_co.z))]
                max_distance = max((corner - center).length for corner in bbox_corners)
        
        if center is None:
            center, max_distance, _ = get_scene_bounds(context)

        if center is None:
            self.report({'ERROR'}, "No renderable objects in scene or selected object")
            return {'CANCELLED'}

        camera = context.scene.camera
        if camera and camera.type == 'CAMERA':
            fov_rad = camera.data.angle
        else:
            fov_rad = math.radians(DEFAULT_FOV_DEGREES)

        radius = (max_distance / math.tan(fov_rad / 2)) * RADIUS_SAFETY_MARGIN

        bpy.ops.object.empty_add(type='SPHERE', location=center)
        sphere = context.active_object
        sphere.name = CAMERA_RIG_SPHERE_NAME
        sphere.empty_display_size = radius

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

        props.preview_enabled = not props.preview_enabled

        if props.preview_enabled:
            props.original_camera = context.scene.camera
            update_preview_camera(None, context)
            self.report({'INFO'}, "Preview enabled - use slider to adjust radius")
        else:
            if props.original_camera:
                context.scene.camera = props.original_camera

            for obj in bpy.data.objects:
                if obj.name == PREVIEW_CAMERA_NAME:
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

        props.preview_camera_index = (props.preview_camera_index - 1) % props.num_cameras

        return {'FINISHED'}


class RENDER_OT_apply_preview_radius(Operator):
    """Apply the preview radius to be used in final render (sets render variation to this fixed value)"""
    bl_idname = "render.apply_preview_radius"
    bl_label = "Use This Radius"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.multi_cam_props

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

    _camera_times = []

    def execute(self, context):
        props = context.scene.multi_cam_props
        scene = context.scene

        if props.preview_enabled:
            bpy.ops.render.toggle_camera_preview()

        # Validation
        errors = self.validate_settings(context)
        if errors:
            self.report({'ERROR'}, "\n".join(errors))
            return {'CANCELLED'}

        scene.unit_settings.system = 'METRIC'
        scene.unit_settings.length_unit = 'METERS'

        settings_restore = SceneSettingsRestore(scene)
        scene.render.resolution_x = DEFAULT_RESOLUTION_X
        scene.render.resolution_y = DEFAULT_RESOLUTION_Y
        scene.render.resolution_percentage = 100

        sphere_center = props.camera_sphere.location
        sphere_radius = props.camera_sphere.empty_display_size
        num_cameras = props.num_cameras
        start_frame = props.start_frame
        end_frame = props.end_frame
        num_frames = end_frame - start_frame + 1

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

        if frame_change_handler not in bpy.app.handlers.frame_change_pre:
            bpy.app.handlers.frame_change_pre.append(frame_change_handler)

        output_folder, folder_error = validate_and_resolve_path(props.output_folder, context)
        if folder_error:
            self._log_error(f"Output folder error: {folder_error}")
            return {'CANCELLED'}
        
        safe_folder_name = sanitize_filename(props.folder_name)
        output_base = os.path.join(output_folder, safe_folder_name)
        images_dir = os.path.join(output_base, "images")
        
        try:
            os.makedirs(output_base, exist_ok=True)
        except (OSError, PermissionError) as e:
            self._log_error(f"Cannot create output directory: {e}", exc=e, print_traceback=True)
            return {'CANCELLED'}

        points = fibonacci_sphere_points(num_cameras)
        temp_cameras = []
        camera_matrices = []

        self._log_info(f"Creating {num_cameras} cameras...")

        for i, (x, y, z) in enumerate(points):
            if props.use_fixed_radius:
                varied_radius = sphere_radius * props.fixed_radius_value
            else:
                radius_variation = random.uniform(RADIUS_VARIATION_MIN, RADIUS_VARIATION_MAX)
                varied_radius = sphere_radius * radius_variation

            cam_location = Vector((
                sphere_center.x + x * varied_radius,
                sphere_center.y + y * varied_radius,
                sphere_center.z + z * varied_radius
            ))

            cam = create_temp_camera(f"TempCam_{format_camera_name(i)}", cam_location, sphere_center, context)
            temp_cameras.append(cam)

        context.view_layer.update()

        for cam in temp_cameras:
            camera_matrices.append(cam.matrix_world.copy())

        scene.frame_start = start_frame
        scene.frame_end = end_frame

        scene.frame_set(start_frame)
        context.view_layer.update()

        self._log_info("Culling objects not visible from any camera...")
        hidden_count, visible_count, visibility_map = cull_objects_outside_all_frustums(
            context, temp_cameras, scene
        )
        props.original_object_visibility = json.dumps(visibility_map)
        self._log_info(
            f"Culled {hidden_count} objects. {visible_count} objects visible from at least one camera."
        )

        K, width, height = calculate_camera_intrinsics(scene)

        try:
            for i in range(num_cameras):
                self._render_camera_sequence(
                    context, scene, temp_cameras[i], i, num_cameras,
                    images_dir, start_frame, num_frames, props
                )

            self._export_ply_files(
                context, scene, temp_cameras, output_base,
                start_frame, end_frame, num_frames
            )

            self._log_info("Exporting camera parameters...")
            self.export_intrinsics(output_base, K, width, height, num_cameras)
            self.export_extrinsics(output_base, camera_matrices, num_cameras)

            self._log_info("Creating ZIP file...")
            zip_path = output_base + ".zip"
            create_zip(output_base, zip_path)

            total_time = time.time() - props.overall_start_time
            self._log_info(
                f"Render complete! {num_cameras} cameras × {num_frames} frames in {format_time(total_time)}"
            )
            self._log_info(f"Output: {zip_path}")

        finally:
            props.is_rendering = False
            props.use_fixed_radius = False
            settings_restore.__exit__(None, None, None)

            try:
                if frame_change_handler in bpy.app.handlers.frame_change_pre:
                    bpy.app.handlers.frame_change_pre.remove(frame_change_handler)
            except (AttributeError, ValueError, RuntimeError) as e:
                self._log_warning(f"Could not unregister frame handler: {e}", exc=e)

            if props.original_object_visibility:
                try:
                    visibility_map = json.loads(props.original_object_visibility)
                    for obj_name, was_hidden in visibility_map.items():
                        try:
                            obj = scene.objects.get(obj_name)
                            if obj:
                                obj.hide_render = was_hidden
                        except (AttributeError, KeyError, RuntimeError) as e:
                            self._log_warning(
                                f"Could not restore visibility for '{obj_name}': {e}", exc=e
                            )
                except (json.JSONDecodeError, TypeError, ValueError) as e:
                    self._log_warning(f"Could not restore object visibility: {e}", exc=e)

            for cam in temp_cameras:
                try:
                    if cam and cam.name in bpy.data.objects:
                        bpy.data.objects.remove(cam)
                except (AttributeError, KeyError, RuntimeError) as e:
                    self._log_warning(
                        f"Could not remove temporary camera '{cam.name if cam else 'unknown'}': {e}",
                        exc=e,
                    )

        return {'FINISHED'}

    def _render_camera_sequence(self, context, scene, camera_obj, camera_index, num_cameras,
                                images_dir, start_frame, num_frames, props):
        """Render animation sequence for a single camera."""
        props.current_camera = camera_index + 1
        props.camera_start_time = time.time()
        props.current_frame = 0

        scene.camera = camera_obj

        cam_name = format_camera_name(camera_index)
        cam_images_dir = os.path.join(images_dir, cam_name)
        os.makedirs(cam_images_dir, exist_ok=True)

        color_path = os.path.join(cam_images_dir, "######.png")
        scene.render.filepath = color_path

        scene.render.film_transparent = True
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGBA'

        self._log_info(f"Rendering Camera {camera_index + 1}/{num_cameras}...")
        bpy.ops.render.render(animation=True)

        rename_frames_from_zero(cam_images_dir, start_frame, num_frames, ".png")

        camera_time = time.time() - props.camera_start_time
        self._camera_times.append(camera_time)

        if camera_index == 0:
            props.first_camera_total_time = camera_time

        props.render_progress = (camera_index + 1) / num_cameras

        if len(self._camera_times) > 0:
            avg_camera_time = sum(self._camera_times) / len(self._camera_times)
            remaining_cameras = num_cameras - (camera_index + 1)
            props.overall_eta = avg_camera_time * remaining_cameras

        props.current_frame = 0
        props.current_camera_eta = 0

        self._log_info(
            f"Camera {camera_index + 1}/{num_cameras} completed in {format_time(camera_time)}"
        )

        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()

        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

    def _export_ply_files(self, context, scene, temp_cameras, output_base,
                         start_frame, end_frame, num_frames):
        """Export PLY point cloud files for each frame."""
        self._log_info("Exporting PLY files per frame...")
        pointclouds_dir = os.path.join(output_base, "pointclouds")
        os.makedirs(pointclouds_dir, exist_ok=True)
        
        all_mesh_objects = [obj for obj in scene.objects if obj.type == 'MESH']
        
        if not all_mesh_objects:
            self._log_warning("No mesh objects found for PLY export")
            return
        
        self._log_info("Calculating union bounding box across all frames...")
        union_min, union_max = calculate_union_bounds(
            context, all_mesh_objects, start_frame, end_frame, apply_modifiers=True
        )
        
        if union_min and union_max:
            union_size = union_max - union_min
            self._log_info(
                f"Union bounds: min=({union_min.x:.3f}, {union_min.y:.3f}, {union_min.z:.3f}), "
                f"max=({union_max.x:.3f}, {union_max.y:.3f}, {union_max.z:.3f}), "
                f"size=({union_size.x:.3f}, {union_size.y:.3f}, {union_size.z:.3f})"
            )
        
        for frame_offset in range(num_frames):
            scene.frame_set(start_frame + frame_offset)
            context.view_layer.update()
            
            frame_visible_objects = [
                obj for obj in all_mesh_objects
                if obj.visible_get() and any(
                    is_object_in_camera_frustum(obj, cam, scene) for cam in temp_cameras
                )
            ]
            
            ply_filename = f"{frame_offset:06d}.ply"
            ply_path = os.path.join(pointclouds_dir, ply_filename)
            
            try:
                success = export_meshes_to_ply(
                    context, 
                    frame_visible_objects,
                    ply_path, 
                    apply_modifiers=True
                )
                
                if success:
                    self._log_info(
                        f"Exported frame {frame_offset} to {ply_filename} "
                        f"({len(frame_visible_objects)} objects visible)"
                    )
                else:
                    self._log_warning(f"No geometry exported for frame {frame_offset}")
            except (IOError, OSError, PermissionError, RuntimeError) as e:
                self._log_warning(
                    f"Failed to export PLY for frame {frame_offset}: {e}", exc=e, print_traceback=True
                )
    
    def _log_info(self, message: str) -> None:
        """Log an informational message via Blender's reporting system."""
        self.report({'INFO'}, message)

    def _log_warning(self, message: str, exc: Exception | None = None, print_traceback: bool = False) -> None:
        """Log a warning message, optionally printing a traceback."""
        self.report({'WARNING'}, message)
        if print_traceback and exc is not None:
            traceback.print_exc()

    def _log_error(self, message: str, exc: Exception | None = None, print_traceback: bool = False) -> None:
        """Log an error message, optionally printing a traceback."""
        self.report({'ERROR'}, message)
        if print_traceback and exc is not None:
            traceback.print_exc()

    def validate_settings(self, context):
        """Validate render settings before starting"""
        errors = []
        props = context.scene.multi_cam_props

        if not props.camera_sphere or props.camera_sphere.type != 'EMPTY':
            errors.append("Camera sphere not found or invalid type")

        if props.end_frame < props.start_frame:
            errors.append("End frame must be greater than or equal to start frame")

        output_folder, folder_error = validate_and_resolve_path(props.output_folder, context)
        if folder_error:
            errors.append(f"Output folder: {folder_error}")
        
        if props.folder_name:
            sanitized = sanitize_filename(props.folder_name)
            if sanitized != props.folder_name:
                errors.append(f"Folder name contains invalid characters. Use: {sanitized}")

        has_objects = any(obj.type == 'MESH' and obj.visible_get()
                          for obj in context.scene.objects)
        if not has_objects:
            errors.append("No renderable objects in scene")

        return errors

    def export_intrinsics(self, output_dir, K, width, height, num_cameras):
        """Export intrinsics YAML file"""
        intri_path = os.path.join(output_dir, "intri.yml")

        with open(intri_path, 'w') as f:
            write_yaml_header(f, num_cameras)

            for i in range(num_cameras):
                cam_name = format_camera_name(i)

                K_flat = [K[row][col] for row in range(3) for col in range(3)]
                write_opencv_matrix(f, f"K_{cam_name}", 3, 3, K_flat)

                f.write(f"H_{cam_name}: {format_float(height)}\n")
                f.write(f"W_{cam_name}: {format_float(width)}\n")

                write_opencv_matrix(f, f"D_{cam_name}", 5, 1, [0.0] * 5)

                ccm = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
                write_opencv_matrix(f, f"ccm_{cam_name}", 3, 3, ccm)

                f.write("\n")

    def export_extrinsics(self, output_dir, camera_matrices, num_cameras):
        """Export extrinsics YAML file in world-to-camera format"""
        extri_path = os.path.join(output_dir, "extri.yml")

        with open(extri_path, 'w') as f:
            write_yaml_header(f, num_cameras)

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

            for i, blender_matrix in enumerate(camera_matrices):
                cam_name = format_camera_name(i)

                opencv_matrix = blender_to_opencv_matrix(blender_matrix)

                R_cw = opencv_matrix.to_3x3()
                T_cw = opencv_matrix.translation

                R_wc = R_cw.transposed()
                T_wc = -(R_wc @ T_cw)

                R_final = R_wc @ R_combined_flip
                T_final = T_wc

                rodrigues = matrix_to_rodrigues(R_final)
                write_opencv_matrix(f, f"R_{cam_name}", 3, 1, rodrigues)

                R_flat = [R_final[row][col] for row in range(3) for col in range(3)]
                write_opencv_matrix(f, f"Rot_{cam_name}", 3, 3, R_flat)

                write_opencv_matrix(f, f"T_{cam_name}", 3, 1, [T_final.x, T_final.y, T_final.z])

                f.write(f"t_{cam_name}: {format_float(0.0)}\n")

                f.write(f"n_{cam_name}: {format_float(NEAR_PLANE)}\n")
                f.write(f"f_{cam_name}: {format_float(FAR_PLANE)}\n")

                bounds = [BOUNDS_MIN, BOUNDS_MIN, BOUNDS_MIN,
                          BOUNDS_MAX, BOUNDS_MAX, BOUNDS_MAX]
                write_opencv_matrix(f, f"bounds_{cam_name}", 2, 3, bounds)

                f.write("\n")


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

        box = layout.box()
        box.label(text="Camera Sphere:", icon='SPHERE')
        box.prop(props, "camera_sphere", text="")
        box.operator("render.create_camera_sphere", icon='ADD')

        if props.camera_sphere:
            sphere = props.camera_sphere
            col = box.column(align=True)
            col.label(text=f"Radius: {sphere.empty_display_size:.2f}m")
            loc = sphere.location
            col.label(text=f"Center: ({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f})")

        layout.separator()

        box = layout.box()
        box.label(text="Camera Preview:", icon='CAMERA_DATA')

        if props.camera_sphere:
            if props.preview_enabled:
                box.operator("render.toggle_camera_preview", text="Disable Preview", icon='HIDE_ON')

                col = box.column(align=True)

                row = col.row(align=True)
                row.label(text="Radius:")
                row.prop(props, "preview_radius_multiplier", text="", slider=True)

                actual_dist = props.camera_sphere.empty_display_size * props.preview_radius_multiplier
                col.label(text=f"Distance: {actual_dist:.2f}m", icon='DRIVER_DISTANCE')

                row = col.row(align=True)
                row.label(text=f"Camera: {props.preview_camera_index + 1}/{props.num_cameras}")
                row.operator("render.preview_previous_camera", text="", icon='TRIA_LEFT')
                row.operator("render.preview_next_camera", text="", icon='TRIA_RIGHT')

                col.separator()
                if props.preview_radius_multiplier != 1.0:
                    col.operator("render.apply_preview_radius",
                                 text=f"Use {props.preview_radius_multiplier:.2f}x for Render",
                                 icon='CHECKMARK')

                if props.use_fixed_radius:
                    col.label(text=f"Render radius: {props.fixed_radius_value:.2f}x", icon='LOCKED')
            else:
                box.operator("render.toggle_camera_preview", text="Enable Preview", icon='HIDE_OFF')
                box.label(text="Preview disabled", icon='INFO')
        else:
            box.label(text="Create sphere first", icon='ERROR')

        layout.separator()

        box = layout.box()
        box.label(text="Render Settings:", icon='RENDER_STILL')
        box.prop(props, "num_cameras")

        row = box.row(align=True)
        row.prop(props, "start_frame")
        row.prop(props, "end_frame")

        num_frames = max(0, props.end_frame - props.start_frame + 1)
        frame_count_row = box.row()
        frame_count_row.label(text=f"Total frames: {num_frames}")

        if props.use_fixed_radius:
            box.label(text=f"Using fixed radius: {props.fixed_radius_value:.2f}x", icon='LOCKED')
        else:
            box.label(text="Using random radius variation (0.8-1.0x)", icon='MOD_NOISE')

        layout.separator()

        box = layout.box()
        box.label(text="Output:", icon='FILE_FOLDER')
        box.prop(props, "output_folder")
        box.prop(props, "folder_name")

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

        if not props.is_rendering:
            row = layout.row()
            row.enabled = not props.preview_enabled
            row.operator("render.render_all_cameras",
                         text="▶ RENDER ALL CAMERAS",
                         icon='RENDER_ANIMATION')
            if props.preview_enabled:
                layout.label(text="Disable preview to render", icon='INFO')
        else:
            box = layout.box()
            box.label(text="Rendering...", icon='RENDER_ANIMATION')

            progress = props.render_progress
            col = box.column(align=True)

            progress_pct = int(progress * 100)
            filled = int(progress * PROGRESS_BAR_WIDTH)
            empty = PROGRESS_BAR_WIDTH - filled
            progress_bar = "=" * filled + (">" if empty > 0 else "") + " " * empty
            col.label(text=f"Progress: [{progress_bar}] {progress_pct}%")

            if props.current_frame > 0:
                col.label(
                    text=f"Camera {props.current_camera}/{props.num_cameras} | Frame {props.current_frame}/{props.total_frames}")
            else:
                col.label(text=f"Camera {props.current_camera}/{props.num_cameras} | Processing...")

            if props.current_camera_eta > 0:
                camera_eta_text = f"Current camera ETA: {format_time(props.current_camera_eta)}"
            elif props.current_camera == 1:
                camera_eta_text = "Current camera ETA: Calculating..."
            else:
                camera_eta_text = "Current camera ETA: Processing..."
            col.label(text=camera_eta_text)

            if props.overall_eta > 0:
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
        default=24,
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

    original_object_visibility: StringProperty(
        name="Original Object Visibility",
        description="JSON string storing original hide_render state (internal use)",
        default="{}"
    )

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
    for obj in bpy.data.objects:
        if obj.name == PREVIEW_CAMERA_NAME:
            bpy.data.objects.remove(obj, do_unlink=True)

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

    del bpy.types.Scene.multi_cam_props


if __name__ == "__main__":
    register()