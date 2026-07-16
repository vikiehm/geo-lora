from time import time
import igl
from scipy.spatial.distance import cdist
import trimesh


import open3d as o3d
import numpy as np


class RayCaster:
    def __init__(self, width=300, height=300):
        self.width = width
        self.height = height
        self.up = [0, 1, 0]
        self.fov_deg = 90

    def get_rays(self, center, eye):
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            fov_deg=self.fov_deg,
            center=center,
            eye=eye,
            up=self.up,
            width_px=self.width,
            height_px=self.height,
        )
        return rays

    def ray_casting(self, mesh, camera_location):
        pnts = mesh.vertex.positions.numpy()
        diag_extent = igl.bounding_box_diagonal(pnts)
        mesh_updated = False
        if diag_extent >= 2:
            pnts = 0.8 * pnts
            mesh_v2 = o3d.geometry.TriangleMesh()
            mesh_v2.vertices = o3d.utility.Vector3dVector(pnts)
            mesh_v2.triangles = o3d.utility.Vector3iVector(
                mesh.triangle.indices.numpy()
            )
            mesh_v2 = o3d.t.geometry.TriangleMesh.from_legacy(mesh_v2)
            mesh_updated = True
        center = np.array([0.0, 0.0, 0.0])
        scene = o3d.t.geometry.RaycastingScene()
        if mesh_updated:
            scene.add_triangles(mesh_v2)
        else:
            scene.add_triangles(mesh)
        rays = self.get_rays(center=center, eye=camera_location)
        ans = scene.cast_rays(rays)
        return ans, mesh_updated

    def ray_casting_trimesh(self, vertices, faces, camera_location):
        """
        Ray casting implementation using trimesh library
        """
        # Create trimesh object
        # mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        bound_min, bound_max = np.array([vertices.min(axis=0), vertices.max(axis=0)])

        # Scale if needed (same logic as original)
        diag_extent = np.linalg.norm(bound_max - bound_min)
        if diag_extent >= 2:
            vertices = 0.8 * vertices

        rays = self._generate_camera_rays_trimesh(
            center=np.array([0.0, 0.0, 0.0]), eye=camera_location
        )
        rays_org = rays[:, :, :3].numpy()
        rays_dir = rays[:, :, 3:].numpy()

        # stack to (N, 3)
        rays_org = rays_org.reshape(-1, 3)
        rays_dir = rays_dir.reshape(-1, 3)

        start_rays = time.time()
        all_triangles = []
        for i in range(len(rays_dir)):
            hit = igl.ray_mesh_intersect(
                rays_org[i, :], rays_dir[i, :], vertices, faces
            )
            if len(hit) > 0:
                for x in hit[0]:
                    all_triangles.append(x)
                    break
        end_rays = time.time()
        print(f"Trimesh ray intersection time: {end_rays - start_rays:.4f} seconds")
        # Get unique triangle IDs that were hit
        unique_triangle_ids = np.unique(all_triangles)

        return unique_triangle_ids

    def _generate_camera_rays_trimesh(self, center, eye):
        """Generate rays optimized for trimesh raycasting"""
        # Calculate camera coordinate system
        forward = center - eye
        forward = forward / np.linalg.norm(forward)

        up = np.array(self.up)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        # Generate rays in a grid pattern
        rays_origin = []
        rays_direction = []

        # Use a reasonable grid size for trimesh (can be larger than numpy version)
        grid_size = 100  # Trimesh is more efficient

        for i in range(grid_size):
            for j in range(grid_size):
                # Map to normalized device coordinates
                u = (i / (grid_size - 1)) * 2 - 1  # [-1, 1]
                v = (j / (grid_size - 1)) * 2 - 1  # [-1, 1]

                # Apply field of view
                fov_rad = np.radians(self.fov_deg)
                tan_fov = np.tan(fov_rad / 2)

                # Calculate ray direction in camera space
                ray_dir_cam = np.array([u * tan_fov, v * tan_fov, 1.0])

                # Transform to world space
                ray_dir_world = (
                    ray_dir_cam[0] * right
                    + ray_dir_cam[1] * up
                    + ray_dir_cam[2] * forward
                )
                ray_dir_world = ray_dir_world / np.linalg.norm(ray_dir_world)

                rays_origin.append(eye)
                rays_direction.append(ray_dir_world)

        return rays_origin, rays_direction

    def get_partial_shape_trimesh(self, mesh, camera_location, triangle_ids=None):
        """
        Get partial shape using trimesh-based raycasting
        """
        faces = np.array(mesh.faces)
        verts = np.array(mesh.vertices)

        # Perform raycasting if triangle_ids not provided
        if triangle_ids is None:
            triangle_ids = self.ray_casting_trimesh(verts, faces, camera_location)

        # Extract visible triangles
        vis_tri = faces[triangle_ids]

        # Create trimesh and extract biggest component
        trimesh_mesh = trimesh.Trimesh(vertices=verts, faces=vis_tri, process=False)

        # Get connected components and select the largest
        if len(trimesh_mesh.faces) > 0:
            biggest_connected_components = trimesh_mesh.split(only_watertight=False)
            if len(biggest_connected_components) > 0:
                biggest_connected_component = max(
                    biggest_connected_components, key=lambda x: x.area
                )
            else:
                biggest_connected_component = trimesh_mesh
        else:
            # Fallback if no faces
            biggest_connected_component = trimesh_mesh

        partial_verts = np.array(biggest_connected_component.vertices)
        partial_faces = np.array(biggest_connected_component.faces)

        # Get correspondence to original vertices
        partial2full = self.get_corres_vert_in_full_shape(verts, partial_verts)

        return partial_verts, partial_faces, partial2full, triangle_ids

    def get_mesh(self, ans, mesh, triangle_ids=None):
        hit = ans["t_hit"].isfinite()
        if triangle_ids is None:
            triangle_ids = np.unique(ans["primitive_ids"][hit].numpy())
        # if triangle ids exist
        vis_tri = mesh.triangle.indices[triangle_ids, :]
        vertex = mesh.vertex.positions
        # extracty biggest component
        trimesh_mesh = trimesh.Trimesh(
            vertices=vertex.numpy(), faces=vis_tri.numpy(), process=False
        )
        biggest_connected_components = trimesh_mesh.split(only_watertight=False)
        biggest_connected_component = sorted(
            biggest_connected_components, key=lambda x: x.area, reverse=True
        )[0]
        return (
            np.array(biggest_connected_component.vertices),
            np.array(biggest_connected_component.faces),
            triangle_ids,
        )

    def get_corres_vert_in_full_shape(self, verts, partial_verts):
        distances = cdist(partial_verts, verts)
        indices = np.argmin(distances, axis=1)
        return indices

    def get_partial_shape(self, verts, faces, camera_location, triangle_ids=None):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        # mesh = trimesh.Trimesh(
        #     vertices=verts, faces=faces, process=False, maintain_order=True
        # ).as_open3d
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        ans, _ = self.ray_casting(mesh, camera_location=camera_location)
        partial_verts, partial_faces, triangle_ids = self.get_mesh(
            ans, mesh, triangle_ids=triangle_ids
        )
        partial2full = self.get_corres_vert_in_full_shape(verts, partial_verts)

        return partial_verts, partial_faces, partial2full, triangle_ids


def cam_generator(similarity, n_cam_pos=3):
    """Generate 3D camera positions on the unit sphere (N x 3 matrix, one row per camera).

    `similarity` controls how close the generated camera positions are to each other:
    "high" (very near), "medium" (in between), "low" (very spread out).
    """

    def _cam_generator(view_point_range, number_of_cam=2):
        # alpha and phi are the yaw and pitch respectively in the Euler angle.
        alpha = np.random.rand(number_of_cam) * (2 * view_point_range)
        phi = np.random.rand(number_of_cam) * view_point_range - view_point_range / 2.0
        # create a random deviation
        alpha += np.random.rand(1) * 2 * (np.pi - view_point_range)
        alpha = np.clip(alpha, 0, 2 * np.pi)
        if np.random.rand(1) > 0.5:
            phi += np.random.rand(1) * (np.pi - view_point_range) / 2
        else:
            phi -= np.random.rand(1) * (np.pi - view_point_range) / 2
        phi = np.clip(phi, -np.pi / 2, np.pi / 2)
        z = np.sin(phi)
        x, y = np.cos(phi) * np.sin(alpha), np.cos(phi) * np.cos(alpha)

        points = np.column_stack((x[..., None], y[..., None], z[..., None]))

        return points / np.linalg.norm(points, axis=1, keepdims=True)

    def generate_and_concatenate(num_arrays, view_point_range):
        arrays = [
            _cam_generator(view_point_range=view_point_range) for _ in range(num_arrays)
        ]
        return np.stack(arrays, axis=1)

    if similarity == "high":
        return generate_and_concatenate(n_cam_pos, view_point_range=np.pi / 8)
    elif similarity == "medium":
        return generate_and_concatenate(n_cam_pos, view_point_range=np.pi / 4)
    elif similarity == "low":
        return np.dstack(
            [_cam_generator(view_point_range=np.pi) for _ in range(n_cam_pos)]
        )
    else:
        raise ValueError("Undefined Type!")
