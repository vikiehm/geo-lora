import math
import random

import torch
import numpy as np
from torchvision import transforms as tfs
from pytorch3d.ops import ball_query
from pytorch3d.renderer import MeshRenderer
from pytorch3d.renderer.cameras import look_at_view_transform, PerspectiveCameras
from pytorch3d.renderer.lighting import PointLights
from pytorch3d.renderer.mesh.rasterizer import RasterizationSettings, MeshRasterizer
from pytorch3d.renderer.mesh.shader import HardPhongShader

from utils.utils import convert_verts_faces_to_torch_mesh
from utils.paths import load_paths

VERTEX_GPU_LIMIT = 35000


@torch.no_grad()
def run_rendering(
    device,
    mesh,
    num_views,
    H,
    W,
    add_angle_azi=0,
    add_angle_ele=0,
):
    bbox = mesh.get_bounding_boxes()
    bbox_min = bbox.min(dim=-1).values[0]
    bbox_max = bbox.max(dim=-1).values[0]
    bb_diff = bbox_max - bbox_min
    bbox_center = (bbox_min + bbox_max) / 2.0
    scaling_factor = 0.65
    distance = torch.sqrt((bb_diff * bb_diff).sum())
    distance *= scaling_factor
    steps = int(math.sqrt(num_views))
    end = 360 - 360 / steps
    elevation = (
        torch.linspace(start=0, end=end, steps=steps).repeat(steps) + add_angle_ele
    )
    azimuth = torch.linspace(start=0, end=end, steps=steps)
    azimuth = torch.repeat_interleave(azimuth, steps) + add_angle_azi
    bbox_center = bbox_center.unsqueeze(0)
    rotation, translation = look_at_view_transform(
        dist=distance, azim=azimuth, elev=elevation, device=device, at=bbox_center
    )

    normal_batched_renderings = None

    camera = PerspectiveCameras(R=rotation, T=translation, device=device)
    camera_centre = camera.get_camera_center()
    lights = PointLights(
        diffuse_color=((0.4, 0.4, 0.5),),
        ambient_color=((0.6, 0.6, 0.6),),
        specular_color=((0.01, 0.01, 0.01),),
        location=camera_centre,
        device=device,
    )

    rasterization_settings = RasterizationSettings(
        image_size=(H, W),
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=None,
        cull_backfaces=True,
    )
    shader = HardPhongShader(device=device, cameras=camera, lights=lights)
    rasterizer = MeshRasterizer(cameras=camera, raster_settings=rasterization_settings)
    batch_renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
    render_obj = mesh.extend(num_views)

    batched_renderings = batch_renderer(render_obj)
    fragments = rasterizer(render_obj)
    depth = fragments.zbuf
    return batched_renderings, normal_batched_renderings, camera, depth


def batch_render(
    device,
    mesh,
    num_views,
    H,
    W,
):
    trials = 0
    add_angle_azi = 0
    add_angle_ele = 0
    while trials < 5:
        try:
            return run_rendering(
                device,
                mesh,
                num_views,
                H,
                W,
                add_angle_azi=add_angle_azi,
                add_angle_ele=add_angle_ele,
            )
        except torch.linalg.LinAlgError as e:
            trials += 1
            print("lin alg exception at rendering, retrying ", trials)
            add_angle_azi = torch.randn(1)
            add_angle_ele = torch.randn(1)
            continue


def init_dino(device, model_name="dinov2_vitb14"):
    if "dinov2" in model_name:
        model = torch.hub.load(
            "facebookresearch/dinov2",
            model_name,
        )
    elif "dinov3_vitb" in model_name:
        cfg = load_paths()["dinov3"]
        model = torch.hub.load(
            cfg["repo_dir"],
            model_name,
            weights=cfg["vitb16_weights"],
            source="local",
        )
    elif "dinov3_vitl" in model_name:
        cfg = load_paths()["dinov3"]
        model = torch.hub.load(
            cfg["repo_dir"],
            model_name,
            weights=cfg["vitl16_weights"],
            source="local",
        )
    model = model.to(device)
    return model


def get_dino_features(dino_model, img, grid):
    img = img.permute(2, 0, 1)
    patch_size = dino_model.patch_size
    transform = tfs.Compose(
        [
            tfs.Resize((518, 518)),
            tfs.Lambda(lambda x: x / 255.0),
            tfs.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    img = transform(img)[:3].unsqueeze(0)
    features = dino_model.get_intermediate_layers(img, n=1)[0].half()

    h = int(img.shape[2] / patch_size)
    w = int(img.shape[3] / patch_size)
    dim = features.shape[-1]

    features = features.reshape(-1, h, w, dim).permute(0, 3, 1, 2)

    features = torch.nn.functional.grid_sample(
        features, grid, align_corners=False
    ).reshape(1, dino_model.embed_dim, -1)
    features = torch.nn.functional.normalize(features, dim=1)

    return features


def arange_pixels(
    resolution=(128, 128),
    batch_size=1,
    subsample_to=None,
    invert_y_axis=False,
    margin=0,
    corner_aligned=True,
    jitter=None,
    device="cuda",
):
    h, w = resolution
    n_points = resolution[0] * resolution[1]
    uh = 1 if corner_aligned else 1 - (1 / h)
    uw = 1 if corner_aligned else 1 - (1 / w)
    if margin > 0:
        uh = uh + (2 / h) * margin
        uw = uw + (2 / w) * margin
        w, h = w + margin * 2, h + margin * 2

    x, y = (
        torch.linspace(-uw, uw, w, device=device),
        torch.linspace(-uh, uh, h, device=device),
    )
    if jitter is not None:
        dx = (torch.ones_like(x).uniform_() - 0.5) * 2 / w * jitter
        dy = (torch.ones_like(y).uniform_() - 0.5) * 2 / h * jitter
        x, y = x + dx, y + dy
    x, y = torch.meshgrid(x, y)
    pixel_scaled = (
        torch.stack([x, y], -1)
        .permute(1, 0, 2)
        .reshape(1, -1, 2)
        .repeat(batch_size, 1, 1)
    )

    if subsample_to is not None and subsample_to > 0 and subsample_to < n_points:
        idx = np.random.choice(
            pixel_scaled.shape[1], size=(subsample_to,), replace=False
        )
        pixel_scaled = pixel_scaled[:, idx]

    if invert_y_axis:
        pixel_scaled[..., -1] *= -1.0

    return pixel_scaled


def get_features_per_vertex(
    device,
    dino_model,
    mesh,
    num_views=100,
    H=512,
    W=512,
    tolerance=0.01,
    mesh_vertices=None,
    bq=True,
    set_missing_features=False,
):
    if mesh_vertices is None:
        mesh_vertices = mesh.verts_list()[0]
    if len(mesh_vertices) > VERTEX_GPU_LIMIT:
        samples = random.sample(range(len(mesh_vertices)), VERTEX_GPU_LIMIT)
        maximal_distance = torch.cdist(
            mesh_vertices[samples], mesh_vertices[samples]
        ).max()
    else:
        maximal_distance = torch.cdist(mesh_vertices, mesh_vertices).max()
    ball_drop_radius = maximal_distance * tolerance
    grid = (
        arange_pixels((H, W), invert_y_axis=False, device=device)[0]
        .to(device)
        .reshape(1, H, W, 2)
        .half()
    )

    ft_per_vertex_old = torch.zeros(
        (len(mesh_vertices), dino_model.embed_dim), device=device
    ).half()
    ft_per_vertex_count_old = torch.zeros((len(mesh_vertices), 1), device=device).int()

    batched_renderings, _, camera, depth = batch_render(
        device,
        mesh,
        num_views,
        H,
        W,
    )

    pixel_coords = arange_pixels((H, W), invert_y_axis=True, device=device)[0]
    pixel_coords[:, 0] = torch.flip(pixel_coords[:, 0], dims=[0])

    for idx in range(num_views):
        dp = depth[idx].flatten().unsqueeze(1)
        xy_depth = torch.cat((pixel_coords, dp), dim=1)  # .contiguous()
        indices = xy_depth[:, 2] != -1
        xy_depth = xy_depth[indices]

        world_coords = camera[idx].unproject_points(
            xy_depth, world_coordinates=True, from_ndc=True
        )
        diffusion_input_img = batched_renderings[idx, :, :, :3] * 255
        aligned_dino_features = get_dino_features(
            dino_model, diffusion_input_img, grid
        )[0]
        features_per_pixel = aligned_dino_features[:, indices]

        bq = True
        if bq:
            knn_results = ball_query(
                world_coords.unsqueeze(0),
                mesh_vertices.unsqueeze(0),
                K=100,
                radius=ball_drop_radius,
                return_nn=False,
            )
            dists = knn_results.dists[0]
            queried_indices = knn_results.idx[0]
            if torch.sum(queried_indices != -1) == 0:
                print(
                    f"Warning: No vertices found within the ball query radius for view {idx}. Consider increasing the tolerance."
                )
                continue
            dists[queried_indices == -1] = torch.max(dists)  # (num_points, K)
            mask = queried_indices != -1

            repeat = mask.sum(dim=1)
            ft_per_vertex_count_old[queried_indices[mask]] += 1
            ft_per_vertex_old[queried_indices[mask]] += (
                features_per_pixel.repeat_interleave(repeat, dim=1).T
            )
        else:
            distances = torch.cdist(world_coords, mesh_vertices, p=2)
            closest_vertex_indices = torch.argmin(distances, dim=1)
            dist_closest = distances[
                torch.arange(len(distances)), closest_vertex_indices
            ]
            closest_vertex_indices = closest_vertex_indices[
                dist_closest < ball_drop_radius
            ]
            features_per_pixel = features_per_pixel[:, dist_closest < ball_drop_radius]
            ft_per_vertex_old[closest_vertex_indices] = (
                features_per_pixel.T.float().half()
            )
    idxs = (ft_per_vertex_count_old != 0)[:, 0]

    ft_per_vertex_old[idxs, :] = (
        ft_per_vertex_old[idxs, :] / ft_per_vertex_count_old[idxs, :]
    )
    ft_per_vertex_final = ft_per_vertex_old
    ft_per_vertex_count = ft_per_vertex_count_old

    if set_missing_features:
        missing_features = len(ft_per_vertex_count[ft_per_vertex_count == 0])
        if missing_features > 0:
            filled_indices = ft_per_vertex_count[:, 0] != 0
            missing_indices = ft_per_vertex_count[:, 0] == 0
            if len(missing_indices) > 0:
                distances = torch.cdist(
                    mesh_vertices[missing_indices], mesh_vertices[filled_indices], p=2
                )
                closest_vertex_indices = torch.argmin(distances, dim=1)
                ft_per_vertex_final[missing_indices, :] = ft_per_vertex_final[
                    filled_indices
                ][closest_vertex_indices, :]
    return ft_per_vertex_old


def compute_features(
    device,
    dino_model,
    V,
    F,
    num_views=100,
    set_missing_features=False,
    mesh_filename="default",
):
    H = 512
    W = 512
    tolerance = 0.004
    mesh = convert_verts_faces_to_torch_mesh(V, F, device=device)
    mesh_vertices = mesh.verts_list()[0]
    features = get_features_per_vertex(
        device=device,
        dino_model=dino_model,
        mesh=mesh,
        num_views=num_views,
        H=H,
        W=W,
        tolerance=tolerance,
        set_missing_features=set_missing_features,
        mesh_vertices=mesh_vertices,
    )
    return features
