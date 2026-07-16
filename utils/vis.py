

import numpy as np
import polyscope as ps
import torch
from sklearn.decomposition import PCA
import wandb

_polyscope_initialized = False

def _ensure_polyscope_init():
    """Lazily initialize polyscope to avoid conflicts with PyTorch3D's OpenGL context."""
    global _polyscope_initialized
    if not _polyscope_initialized:
        ps.set_allow_headless_backends(True)
        ps.init()
        _polyscope_initialized = True

def vis_features_pca(verts1, verts2, faces1, faces2, out1, out2):
    _ensure_polyscope_init()
    # apply PCA to features
    pca = PCA(n_components=3)
    combined_out = torch.cat((out1, out2), dim=0)
    pca.fit(combined_out.detach().cpu().numpy())
    out1_pca = pca.transform(out1.detach().cpu().numpy())
    out2_pca = pca.transform(out2.detach().cpu().numpy())
    # visualize features
    max_val = max(np.max(out1_pca), np.max(out2_pca))
    min_val = min(np.min(out1_pca), np.min(out2_pca))
    color1 = out1_pca - min_val
    color1 /= max_val - min_val
    color2 = out2_pca - min_val
    color2 /= max_val - min_val
    poly_x = ps.register_surface_mesh(
        "x", verts1.detach().cpu().numpy(), faces1.detach().cpu().numpy()
    )
    poly_x.add_color_quantity("features", color1, defined_on="vertices", enabled=True)
    poly_y = ps.register_surface_mesh(
        "y",
        verts2.detach().cpu().numpy() + np.array([0.7, 0, 0]),
        faces2.detach().cpu().numpy(),
    )
    poly_y.add_color_quantity("features", color2, defined_on="vertices", enabled=True)
    image_buffer = ps.screenshot_to_buffer(transparent_bg=True, vertical_flip=True)
    return wandb.Image(image_buffer)


def vis_shapes(VX, VY, FX, FY, featX, featY, corres01=None):
    _ensure_polyscope_init()
    numpy_VX = VX.cpu().numpy()
    numpy_VY = VY.cpu().numpy() + np.array([0.7, 0, 0])
    numpy_FX = FX.cpu().numpy()
    numpy_FY = FY.cpu().numpy()
    poly_x = ps.register_surface_mesh("x", numpy_VX, numpy_FX)
    # nearest neighbor search in feature space
    if corres01 is None:
        dist01 = torch.cdist(featX, featY)
        corres01 = torch.argmin(dist01, dim=1).cpu().numpy()
    else:
        corres01 = corres01.cpu().numpy()
    poly_x.add_color_quantity(
        "match", numpy_VY[corres01], defined_on="vertices", enabled=True
    )
    poly_y = ps.register_surface_mesh("y", numpy_VY, numpy_FY)
    poly_y.add_color_quantity("match", numpy_VY, defined_on="vertices", enabled=True)
    image_buffer = ps.screenshot_to_buffer(transparent_bg=True, vertical_flip=True)
    return wandb.Image(image_buffer)