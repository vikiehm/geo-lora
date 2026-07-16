"""Example visualization.

Loads the two example shapes in ``example_shapes/`` (a full mesh and a partial
cut of the same shape), computes DINOv3 + LoRA features for both using the saved
checkpoint, and renders the *correspondences* color-coded: the full shape is
colored by a smooth position-based color map, and every partial vertex is colored
by its nearest neighbor in feature space on the full shape. Matching regions
therefore share the same color. The result is written to ``example_result.png``.

Run with:
    python example_vis.py
"""

import numpy as np
import polyscope as ps
import torch
from peft import LoraConfig, get_peft_model
from PIL import Image

from utils.dino_utils import compute_features, init_dino
from utils.shape_dataset import BaseDataset
from utils.utils import load_state_dict

FULL_SHAPE = "example_shapes/19_0_tr_reg_000.off"
PARTIAL_SHAPE = "example_shapes/19_1_tr_reg_099.off"
CHECKPOINT = "./saved_models/becos.pth"
OUT_PNG = "example_result.png"

DINO_MODEL_NAME = "dinov3_vitb16"
FULL_VIEWS = 49  # during inference we use 49 views for the full shape and 49 views for the partial shape
PARTIAL_VIEWS = 49


def load_lora_dino(device):
    """Build the DINOv3 + LoRA model exactly as in train.py and load a checkpoint."""
    dino_model = init_dino(device, model_name=DINO_MODEL_NAME)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["qkv"],
        bias="none",
    )
    dino_model = get_peft_model(dino_model, lora_config)
    state_dict = torch.load(CHECKPOINT, map_location=device)
    dino_model = load_state_dict(
        state_dict, dino_model
    )  # et_peft_model_state_dict(dino_model, state_dict)
    dino_model.eval()
    return dino_model


def position_colors(verts):
    """Map vertex XYZ coordinates to a smooth RGB color in [0, 1] per axis."""
    v = verts.detach().cpu().numpy()
    lo = v.min(axis=0, keepdims=True)
    hi = v.max(axis=0, keepdims=True)
    return (v - lo) / (hi - lo + 1e-8)


def feature_nn(feat_full, feat_partial):
    """Nearest neighbor (feature space) on the full shape for each partial vertex."""
    dist = torch.cdist(feat_partial, feat_full)  # (n_partial, n_full)
    return torch.argmin(dist, dim=1).cpu().numpy()  # partial -> full index


_ps_ready = False


def _init_polyscope():
    """Lazily initialize polyscope once with the right up/front axes."""
    global _ps_ready
    if not _ps_ready:
        ps.set_allow_headless_backends(True)
        ps.init()
        ps.set_ground_plane_mode("none")
        ps.set_up_dir("y_up")  # BeCoS tr_reg shapes are Y-up
        ps.set_front_dir("neg_z_front")
        _ps_ready = True


def render_correspondence(
    verts_full,
    faces_full,
    verts_partial,
    faces_partial,
    color_full,
    color_partial,
    out_png,
):
    """Render the two meshes side by side (colored) and save a screenshot."""
    _init_polyscope()
    ps.remove_all_structures()
    poly_full = ps.register_surface_mesh(
        "full", verts_full.cpu().numpy(), faces_full.cpu().numpy()
    )
    poly_full.add_color_quantity(
        "features", color_full, defined_on="vertices", enabled=True
    )
    # offset the partial shape so the two don't overlap
    shift = np.array([0.8, 0.0, 0.0])
    poly_partial = ps.register_surface_mesh(
        "partial", verts_partial.cpu().numpy() + shift, faces_partial.cpu().numpy()
    )
    poly_partial.add_color_quantity(
        "features", color_partial, defined_on="vertices", enabled=True
    )
    ps.reset_camera_to_home_view()
    ps.frame_tick()
    image_buffer = ps.screenshot_to_buffer(transparent_bg=False, vertical_flip=True)
    Image.fromarray(image_buffer).save(out_png)  # buffer is uint8 RGBA


def compute_pair(dino_model, device, full_off, partial_off):
    """Load a full/partial pair and return their verts, faces and DINO features."""
    _, verts_full, faces_full = BaseDataset.load_mesh(full_off)
    _, verts_partial, faces_partial = BaseDataset.load_mesh(partial_off)
    # reorient faces exactly like BecosDataset does
    faces_full = BaseDataset.orient_faces(faces_full).long()
    faces_partial = BaseDataset.orient_faces(faces_partial).long()
    verts_full, faces_full = verts_full.to(device), faces_full.to(device)
    verts_partial, faces_partial = verts_partial.to(device), faces_partial.to(device)
    with torch.no_grad():
        feat_full = compute_features(
            device,
            dino_model,
            verts_full,
            faces_full.int(),
            num_views=FULL_VIEWS,
            set_missing_features=True,
        ).float()
        feat_partial = compute_features(
            device,
            dino_model,
            verts_partial,
            faces_partial.int(),
            num_views=PARTIAL_VIEWS,
            set_missing_features=True,
        ).float()
    return verts_full, faces_full, verts_partial, faces_partial, feat_full, feat_partial


def main():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    dino_model = load_lora_dino(device)

    # load meshes and compute features (fill unobserved vertices with nearest
    # neighbor so every vertex gets a color)
    verts_full, faces_full, verts_partial, faces_partial, feat_full, feat_partial = (
        compute_pair(dino_model, device, FULL_SHAPE, PARTIAL_SHAPE)
    )

    # predicted correspondence: nearest neighbor in feature space (partial -> full)
    pred = feature_nn(feat_full, feat_partial)

    # full shape: smooth position-based colors; partial shape: color transferred
    # from its predicted match on the full shape
    color_full = position_colors(verts_full)
    color_partial = color_full[pred]

    # render both meshes side by side and screenshot
    render_correspondence(
        verts_full,
        faces_full,
        verts_partial,
        faces_partial,
        color_full,
        color_partial,
        OUT_PNG,
    )
    print(f"Saved visualization to {OUT_PNG}")


if __name__ == "__main__":
    main()
