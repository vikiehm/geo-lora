import os
import scipy
import torch
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.renderer import Textures
import trimesh
import networkx as nx
from sklearn import neighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import scipy.sparse.linalg as sla
import robust_laplacian
import potpourri3d as pp3d

import torch.nn.functional as F


def laplacian_decomposition(verts, faces, k=150):
    """
    Laplacian decomposition
    Args:
        verts (np.ndarray): vertices [V, 3].
        faces (np.ndarray): faces [F, 3]
        k (int, optional): number of eigenvalues/vectors to compute. Default 120.

    Returns:
        - evals: (k) list of eigenvalues of the Laplacian matrix.
        - evecs: (V, k) list of eigenvectors of the Laplacian.
        - evecs_trans: (k, V) list of pseudo inverse of eigenvectors of the Laplacian.
    """
    assert k >= 0, f"Number of eigenvalues/vectors should be non-negative, bug get {k}"
    is_cloud = faces is None
    eps = 1e-8

    # Build Laplacian matrix
    if is_cloud:
        L, M = robust_laplacian.point_cloud_laplacian(verts)
        massvec = M.diagonal()
    else:
        L = pp3d.cotan_laplacian(verts, faces, denom_eps=1e-10)
        massvec = pp3d.vertex_areas(verts, faces)
        massvec += eps * np.mean(massvec)

    if np.isnan(L.data).any():
        raise RuntimeError("NaN Laplace matrix")
    if np.isnan(massvec).any():
        raise RuntimeError("NaN mass matrix")

    # Compute the eigenbasis
    # Prepare matrices
    L_eigsh = (L + eps * scipy.sparse.identity(L.shape[0])).tocsc()
    massvec_eigsh = massvec
    Mmat = scipy.sparse.diags(massvec_eigsh)
    eigs_sigma = eps

    fail_cnt = 0
    while True:
        try:
            evals, evecs = sla.eigsh(L_eigsh, k=k, M=Mmat, sigma=eigs_sigma)
            # Clip off any eigenvalues that end up slightly negative due to numerical error
            evals = np.clip(evals, a_min=0.0, a_max=float("inf"))
            evals = evals.reshape(-1, 1)
            break
        except:
            if fail_cnt > 3:
                raise ValueError("Failed to compute eigen-decomposition")
            fail_cnt += 1
            print("Decomposition failed; adding eps")
            L_eigsh = L_eigsh + (eps * 10**fail_cnt) * scipy.sparse.identity(L.shape[0])

    evecs = np.array(evecs, ndmin=2)
    evecs_trans = evecs.T @ Mmat

    sqrt_area = np.sqrt(Mmat.diagonal().sum())
    return evals, evecs, evecs_trans, sqrt_area


def geodesic_dist_echo(verts, faces, normalize=True, normalize_size=None):
    if normalize:
        if normalize_size is not None:
            print(
                f"Normalizing geodesic distance with size: {normalize_size:.3f} from _info.pkl"
            )
            verts /= normalize_size
        else:
            old_sqrt_area = laplacian_decomposition(verts=verts, faces=faces, k=1)[-1]
            print(f"Old face sqrt area: {old_sqrt_area:.3f}")
            verts /= old_sqrt_area

    NN = min(500, verts.shape[0] - 1)

    # get adjacency matrix
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    vertex_adjacency = mesh.vertex_adjacency_graph

    assert nx.is_connected(vertex_adjacency), "Graph not connected"
    vertex_adjacency_matrix = nx.adjacency_matrix(
        vertex_adjacency, range(verts.shape[0])
    )
    # get adjacency distance matrix
    graph_x_csr = neighbors.kneighbors_graph(
        verts, n_neighbors=NN, mode="distance", include_self=False
    )
    distance_adj = csr_matrix((verts.shape[0], verts.shape[0])).tolil()
    distance_adj[vertex_adjacency_matrix != 0] = graph_x_csr[
        vertex_adjacency_matrix != 0
    ]
    # compute geodesic matrix
    geodesic_x = shortest_path(distance_adj, directed=False)
    if np.any(np.isinf(geodesic_x)):
        print("Inf number in geodesic distance. Increase NN.")
    return geodesic_x


def convert_verts_faces_to_torch_mesh(verts, faces, device):
    verts = verts.float()
    verts_rgb = torch.ones_like(verts)[None] * 0.8
    textures = Textures(verts_rgb=verts_rgb)
    mesh = Meshes(verts=[verts], faces=[faces], textures=textures)
    mesh = mesh.to(device)
    return mesh


def sample_random_rotation(one_axis=False):
    # Generate random rotation angles around x, y, and z axes
    if one_axis:
        theta_x = 0
        theta_z = 0
    else:
        theta_x = np.random.uniform(0, 2 * np.pi)
        theta_z = np.random.uniform(0, 2 * np.pi)
    theta_y = np.random.uniform(0, 2 * np.pi)

    # Create rotation matrices around x, y, and z axes
    rotation_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta_x), -np.sin(theta_x)],
            [0, np.sin(theta_x), np.cos(theta_x)],
        ]
    )

    rotation_y = np.array(
        [
            [np.cos(theta_y), 0, np.sin(theta_y)],
            [0, 1, 0],
            [-np.sin(theta_y), 0, np.cos(theta_y)],
        ]
    )

    rotation_z = np.array(
        [
            [np.cos(theta_z), -np.sin(theta_z), 0],
            [np.sin(theta_z), np.cos(theta_z), 0],
            [0, 0, 1],
        ]
    )

    # Combine rotation matrices
    rotation_matrix = np.dot(rotation_z, np.dot(rotation_y, rotation_x))

    return rotation_matrix


def weighted_contrastive_loss(
    query,
    positive_key,
    negative_keys,
    gt_map,
    geo_f,
    temperature=0.1,
    beta=2.0,
    reduction="mean",
):
    """
    feat_f: (N, D) reference features
    feat_p: (M, D) predicted/query features
    gt_map: (M,) integer indices, giving ground-truth match in feat_f
    feat_all: (N, D) all reference features for denominator
    geo_f: (N, N) geodesic error between reference features
    temperature: scalar temperature for softmax
    beta: scaling factor for exponential weighting of geodesic error
    detach_weights: whether to stop gradients through weights

    Returns scalar loss.
    """

    # Normalize features for cosine similarity
    query = F.normalize(query, dim=1)
    positive_key = F.normalize(positive_key, dim=1)
    negative_keys = F.normalize(negative_keys, dim=1)

    # Cosine between positive pairs
    positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

    # Similarity matrix (M x N)
    negative_logits = query @ transpose(negative_keys)

    weights = 0.5 + geo_f[gt_map]  # shape (M, N), pick correct row per gt match

    negative_logits = negative_logits * weights

    logits = torch.cat([positive_logit, negative_logits], dim=1)
    labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def calculate_geodesic_error_faust(dist_x, corr_x, corr_y, p2p, return_mean=False):
    """
    Calculate the geodesic error between predicted correspondence and gt correspondence

    Args:
        dist_x (np.ndarray): Geodesic distance matrix of shape x. shape [Vx, Vx]
        corr_x (np.ndarray): Ground truth correspondences of shape x. shape [V]
        corr_y (np.ndarray): Ground truth correspondences of shape y. shape [V]
        p2p (np.ndarray): Point-to-point map (shape y -> shape x). shape [Vy]
        return_mean (bool, optional): Average the geodesic error. Default True.
    Returns:
        avg_geodesic_error (np.ndarray): Average geodesic error.
    """
    ind21 = np.stack([corr_x, p2p[corr_y]], axis=-1)
    ind21 = np.ravel_multi_index(ind21.T, dims=[dist_x.shape[0], dist_x.shape[0]])
    geo_err = np.take(dist_x, ind21)
    if return_mean:
        return geo_err.mean()
    else:
        return geo_err


def get_state_dict_lora(model):
    state = model.state_dict()
    for name in list(state.keys()):
        if "lora" not in name:
            state.pop(name)
    return state


def load_state_dict(state_dict, dino_model):
    model_dict = dino_model.state_dict()
    # 1. filter out unnecessary keys
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(state_dict)
    # 3. load the new state dict
    dino_model.load_state_dict(model_dict)
    return dino_model


def get_save_filename(
    folder, basename1, basename2, dataset_name, save_features_folder, mode
):
    if dataset_name == "becos":
        if mode != "p2f":
            save_filename1 = os.path.join(
                save_features_folder,
                f"{folder}_{basename1}.pt",
            )
        else:
            save_filename1 = os.path.join(
                save_features_folder,
                f"{basename1}.pt",
            )
        save_filename2 = os.path.join(
            save_features_folder,
            f"{folder}_{basename2}.pt",
        )
    elif dataset_name == "faust":
        save_filename1 = os.path.join(
            save_features_folder,
            f"full_{basename1}.pt",
        )
        save_filename2 = os.path.join(
            save_features_folder,
            f"partial_{basename2}.pt",
        )
    else:
        save_filename1 = os.path.join(
            save_features_folder,
            f"{basename1}.pt",
        )
        save_filename2 = os.path.join(
            save_features_folder,
            f"{basename2}.pt",
        )
    return save_filename1, save_filename2
