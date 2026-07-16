import re
import torch
import os
import trimesh
import numpy as np
from pathlib import Path
from utils.ray_caster import RayCaster, cam_generator

import open3d as o3d
from utils.utils import geodesic_dist_echo, sample_random_rotation
from itertools import product
import pickle
import igl


SMAL_FLIP = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self):
        pass

    @staticmethod
    def load_mesh(path):
        """Load a mesh and return (mesh, verts[float], faces[long])."""
        mesh = trimesh.load(path, process=False)
        verts = torch.from_numpy(np.asarray(mesh.vertices)).float()
        faces = torch.from_numpy(np.asarray(mesh.faces)).long()
        return mesh, verts, faces

    @staticmethod
    def orient_faces(faces):
        """Consistently reorient triangle faces via libigl BFS (int tensor)."""
        oriented, _ = igl.bfs_orient(faces.numpy())
        return torch.from_numpy(oriented)

    @staticmethod
    def keep_largest_component(mesh_t):
        """Drop all but the largest connected component of an open3d tensor mesh."""
        faces = mesh_t.triangle.indices.numpy()
        components = igl.facet_components(faces)
        largest = np.argmax(np.bincount(components))
        mask = components == largest
        mesh_t.triangle.indices = o3d.core.Tensor(
            faces[mask], dtype=o3d.core.Dtype.Int32
        )
        mesh_t.remove_unreferenced_vertices()
        return mesh_t

    def get_geodesic_distmat(
        self, verts, faces, filename, loading=False, normalize=True, normalize_size=None
    ):
        if self.use_geodesic:
            file_path = (
                "./geodesic_dist_matrices/" + filename.split(".off")[0] + ".npy"
            )
            if os.path.exists(file_path) and loading:
                dist_x = np.load(file_path)
                dist_x = torch.from_numpy(dist_x).float()
            else:
                dist_x = geodesic_dist_echo(
                    verts.cpu().numpy(),
                    faces.cpu().numpy(),
                    normalize=normalize,
                    normalize_size=normalize_size,
                )
                if loading:
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    np.save(file_path, dist_x)
                dist_x = torch.from_numpy(dist_x).float()
        else:
            dist_x = torch.cdist(verts, verts)
        return dist_x


class PartialShapeDataset(BaseDataset):
    def __init__(
        self,
        folder,
        phase,
        dataset,
        pattern=None,
        index_range=None,
        downsampled_num_verts=None,
        num_partial_shapes=2,
        compute_geo_dist=False,
        partial_generation_methods=None,
        red_list=None,
        random_rotation_all_axis=False,
    ):
        assert phase in [
            "train",
            "test",
            "full",
        ], f'Invalid phase {phase}, only "train" or "test" or "full"'
        self.use_geodesic = True
        self.downsampled_num_verts = downsampled_num_verts
        self.root = folder
        self.folders = os.listdir(folder)
        self.folders.sort()
        self.num_partial_shapes = num_partial_shapes
        self.compute_geo_dist = compute_geo_dist
        self.dataset = dataset
        self.red_list = red_list
        self.partial_generation_methods = partial_generation_methods
        self.random_rotation_all_axis = random_rotation_all_axis
        self.all_files = []
        selected_folders = []
        if dataset == "becos" or dataset == "shrec16":
            for i in range(len(self.folders)):
                files_in_folder = os.listdir(os.path.join(self.root, self.folders[i]))
                for f in files_in_folder:
                    if f.endswith(".off"):
                        # check if file already exists in list
                        if f.startswith("0_"):
                            f_other_way = f"1_{f[2:]}"
                        elif f.startswith("1_"):
                            f_other_way = f"0_{f[2:]}"
                        if (
                            f not in self.all_files
                            and f_other_way not in self.all_files
                        ):
                            if red_list is not None and red_list != "None":
                                if not any(re.search(pat, f) for pat in red_list):
                                    self.all_files.append(f)
                                    selected_folders.append(self.folders[i])
                            else:
                                self.all_files.append(f)
                                selected_folders.append(self.folders[i])
            self.folders = selected_folders
        elif pattern is not None and dataset == "smal" and pattern != "None":
            # if pattern in self.folders keep only those folders
            new_folders = []
            self.pattern = pattern
            for i in range(len(self.folders)):
                if any(pat in self.folders[i] for pat in pattern):
                    new_folders.append(self.folders[i])
            self.folders = new_folders
        if dataset == "faust":
            if phase == "train":
                index_range = (0, 80)
            elif phase == "test":
                index_range = (80, 101)
        # sort folders and save order to self.folders
        if index_range is not None:
            self.folders = self.folders[
                index_range[0] : min(index_range[1], len(self.folders))
            ]
            if len(self.all_files) > 0:
                self.all_files = self.all_files[
                    index_range[0] : min(index_range[1], len(self.all_files))
                ]

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        if self.dataset == "becos" or self.dataset == "shrec16":
            curr_folder = Path(self.root, self.folders[idx])
            file = self.all_files[idx]

        elif self.dataset == "faust" or self.dataset == "smal":
            curr_folder = self.root
            file = self.folders[idx]
        else:
            raise NotImplementedError(
                f"Dataset {self.dataset} not implemented in PartialShapeDataset"
            )

        mesh_null = trimesh.load(os.path.join(curr_folder, file), process=False)
        # check if mesh_null is a scene
        if isinstance(mesh_null, trimesh.Scene):
            mesh_null = mesh_null.to_mesh()
        # check if number of vertices > 12 000 then downsample to 10 000
        if mesh_null.vertices.shape[0] > 12000:
            reduction_ratio = 1 - (12000 / mesh_null.vertices.shape[0])
            mesh_null = mesh_null.simplify_quadric_decimation(reduction_ratio)
        if self.dataset == "smal":
            # meshes are upside down in smal dataset, rotate 180 degrees around x axis
            mesh_null.vertices = mesh_null.vertices @ SMAL_FLIP
        random_rotation = sample_random_rotation(one_axis=True)
        mesh_null.vertices = mesh_null.vertices @ random_rotation
        verts_2 = torch.from_numpy(np.asarray(mesh_null.vertices)).float()
        faces_2 = torch.from_numpy(np.asarray(mesh_null.faces)).long()

        filename = file.split(".off")[0]
        # remove everything before _
        filename = ("_").join(filename.split("_")[1:])

        # partial shapes are generated on the fly in the training loop via
        # generate_partial_shape(); the dataset only returns the full shape.
        dict_result = {
            "filename": filename,
            "verts_2": verts_2,
            "faces_2": faces_2,
        }

        if self.compute_geo_dist:
            dist_x = self.get_geodesic_distmat(
                verts_2, faces_2, file, loading=False, normalize=True
            )
            dict_result["dist_full"] = dist_x

        return dict_result

    def generate_partial_shape(
        self,
        verts_2,
        faces_2,
        dist_mat=None,
    ):
        gt_counter = 0
        gt01 = None
        # generate open3d full shape
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(verts_2.cpu().numpy())
        o3d_mesh.triangles = o3d.utility.Vector3iVector(faces_2.cpu().numpy())
        while gt01 is None or torch.sum(gt01 != -1) < 10 or verts_2.shape[0] < 100:
            # select partial gen randomly, select random number 0 or 1
            random_nr = np.random.randint(0, len(self.partial_generation_methods))
            partial_gen = self.partial_generation_methods[random_nr]
            if partial_gen == "random_view":
                verts_1, faces_1 = self.generate_random_view_ray(o3d_mesh)
            elif partial_gen == "random_cut":
                verts_1, faces_1 = self.generate_random_cut(o3d_mesh)
            elif partial_gen == "random_holes":
                verts_1, faces_1 = self.generate_random_holes(o3d_mesh, dist_mat)

            verts_1 = torch.from_numpy(verts_1).float().to("cuda:0")
            faces_1 = torch.from_numpy(faces_1).to("cuda:0")
            verts_2 = verts_2.to("cuda:0")
            faces_2 = faces_2.to("cuda:0")
            gt_counter += 1

            gt01 = self.get_gt_corres(verts_1, verts_2)
            if gt_counter > 10:
                return None

        random_rotation_partial = sample_random_rotation(
            one_axis=not self.random_rotation_all_axis
        )
        verts_1 = verts_1 @ torch.from_numpy(random_rotation_partial).float().to(
            verts_1.device
        )

        return (
            verts_1,
            faces_1.int(),
            gt01,
        )

    def collate_single(self, batch):
        # Training uses batch_size=1; return the single sample unwrapped (no padding,
        # no batch dimension). Each shape has a different vertex count, so batching
        # would require padding the whole pipeline for no benefit.
        return batch[0]

    def generate_random_view_ray(self, mesh):
        cam_setting = "low"
        ray_caster = RayCaster()
        cam_angles = cam_generator(cam_setting, n_cam_pos=1)
        cam_angle_1 = cam_angles[0, :, 0]
        verts_1, faces_1, p2f1, _ = ray_caster.get_partial_shape(
            mesh.vertices, mesh.triangles, cam_angle_1
        )
        return verts_1, faces_1

    def generate_random_cut(self, mesh):
        found_valid_cut = False
        max_iter = 10
        while not found_valid_cut and max_iter > 0:
            max_iter -= 1
            bbox = mesh.get_axis_aligned_bounding_box()
            mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

            mesh_t = mesh_t.clip_plane(
                point=bbox.get_center(), normal=np.random.randn(3)
            )
            mesh_t = self.keep_largest_component(mesh_t)
            verts_1 = mesh_t.vertex.positions.numpy()
            faces_1 = mesh_t.triangle.indices.numpy()
            # check if number of verts > 1/10 of original mesh and < 9/10 of original mesh
            if (
                verts_1.shape[0] > np.asarray(mesh.vertices).shape[0] / 10
                and verts_1.shape[0] < np.asarray(mesh.vertices).shape[0] * 9 / 10
            ):
                found_valid_cut = True
        if max_iter == 0:
            verts_1 = np.asarray(mesh.vertices)
            faces_1 = np.asarray(mesh.triangles)
        return verts_1, faces_1

    def generate_random_holes(self, mesh, geodesic_dist_matrix):
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        m_min = 4
        m_max = 13
        r_min = 0.1
        r_max = 0.16
        verts = torch.from_numpy(np.asarray(mesh.vertices)).to(
            geodesic_dist_matrix.device
        )
        # choose random number m between m_min and m_max
        m = torch.randint(m_min, m_max, size=(1,), device=verts.device)[0]
        # select a random r between r_min and r_max
        r = (r_max - r_min) * torch.rand(1, device=verts.device) + r_min
        random_vert_idc = torch.randint(
            0, verts.shape[0], size=(m,), device=verts.device
        )
        keep_verts = torch.ones(verts.shape[0], dtype=bool, device=verts.device)
        for rv in random_vert_idc:
            curr_geo_dist = geodesic_dist_matrix[rv, :]
            keep_verts = keep_verts & (curr_geo_dist >= r)
        mesh_t = mesh_t.select_by_index(
            torch.nonzero(keep_verts).squeeze().cpu().numpy()
        )
        mesh_t.remove_unreferenced_vertices()
        # remove small disconnected components
        mesh_t = self.keep_largest_component(mesh_t)
        verts_1 = mesh_t.vertex.positions.numpy()
        faces_1 = mesh_t.triangle.indices.numpy()
        return verts_1, faces_1

    def get_gt_corres(self, verts1, verts2):
        dist = torch.cdist(verts1, verts2)
        min_dist, min_idx = torch.min(dist, dim=1)
        return min_idx


class BecosDataset(BaseDataset):
    def __init__(
        self,
        folder,
        pattern=None,
        index_range=None,
        downsampled_num_verts=None,
        compute_partial=False,
        mode="p2f",
        compute_geodesic_dist=True,
    ):
        self.downsampled_num_verts = downsampled_num_verts
        self.root = folder
        self.folders = os.listdir(folder)
        self.use_geodesic = True
        self.compute_partial = compute_partial
        self.compute_bijective_maps = False
        if mode == "f2f" or mode == "p2p":
            self.compute_bijective_maps = True
            self.compute_partial = True
        self.mode = mode
        self.compute_geodesic_dist = compute_geodesic_dist

        if pattern is not None and pattern != "None":
            new_folders = []
            self.pattern = pattern
            for i in range(len(self.folders)):
                # check files in folder
                curr_folder = os.path.join(self.root, self.folders[i])
                files = os.listdir(curr_folder)
                files = [f for f in files if f.endswith(".off")]
                # check if pattern exists in files
                if isinstance(pattern, list):
                    files_filtered = []
                    for pat in pattern:
                        files_filtered += [f for f in files if pat in f]
                    files = files_filtered
                else:
                    files = [f for f in files if pattern in f]
                if len(files) != 0:
                    new_folders.append(self.folders[i])
            self.folders = new_folders
        # check if folder is number otherwise remove
        for folder in self.folders:
            if not folder.isdigit():
                self.folders.remove(folder)
        # iterate over all folders and check if pattern exists

        # sort based on integer value
        self.folders.sort(key=lambda x: int(x))
        if index_range is not None:
            self.folders = self.folders[
                index_range[0] : min(index_range[1], len(self.folders))
            ]

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        curr_folder = Path(self.root, self.folders[idx])
        files = os.listdir(curr_folder)
        files = [f for f in files if f.endswith(".off")]
        files.sort()
        full_file = files[0]
        partial_file = files[1]

        _, verts_full, faces_full = self.load_mesh(
            os.path.join(curr_folder, full_file)
        )
        faces_full = self.orient_faces(faces_full).long()
        if self.compute_partial:
            _, verts_partial, faces_partial = self.load_mesh(
                os.path.join(curr_folder, partial_file)
            )
            faces_partial = self.orient_faces(faces_partial).long()

        filename = full_file.split(".off")[0]
        partial_filename = partial_file.split(".off")[0]

        if self.mode == "p2p":
            # normalize by dist from _info.pkl file
            # load pkl file
            normalize_size = self.load_surface_area_info(idx, filename)
        else:
            normalize_size = None

        if self.mode == "p2f":
            loading_geodist = True
        else:
            loading_geodist = False

        if self.compute_geodesic_dist:
            dist = self.get_geodesic_distmat(
                verts_full,
                faces_full,
                full_file,
                loading=loading_geodist,
                normalize=True,
                normalize_size=normalize_size,
            )

        gt10 = np.load(os.path.join(curr_folder, "corres_10.npy"))
        gt10 = torch.from_numpy(gt10).long()

        if self.compute_bijective_maps:
            gt01 = np.load(os.path.join(curr_folder, "corres_01.npy"))
            gt01 = torch.from_numpy(gt01).long()
            if self.mode == "p2p":
                normalize_size_y = self.load_surface_area_info(idx, partial_filename)
            else:
                normalize_size_y = None
            if self.compute_geodesic_dist:
                dist_y = self.get_geodesic_distmat(
                    verts_partial,
                    faces_partial,
                    partial_file,
                    loading=False,
                    normalize=True,
                    normalize_size=normalize_size_y,
                )

        dict_result = {
            "verts_full": verts_full,
            "faces_full": faces_full,
            "filename_full": filename,
            "filename_partial": partial_filename,
            "gt10": gt10,
            "foldername": self.folders[idx],
        }
        if self.compute_geodesic_dist:
            dict_result["dist_x"] = dist
        if self.compute_partial:
            dict_result["verts_partial"] = verts_partial
            dict_result["faces_partial"] = faces_partial
        if self.compute_bijective_maps:
            dict_result["gt01"] = gt01
            if self.compute_geodesic_dist:
                dict_result["dist_y"] = dist_y

        return dict_result

    def load_surface_area_info(self, idx, filename):
        file_path = os.path.join(self.root, self.folders[idx], f"{filename}_info.pkl")
        with open(file_path, "rb") as f:
            info_data = pickle.load(f)
        return info_data["surface_area_after_scaling"]


class FaustValSet(BaseDataset):
    def __init__(self, folder, index_range=None, used_type="medium", partition="test"):
        if partition == "test":
            partial_index_range = (80, 101)
            full_index_range = (8, 11)
        else:
            partial_index_range = (0, 80)
            full_index_range = (0, 8)
        self.compute_partial = True
        self.use_geodesic = True
        self.root = folder
        base_path = Path(folder)
        self.full_folder = base_path / "neutral_pose"
        self.partial_folder = base_path / f"off_{used_type}"
        self.corres_folder = base_path / "corres"
        self.corres_partial_folder = base_path / f"corres_{used_type}"
        self.mask_folder = base_path / f"masks_{used_type}"
        self.partial_full_folder = base_path / "off"
        self.full_off_files = os.listdir(self.full_folder)
        self.part_off_files = os.listdir(self.partial_folder)
        # remove 'diffusion' files and 'dist' files
        self.full_off_files = [
            f for f in self.full_off_files if "diffusion" not in f and "dist" not in f
        ]
        self.part_off_files = [
            f for f in self.part_off_files if "diffusion" not in f and "dist" not in f
        ]
        # sort folders
        self.full_off_files.sort()
        self.part_off_files.sort()
        self.full_files = self.full_off_files[full_index_range[0] : full_index_range[1]]
        self.part_files = self.part_off_files[
            partial_index_range[0] : partial_index_range[1]
        ]
        self.combinations = list(
            product(range(len(self.full_files)), range(len(self.part_files)))
        )

    def __len__(self):
        return len(self.combinations)

    def __getitem__(self, idx):
        idx_full, idx_part = self.combinations[idx]
        full_file = self.full_files[idx_full]
        partial_file = self.part_files[idx_part]

        _, verts_full, faces_full = self.load_mesh(self.full_folder / full_file)
        if self.compute_partial:
            _, verts_partial, faces_partial = self.load_mesh(
                self.partial_folder / partial_file
            )

        dist = self.get_geodesic_distmat(
            verts_full, faces_full, full_file, loading=True
        )
        filename = full_file.split(".off")[0]
        partial_filename = partial_file.split(".off")[0]

        mask_file = partial_file.replace(".off", ".vts")
        mask = np.loadtxt(self.mask_folder / mask_file, dtype=np.int32)
        mask = torch.from_numpy(mask).bool()

        corres_full_file = full_file.replace(".off", ".vts")
        corres_full = (
            np.loadtxt(self.corres_folder / corres_full_file, dtype=np.int32) - 1
        )
        corres_full = torch.from_numpy(corres_full).long()
        corres_full = corres_full[mask]

        corres_partial_file = partial_file.replace(".off", ".vts")
        corres_partial = (
            np.loadtxt(self.corres_partial_folder / corres_partial_file, dtype=np.int32)
            - 1
        )
        corres_partial = torch.from_numpy(corres_partial).long()

        dict_result = {
            "verts_full": verts_full,
            "faces_full": faces_full,
            "filename_full": filename,
            "filename_partial": partial_filename,
            "corres_full": corres_full,
            "corres_partial": corres_partial,
            "dist_x": dist,
        }
        if self.compute_partial:
            dict_result["verts_partial"] = verts_partial
            dict_result["faces_partial"] = faces_partial

        return dict_result


class PSMALValSet(BaseDataset):
    def __init__(self, base_path, compute_bidirectional_maps=False):
        base_path = Path(base_path)
        self.off_folder = base_path / "shapes"
        self.maps_folder = base_path / "maps"
        self.use_geodesic = True

        # get all maps
        all_map_files = os.listdir(self.maps_folder)

        self.combinations = []
        self.compute_bidirectional_maps = compute_bidirectional_maps

        for curr_map in all_map_files:
            first_shape = "_".join(curr_map.split("_")[0:4])
            second_shape = "_".join(curr_map.split("_")[4:8]).split(".vts")[0]
            self.combinations.append((first_shape, second_shape))

    def __len__(self):
        return len(self.combinations)

    def __getitem__(self, idx):
        shape_1, shape_2 = self.combinations[idx]
        map_file_12 = f"{shape_1}_{shape_2}.vts"
        map_file_21 = f"{shape_2}_{shape_1}.vts"

        # rotate by 180 degrees around x axis (smal meshes are upside down)
        flip = torch.from_numpy(SMAL_FLIP).float()
        _, verts_1, faces_1 = self.load_mesh(self.off_folder / f"{shape_1}.off")
        verts_1 = verts_1 @ flip

        _, verts_2, faces_2 = self.load_mesh(self.off_folder / f"{shape_2}.off")
        verts_2 = verts_2 @ flip

        dist_1 = self.get_geodesic_distmat(
            verts_1, faces_1, shape_1, loading=False, normalize=False
        )

        corres_21 = np.loadtxt(self.maps_folder / map_file_21, dtype=np.int32)
        corres_21 = torch.from_numpy(corres_21).long()

        dict_result = {
            "verts_full": verts_1,
            "faces_full": faces_1,
            "verts_partial": verts_2,
            "faces_partial": faces_2,
            "filename_full": shape_1,
            "filename_partial": shape_2,
            "dist_x": dist_1,
            "gt10": corres_21,
        }

        if self.compute_bidirectional_maps:
            corres_12 = np.loadtxt(self.maps_folder / map_file_12, dtype=np.int32)
            corres_12 = torch.from_numpy(corres_12).long()
            dist_2 = self.get_geodesic_distmat(verts_2, faces_2, shape_2, loading=False)
            dict_result["gt01"] = corres_12
            dict_result["dist_partial"] = dist_2

        return dict_result


class Shrec16ValSet(BaseDataset):
    def __init__(self, folder, used_type="cuts", partition="val"):
        self.use_geodesic = True
        self.root = Path(folder)
        self.used_type = used_type
        self.maps_folder = self.root / used_type / "corres"
        self.partial_folder = self.root / used_type / "off"
        self.full_folder = self.root / "null" / "off"
        self.all_files = os.listdir(self.partial_folder)
        self.partition = partition
        if self.used_type == "cuts" and self.partition != "train":
            possible_cuts_list = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "assets",
                "shrec16_cuts_list.txt",
            )
            with open(possible_cuts_list, "r") as f:
                possible_cuts = f.read().splitlines()
        self.combinations = []
        for file in self.all_files:
            if file.endswith(".off"):
                if self.used_type == "cuts" and self.partition != "train":
                    # partial_shape_name = file.split(".off")[0]
                    if file not in possible_cuts:
                        continue
                partial_shape = file.split(".off")[0]
                full_shape = partial_shape.split("_")[1]
                map_file = f"{partial_shape}.vts"
                self.combinations.append((full_shape, partial_shape, map_file))

    def __len__(self):
        return len(self.combinations)

    def __getitem__(self, idx):
        first_shape, second_shape, map_file = self.combinations[idx]
        # rotate mesh by 90 degrees around x axis
        rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        rot = torch.from_numpy(rotation_matrix).float()
        _, verts, faces = self.load_mesh(
            os.path.join(self.full_folder, f"{first_shape}.off")
        )
        verts = verts @ rot
        faces = self.orient_faces(faces)

        dist_x = self.get_geodesic_distmat(
            verts, faces, first_shape, loading=False, normalize=False
        )

        _, verts_partial, faces_partial = self.load_mesh(
            os.path.join(self.partial_folder, f"{second_shape}.off")
        )
        verts_partial = verts_partial @ rot
        faces_partial = self.orient_faces(faces_partial)

        gt10 = np.loadtxt(self.maps_folder / map_file, dtype=np.int32) - 1
        gt10 = torch.from_numpy(gt10).long()[: verts.shape[0]]

        dict_result = {
            "verts_full": verts,
            "faces_full": faces,
            "filename_full": first_shape,
            "verts_partial": verts_partial,
            "faces_partial": faces_partial,
            "filename_partial": second_shape,
            "gt10": gt10,
            "dist_x": dist_x,
        }

        return dict_result


class CP2PValSet(BaseDataset):
    def __init__(self, folder):
        self.use_geodesic = True
        self.root = Path(folder)
        self.files = os.listdir(self.root)
        maps_folder = self.root / "maps"
        all_maps = os.listdir(maps_folder)
        self.off_folder = self.root / "off"
        self.combinations = []

        for map_file in all_maps:
            first_shape = map_file.split("_")[0]
            second_shape = map_file.split("_")[1].split(".map")[0]
            self.combinations.append((first_shape, second_shape, map_file))
        # filter only off files

    def __len__(self):
        return len(self.combinations)

    def __getitem__(self, idx):
        first_shape, second_shape, map_file = self.combinations[idx]
        _, verts, faces = self.load_mesh(
            os.path.join(self.off_folder, f"{second_shape}.off")
        )

        dist = self.get_geodesic_distmat(
            verts, faces, first_shape, loading=False, normalize=False
        )

        _, verts_partial, faces_partial = self.load_mesh(
            os.path.join(self.off_folder, f"{first_shape}.off")
        )

        gt10 = np.loadtxt(os.path.join(self.root, "maps", map_file), dtype=np.int32)
        gt10 = torch.from_numpy(gt10).long()[: verts_partial.shape[0]]
        dict_result = {
            "verts_full": verts,
            "faces_full": faces,
            "verts_partial": verts_partial,
            "faces_partial": faces_partial,
            "filename_full": second_shape,
            "filename_partial": first_shape,
            "dist_x": dist,
            "gt10": gt10,
        }

        return dict_result
