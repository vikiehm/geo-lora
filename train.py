import argparse
import os
import numpy as np
import torch
import yaml
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import wandb
from utils.dino_utils import compute_features, init_dino
from utils.utils import (
    weighted_contrastive_loss,
    calculate_geodesic_error_faust,
    get_state_dict_lora,
)

from utils.shape_dataset import (
    PartialShapeDataset,
    BecosDataset,
    FaustValSet,
    PSMALValSet,
    Shrec16ValSet,
)
from utils.vis import vis_shapes, vis_features_pca
from utils.paths import load_paths, data_path

torch.manual_seed(42)
np.random.seed(42)


@torch.no_grad()
def train_val(
    gt_corres,
    features_partial,
    features_full,
    verts_full,
    use_geodesic=False,
    dist_y=None,
):
    # nearest-neighbour match each (observed) partial vertex to a full vertex in
    # feature space, then look up the error between that prediction and the GT match.
    if not use_geodesic:
        dist_y = torch.cdist(verts_full, verts_full)
    dist = torch.cdist(features_partial, features_full)
    _, pred_full = torch.min(dist, dim=1)
    keep = gt_corres != -1
    return dist_y[gt_corres[keep], pred_full[keep]]


@torch.no_grad()
def validation_partial_full(
    dino_model,
    val_folder,
    pattern=None,
    global_step=None,
    dataset="becos",
    num_workers=4,
    num_views_full=9,
    num_views_partial=4,
):
    print("Starting validation")
    dino_model.eval()

    geo_errors = []
    counter = 0

    index_range = None
    if dataset == "becos":
        val_dataset = BecosDataset(
            val_folder, pattern, index_range=index_range, compute_partial=True
        )
    elif dataset == "faust":
        val_dataset = FaustValSet(val_folder, used_type="medium", partition="test")
    elif dataset == "smal":
        val_dataset = PSMALValSet(val_folder, compute_bidirectional_maps=False)
    elif dataset == "shrec16":
        val_dataset = Shrec16ValSet(val_folder, "cuts")

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
    )

    for data in tqdm(val_loader):
        dist_x = data["dist_x"][0]
        verts1 = data["verts_full"][0].to("cuda:0")
        faces1 = data["faces_full"][0].to("cuda:0")
        verts2 = data["verts_partial"][0].to("cuda:0")
        faces2 = data["faces_partial"][0].to("cuda:0")

        features1 = compute_dino_features(
            verts1,
            faces1,
            dino_model,
            num_views=num_views_full,
            set_missing_features=True,
        )

        features2 = compute_dino_features(
            verts2,
            faces2,
            dino_model,
            num_views=num_views_partial,
            set_missing_features=True,
        )

        out1 = features1.float()
        out2 = features2.float()

        val_image = vis_shapes(
            verts2,
            verts1,
            faces2.int(),
            faces1.int(),
            out2,
            out1,
        )
        counter += 1

        if counter < 5:
            wandb.log({f"val_image_{counter}": val_image}, step=global_step)

        dist2 = torch.cdist(out2, out1)
        min_dist2, min_idx2 = torch.min(dist2, dim=1)
        if dataset == "becos" or dataset == "smal" or dataset == "shrec16":
            gt10 = data["gt10"][0].to("cuda:0")
            dist_x = dist_x.to("cuda:0")
            # for p2p case we need to filter gt10 != -1
            geo_error_1 = dist_x[gt10[gt10 != -1], min_idx2[gt10 != -1]]
            print(geo_error_1.mean().item())
        elif dataset == "faust":
            corr_full = data["corres_full"][0].numpy()
            corr_partial = data["corres_partial"][0].numpy()
            geo_error_1 = calculate_geodesic_error_faust(
                dist_x.cpu().numpy(),
                corr_full,
                corr_partial,
                p2p=min_idx2.cpu().numpy(),
            )
            geo_error_1 = torch.from_numpy(geo_error_1)

        geo_errors.append(geo_error_1)

    if len(geo_errors) != 0:
        geo_errors = torch.concatenate(geo_errors)

    avg_geo_error = geo_errors.mean().detach().cpu()
    wandb.log({"Val avg error": avg_geo_error}, step=global_step)
    torch.cuda.empty_cache()

    # train mode
    dino_model.train()


def compute_dino_features(
    V,
    F,
    dino_model,
    num_views=1,
    set_missing_features=False,
):
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # num of views need to be a square number
    f_source = compute_features(
        device,
        dino_model,
        V,
        F,
        num_views=num_views,
        set_missing_features=set_missing_features,
    )
    return f_source


def main(conf_args):
    # load config
    with open(conf_args, "r") as f:
        config = yaml.safe_load(f)
    wandb_cfg = load_paths()["wandb"]
    wandb.init(
        project=wandb_cfg["project"],
        entity=wandb_cfg["entity"],
        mode="online" if wandb_cfg.get("enabled", True) else "disabled",
        config=config,
    )
    num_epochs = 100000
    eval_step = 5000
    global_step = 0
    lr = 0.0001
    batch_size = 1
    num_workers = 5
    val_folder = data_path(config["dataset"]["val_folder"])
    partial_views = config["rendering"]["partial_views"]
    downsampled_num_verts = None
    dino_model_name = "dinov3_vitb16"
    vis_frequency = 1000
    save_model_frequency = 5000
    save_model_name = config["options"]["save_model_name"]
    save_folder = load_paths()["save_folder"]
    num_partial_shapes = config["rendering"]["num_partial_shapes"]
    dataset_name = config["dataset"].get("name", "becos")
    red_list = config["dataset"].get("red_list", None)
    partiality_generation_methods = ["random_view", "random_cut", "random_holes"]
    parameter_list = []
    use_optimizer = True
    do_validation = True
    compute_dist = True
    train_dataset = PartialShapeDataset(
        data_path(config["dataset"]["train_folder"]),
        pattern=config["dataset"]["pattern"],
        phase="train",
        dataset=dataset_name,
        downsampled_num_verts=downsampled_num_verts,
        num_partial_shapes=num_partial_shapes,
        compute_geo_dist=compute_dist,
        partial_generation_methods=partiality_generation_methods,
        random_rotation_all_axis=False,
        red_list=red_list,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=train_dataset.collate_single,
    )
    device = torch.device("cuda:0")
    dino_model = init_dino(device, model_name=dino_model_name)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["qkv"],
        bias="none",
    )
    dino_model = get_peft_model(dino_model, lora_config)
    partial_views_full = config["rendering"]["partial_views_full"]

    dino_model.train()

    parameter_list.extend(list(dino_model.parameters()))
    if len(parameter_list) == 0:
        print("No optimizer")
        use_optimizer = False
    if use_optimizer:
        optimizer = torch.optim.Adam(
            parameter_list,
            lr=lr,
        )

    for i_epoch in range(0, num_epochs):
        # clean every epoch
        torch.cuda.empty_cache()
        for batch in train_loader:
            print("File name:", batch["filename"])
            verts_2 = batch["verts_2"].cuda()
            faces_2 = batch["faces_2"].cuda()
            dist_full = batch["dist_full"].cuda()

            # one DINO feature vector per full-shape vertex (cast half -> float32;
            # F.normalize underflows its eps on zero (unobserved) rows in float16)
            feat_full = compute_dino_features(
                verts_2, faces_2.int(), dino_model, num_views=partial_views_full
            ).float()

            losses = []
            for j in range(num_partial_shapes):
                verts_1, faces_1, gt01 = train_dataset.generate_partial_shape(
                    verts_2, faces_2, dist_full
                )
                feat_partial = compute_dino_features(
                    verts_1, faces_1, dino_model, num_views=partial_views
                ).float()

                if j == 0:
                    # diagnostics: partial verts observed in both shapes
                    observed = feat_partial.sum(dim=1) != 0
                    not_observed_full = torch.where(feat_full.sum(dim=1) == 0)[0]
                    observed[torch.isin(gt01, not_observed_full)] = 0
                    eucl_dist_train = train_val(
                        gt01[observed], feat_partial[observed], feat_full, verts_2
                    )
                    if compute_dist:
                        geo_dist_train = train_val(
                            gt01[observed],
                            feat_partial[observed],
                            feat_full,
                            verts_2,
                            use_geodesic=True,
                            dist_y=dist_full,
                        )

                if global_step % vis_frequency == 0:
                    buffered_image = vis_shapes(
                        verts_1, verts_2, faces_1, faces_2, feat_partial, feat_full
                    )
                    features_buffered_images = vis_features_pca(
                        verts_1, verts_2, faces_1, faces_2, feat_partial, feat_full
                    )
                else:
                    buffered_image = None
                    features_buffered_images = None

                # contrastive loss over observed partial vertices
                observed = feat_partial.sum(dim=1) != 0
                positive_1 = feat_partial[gt01 != -1][observed]
                positive_2 = feat_full[gt01[gt01 != -1]][observed]
                used_gt = gt01[gt01 != -1][observed]
                losses.append(
                    weighted_contrastive_loss(
                        positive_1, positive_2, feat_full, used_gt, dist_full
                    )
                )

            loss = torch.stack(losses)
            if torch.isnan(loss).any():
                # this can happen if no vertices are observed in the partial or full shape
                print("NaN loss detected, skipping update")
                continue
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(dino_model.parameters(), max_norm=1.0)

            optimizer.step()
            if global_step % 10 == 0:
                log_dict = {
                    "loss": loss.item(),
                    "eucl_dist_train": eucl_dist_train.mean(),
                }
                if compute_dist:
                    log_dict["geo_dist_train"] = geo_dist_train.mean()

            if buffered_image is not None:
                log_dict["buffered_image"] = buffered_image
                log_dict["features_buffered_images"] = features_buffered_images

            wandb.log(log_dict, step=global_step)
            if global_step % eval_step == 0 and do_validation:
                validation_partial_full(
                    dino_model=dino_model,
                    val_folder=val_folder,
                    pattern=config["dataset"]["pattern"],
                    global_step=global_step,
                    dataset=dataset_name,
                    num_workers=num_workers,
                )
            if global_step % save_model_frequency == 0:
                lora_state_dict = get_state_dict_lora(dino_model)
                # check if folder exists
                if not os.path.exists(f"{save_folder}/{save_model_name}"):
                    os.makedirs(f"{save_folder}/{save_model_name}")
                save_file_with_folder = (
                    f"{save_folder}/{save_model_name}/{global_step}.pth"
                )
                # if already exists, raise error
                if os.path.exists(save_file_with_folder):
                    raise ValueError(f"Model already exists at {save_file_with_folder}")
                # save model
                torch.save(
                    lora_state_dict,
                    save_file_with_folder,
                )
            global_step += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conf",
        type=str,
        default="./config/train/faust_debug.yaml",
    )
    args = parser.parse_args()
    main(args.conf)
