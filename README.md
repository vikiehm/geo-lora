# GeoLoRA: Teaching DINOv3 About Partial 3D Geometry

Official implementation of **"Teaching DINOv3 About Partial 3D Geometry: A Self-Supervised Geometry-Aware Approach"** (CVPR 2026).

📄 **Project page:** <https://vikiehm.github.io/publications/geolora/>

> Viktoria Ehm, Dongliang Cao, Riccardo Marin, Daniel Scholz, Weikang Wang, Florian Bernard, Daniel Cremers

---

## Installation

This project uses [`uv`](https://github.com/astral-sh/uv) for dependency management and targets **Python 3.10** with **CUDA 11.8**.

```bash
# install dependencies into a local .venv (uses pyproject.toml / uv.lock)
export CUDA_HOME=/usr/local/cuda-11.8   # a CUDA 11.8 toolkit; PyTorch3D compiles against it
export TORCH_CUDA_ARCH_LIST=7.5         # optional: your GPU's compute capability, speeds up the build
uv sync
```

Key dependencies: PyTorch (cu118), [PyTorch3D](https://github.com/facebookresearch/pytorch3d) (built from source), `peft`, `trimesh`, `libigl`, `potpourri3d`, `robust-laplacian`, `polyscope`, and `wandb`. See [pyproject.toml](pyproject.toml) for the full list.

### DINOv3 weights

DINOv3 is gated: request access and download the pretrained checkpoints from the official repository [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) (fill in the access form on the model card). You need both a local clone of the repo and the weight `.pth` file:

- **ViT-B/16** (`dinov3_vitb16`, 768-dim, default) — `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`

The loader in [utils/dino_utils.py](utils/dino_utils.py) uses `torch.hub.load(..., source="local")`. Point it at your clone and downloaded checkpoints via `config/paths.yaml` (see **Setup** below).

---

## Setup: machine-specific paths

All machine-specific paths and settings live in a single, gitignored file. Copy the template and edit it for your machine:

```bash
cp config/paths.example.yaml config/paths.yaml
```

[config/paths.example.yaml](config/paths.example.yaml) documents every field:

- **`dinov3.repo_dir` / `vitb16_weights`** — your local DINOv3 clone and downloaded checkpoints.
- **`data_root`** — base directory under which all datasets live. The `train_folder` / `val_folder` / `test_folder` values in the per-benchmark configs (`config/train/*`, `config/test/*`) are resolved relative to this.
- **`save_folder`** — where training checkpoints are written and where evaluation loads them. The `save_model_name` in test configs is resolved relative to this.
- **`wandb`** — `entity` / `project`, and `enabled: false` to turn logging off.

The benchmark configs ship with dataset paths relative to `data_root` (e.g. `BeCoS/partial_full/val/`); arrange your datasets under `data_root` to match (see **Datasets** below) or edit the relative paths.

## Usage

Training and evaluation are driven by YAML config files. Make sure you have created `config/paths.yaml` (see **Setup**) first — that's where your dataset, weight, and checkpoint locations are configured.

### Training

```bash
uv run train.py --conf ./config/train/becos.yaml
```

Available training configs: `becos.yaml`, `faust.yaml`, `smal.yaml`, `becos_without_shrec16.yaml` (and `faust_debug.yaml`). Training logs metrics, visualizations and feature PCA renders to **Weights & Biases**, and periodically saves LoRA-only checkpoints to `<save_folder>/<save_model_name>/<step>.pth`.

### Visualize a single example matching

```bash
uv run example_vis.py
```

[example_vis.py](example_vis.py) computes DINOv3 + LoRA features for one full/partial pair and renders their correspondences color-coded (matching regions share a color) to `example_result.png`. It ships with example 19 from BeCoS under [example_shapes/](example_shapes/). To visualize any other matching, point `FULL_SHAPE` and `PARTIAL_SHAPE` at the top of the script at your own `.off` meshes (and set `CHECKPOINT` to the LoRA weights you want to use).

### Pretrained checkpoints

The four LoRA checkpoints used to produce the paper's test numbers ship under `saved_models/` (`becos.pth`, `faust.pth`, `becos_without_shrec16.pth`, `smal.pth`). The test configs already point at them via `save_model_name`. To use your own, train and update `save_model_name`.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{ehm2026geolora,
  title     = {Teaching DINOv3 About Partial 3D Geometry: A Self-Supervised Geometry-Aware Approach},
  author    = {Ehm, Viktoria and Cao, Dongliang and Marin, Riccardo and Scholz, Daniel and Wang, Weikang and Bernard, Florian and Cremers, Daniel},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

## Acknowledgements

The multi-view feature-lifting pipeline builds on [Diffusion-3D-Features (Diff3F)](https://github.com/niladridutt/Diffusion-3D-Features). Backbone features come from [DINOv3](https://github.com/facebookresearch/dinov3), adapted via [PEFT/LoRA](https://github.com/huggingface/peft).

## License

Released under the [MIT License](LICENSE). Note that the datasets and the DINOv3 backbone carry their own licenses, which you must comply with separately.
