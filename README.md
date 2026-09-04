# CL-SAM Research Replication

Configuration-driven continual medical segmentation experiments around the
official CA-SAM implementation. The baseline is pinned at revision
`9a4ee0f71e264343719a42027d71099ea7ccb8d9` and remains unmodified.

## Experimental boundary

- `upstream/ca_sam_official/` is the immutable official implementation.
- `src/cl_sam_replication/` contains our preparation, launch, instrumentation,
  validation, and analysis code.
- Pixel accuracy is detached reporting instrumentation. It does not participate
  in the official loss, gradients, optimizer, routing, or checkpoints.
- The official trainer calls its per-epoch evaluation split `test`. Our
  preparation pipeline places validation samples there during training and
  retains final test samples in `dataset.final_test.json`.

Run `python tools/verify_upstream.py` before experiments. It fails if the
official revision or its working tree changed.

## Clone in Colab

```bash
%cd /content
!git clone --recurse-submodules https://github.com/Arun2005-srm/CL_SAM.git
%cd /content/CL_SAM
!pip install -q -r requirements-colab.txt
!pip install -q -e .
```

For a clone that already exists, run `git submodule update --init --recursive`.

## Configure datasets

Copy `configs/five_tasks_colab.yaml` and edit only YAML values. A regular paired
dataset needs image and mask directories:

```yaml
- name: my_task
  modality: ultrasound
  source:
    image_dir: /content/data/my_task/images
    mask_dir: /content/data/my_task/masks
    image_glob: "**/*"
    mask_glob: "**/*"
    mask_matching: {strip_suffixes: ["_mask"]}
  labels: {mode: binary, threshold: 0, foreground_name: lesion}
  split:
    seed: 42
    max_samples: null
    group_aware: false
    ratios: {train: 0.70, val: 0.15, test: 0.15}
```

`source.collections` supports datasets such as EBHI-SEG that contain several
category-specific image/label directories. Multiclass masks can declare a
source-value-to-name mapping under `labels.classes`.

## Prepare and validate

```bash
!cl-sam-prepare --config configs/five_tasks_colab.yaml
!cl-sam-validate --config configs/five_tasks_colab.yaml
```

Preparation produces official sparse `.npz` labels, pseudo masks, deterministic
train/validation/test splits, manifests, and dataset statistics.

## GPU smoke test

```bash
!cl-sam-smoke --config configs/five_tasks_colab.yaml --tasks 1 --samples 4
```

Inspect the planned command without using the GPU:

```bash
!cl-sam-smoke --config configs/five_tasks_colab.yaml --dry-run
```

## Train

Use the configured order:

```bash
!cl-sam-train --config configs/five_tasks_colab.yaml
```

Provide an explicit order:

```bash
!cl-sam-train --config configs/five_tasks_colab.yaml \
  --task-order ebhi_seg,isic_2018,sts_2d,busi,kvasir_seg
```

Or use a reproducible shuffle:

```bash
!cl-sam-train --config configs/five_tasks_colab.yaml \
  --shuffle-tasks --shuffle-seed 17
```

Every epoch displays:

```text
Epoch [1/24] | train acc=... loss=... | val acc=... loss=... |
Dice=... mIoU=... BIoU=...
```

Each task saves `history.csv`, `history.json`, and `training_curves.png`.

## Final held-out evaluation

```bash
!cl-sam-evaluate --config configs/five_tasks_colab.yaml
```

This uses the separately retained final-test manifests and invokes the official
accumulated-task VAE-router evaluation.

## Joint latent and router analysis

```bash
!cl-sam-extract-latents --config configs/five_tasks_colab.yaml
!cl-sam-latents \
  --input /content/drive/MyDrive/CA_SAM/CL_SAM_runs/five_task_default/latent_analysis/embeddings.npz \
  --output-dir /content/drive/MyDrive/CA_SAM/CL_SAM_runs/five_task_default/latent_analysis/results
```

Outputs include:

- one t-SNE fitted jointly to all TaskVAE posterior means;
- one t-SNE fitted jointly to all shared SAM router features;
- router accuracy, unknown rate, predictions, and confusion matrix;
- per-sample internal VAE KL values;
- directional and symmetric pairwise latent KL matrices and heatmap;
- reusable embeddings and t-SNE coordinates.

Task-specific VAEs do not share an explicitly aligned latent basis. Therefore,
the shared-feature t-SNE is provided alongside the requested combined Z-space
plot and is the stronger cross-task representation analysis.

## Tests

```bash
python -m pytest -q
```

See `THIRD_PARTY_NOTICES.md` for CA-SAM and Segment Anything attribution. No
datasets, patient data, SAM weights, or trained checkpoints are committed.
