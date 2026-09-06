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
Pairing is strict by default. A dataset with documented unmatched source files
can set `source.unpaired_policy: skip`; the skipped counts and example keys are
printed rather than silently accepted.

## Prepare and validate

```bash
!cl-sam-prepare --config configs/five_tasks_colab.yaml
!cl-sam-validate --config configs/five_tasks_colab.yaml
```

Preparation produces official sparse `.npz` labels, pseudo masks, deterministic
train/validation/test splits, manifests, and dataset statistics. By default,
`image_transfer: auto` uses storage-free hard links when source and destination
are on the same local filesystem (as they are under `/content` in Colab), then
falls back to ordinary copies when hard links are unavailable. Deleting the
original source path does not remove a successfully hard-linked prepared image.
For unusually large masks, `mask_prepare_size: [1024, 1024]` bounds label and
pseudo-mask intermediates at the spatial size consumed by CA-SAM. Source JPEGs
remain unchanged and are still resized by the official loader. Pseudo-mask IDs
are stored using the smallest safe signed integer dtype and are converted to
float32 by the unchanged official loader.

## GPU smoke test

For the SSH/DGX layout under `/raid/workspace/AI4CV_CLSAM`, use
`configs/five_tasks_server.yaml`. It points training at the extracted prepared
datasets, the SAM checkpoint, and the server output directory. The Colab config
remains separate and unchanged.

```bash
!cl-sam-smoke --config configs/five_tasks_colab.yaml --tasks 1 --samples 4
```

On the SSH server:

```bash
python -m cl_sam_replication.smoke \
  --config configs/five_tasks_server.yaml --tasks 1 --samples 4
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

It also saves a configurable number of held-out qualitative examples per task
under `evaluation/qualitative/<task>/`. Each `qualitative_grid.png` shows the
input image, ground-truth mask, routed prediction, and overlay, annotated with
per-sample Dice, IoU, boundary IoU, and pixel accuracy. The same values are
written to `sample_metrics.csv`. Set `evaluation.qualitative_samples_per_task`
in the configuration (use `0` to disable these panels).

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
