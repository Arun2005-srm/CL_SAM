# CL-SAM Research Replication

This repository evaluates the official CA-SAM implementation on a configurable
sequence of medical image-segmentation tasks. The upstream implementation is
kept as a pinned Git submodule and must not be edited.

## Repository boundary

- `upstream/ca_sam_official/`: immutable official CA-SAM source.
- `configs/`: experiment and dataset declarations owned by this project.
- `pipeline/`: dataset conversion and validation code owned by this project.
- `runners/`: external training, evaluation, and smoke-test launchers.
- `analysis/`: router accuracy, joint t-SNE, and KL-divergence analysis.
- `tests/`: checks that run without modifying the official implementation.

The pinned official CA-SAM revision is
`9a4ee0f71e264343719a42027d71099ea7ccb8d9`.

## Clone

Clone recursively so that the pinned official source is checked out:

```bash
git clone --recurse-submodules https://github.com/Arun2005-srm/CL_SAM.git
```

For an existing clone:

```bash
git submodule update --init --recursive
```

## Integrity check

Run this before an experiment:

```bash
python tools/verify_upstream.py
```

The check fails if the submodule is at a different revision or has local
modifications. Analysis and orchestration code must import or invoke the
official implementation without changing it.

