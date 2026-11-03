# HSCHG
## Open-Vocabulary Audio-Visual Event Localization

The code is based on [OV-AVEL](https://github.com/jasongief/OV-AVEL). We test this release on Ubuntu with Python 3.10, PyTorch 2.0.1+cu118, and NumPy 1.24.4.

### Installation

Run the following commands from the repository root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
  --index-url https://download.pytorch.org/whl/cu118
pip install numpy==1.24.4
```

Cached-feature inference and evaluation require only PyTorch and NumPy. For training with online audio encoding, also install:

```bash
pip install timm==0.4.5 einops==0.6.1 iopath==0.1.10 \
  fvcore==0.1.5.post20221221 ftfy==6.1.1 regex==2023.6.3 \
  pytorchvideo==0.1.5 scipy==1.10.1 Pillow==9.5.0
```

### Important

Training uses 46 seen classes and `other`. Standard validation and testing use all 67 foreground classes and `other`. Both seen and unseen subsets use predictions from the complete 68-class vocabulary.

### Preparing features

Follow [OV-AVEL](https://github.com/jasongief/OV-AVEL) for the dataset, class splits, and ImageBind feature extraction. Keep data, checkpoints, and experiment outputs outside this source repository.

If you already have the cache indexes from the current OVAVEL research workspace, convert them into portable manifests:

```bash
python -m hschg.data --project /path/to/OVAVEL --output /path/to/manifests
```

This command creates `train.json` and `val.json` without copying or modifying cached features. The default evaluation split is validation.

<details>
<summary>Manifest format for independently prepared features</summary>

Paths can be absolute or relative to the manifest. Training manifests use the following structure:

```json
{
  "format": "hschg-training-v1", "split": "train",
  "candidate_ids": ["seen class 1", "...", "other"],
  "entries": [{
    "video": "video_id", "mel": "mel.npy", "vision": "vision.pt",
    "mel_sha256": "file checksum", "vision_sha256": "file checksum",
    "target": [0, 0, 46, 46, 46, 46, 46, 46, 46, 46]
  }]
}
```

`mel.npy` contains FP32 inputs of shape `[10,1,128,204]` or `[1,10,1,128,204]`. `vision.pt` contains FP32 features of shape `[10,1024]`. Targets index the training vocabulary.

Evaluation manifests use the following structure:

```json
{
  "format": "hschg-evaluation-v1", "split": "val",
  "candidate_ids": ["class 1", "...", "other"],
  "batches": [{
    "path": "batch_0000.pt", "sha256": "file checksum",
    "videos": ["video_id"], "types": ["close"]
  }]
}
```

Each batch contains `audio[B,10,1024]`, `vision[B,10,1024]`, `text[C,1024]`, `labels` (whose first item contains `[B,10]` class targets), and `videos`. Preserve the original video order and batch size of 32, including the final partial batch. Use `close` for seen videos and `open` for unseen videos. The standard validation set contains 5,798 videos and 68 candidate classes.

</details>

## Evaluation

### Pre-trained models

The trained model is preserved separately as `hschg-v999-repair-v1.pt` (252.6 MB). It contains the temporal reference, trained temporal modules, graph, geometry, class text features, and configuration. A public download link is pending; the repository owner currently provides the file separately.

Place the checkpoint outside the repository, for example at `/path/to/checkpoints/hschg-v999-repair-v1.pt`. Its SHA-256 is:

```text
9bc7bdc69d4e7eb65ca7ae21de5e1e87b90a2930a7a47cb4efe7a32943a73f7d
```

The separately preserved `training-state.pt` (698.8 MB) archives the original research-format optimizer, scheduler, and random states. Use `hschg-v999-repair-v1.pt` with this standalone repository.

### Evaluate the model

Evaluate the checkpoint using the complete cached validation set:

```bash
python -m hschg.evaluate \
  --checkpoint /path/to/checkpoints/hschg-v999-repair-v1.pt \
  --manifest /path/to/manifests/val.json --device cpu
```

For single-batch inference, prepare a `.pt` file containing `audio`, `vision`, `text`, and `candidate_ids`:

```bash
python -m hschg.predict \
  --checkpoint /path/to/checkpoints/hschg-v999-repair-v1.pt \
  --features /path/to/features.pt --device cpu
```

Audio and visual features have shape `[B,10,1024]`, before temporal encoding. Text features have shape `[C,1024]` and come from the same frozen ImageBind encoder. Candidate IDs follow the text row order and include `other`.

## Training

Online audio encoding uses the external OV-AVEL source and `imagebind_huge.pth`. Prepare the pinned upstream version:

```bash
git clone https://github.com/jasongief/OV-AVEL.git /path/to/OV-AVEL-upstream
git -C /path/to/OV-AVEL-upstream checkout b5fe1d685d0c6d0d6fd80312b5ccde79f9b73ea6
```

The adapter checks source and backbone identities. The ImageBind backbone used for this release has SHA-256:

```text
d6f6c22bedcc90708448d5d2fbb7b2db9c73f505dc89bd0b2e09b23af1b62157
```

Choose an available GPU and run:

```bash
CUDA_VISIBLE_DEVICES=0 python -m hschg.train \
  --checkpoint /path/to/checkpoints/hschg-v999-repair-v1.pt \
  --train-manifest /path/to/manifests/train.json \
  --val-manifest /path/to/manifests/val.json \
  --imagebind-source /path/to/OV-AVEL-upstream/proposed_method/ImageBind-main \
  --imagebind-weights /path/to/imagebind_huge.pth \
  --output /path/to/experiment --device cuda:0 --workers 4
```

This command initializes the trainable model from the checkpoint's frozen T=0.1 baseline reference and initializes fresh repair modules. It uses seed 123, Adam with learning rate `5e-5`, batch size 32, and one epoch of 412 updates. Training uses FP32, FP64 geometry scalars, temporal dropout 0.1, separate gradient clipping at 0.8, and MultiStepLR milestones `[10,20,30]` with factor 0.5. The loader uses four workers, one thread per worker, and prefetch factor one.

The baseline reference already receives 412 updates before repair training, giving 824 updates across the two stages. Progress prints every 10 batches and `last.pt` saves every 25 batches. Append `--resume` to resume from the next committed training batch. The completed run saves a portable `model.pt` in the output directory.

## Results

### Validation performance

All values are percentages from the seed-123 run. Evaluation uses the complete candidate vocabulary and the official batch-32 Segment/Event F1 aggregation. Avg is the mean of Accuracy, Segment F1, and Event F1.

| Split | Accuracy | Segment F1 | Event F1 | Avg |
|---|---:|---:|---:|---:|
| Seen | 72.9800 | 62.6688 | 55.7255 | 63.7914 |
| Unseen | 67.4343 | 57.3437 | 49.4426 | 58.0735 |
| Total | 69.0135 | 58.9995 | 51.4110 | **59.8080** |

The T=0.1 baseline achieves 59.8487% validation total Avg; this release achieves 59.8080%. Test evaluation is pending.

## Project Structure

`hschg/model.py` - Bounded residuals, shared geometry, entailment loss, and gradient clipping.

`hschg/graph.py` and `hschg/graph_config.py` - Four-layer heterogeneous graph and graph configuration.

`hschg/geometry.py` - Lorentz mapping and entailment-cone operations.

`hschg/temporal.py` and `hschg/heads.py` - OV-AVEL-compatible temporal modules and prediction head.

`hschg/network.py` - Complete model and portable checkpoint loading.

`hschg/backbone.py` - External ImageBind adapter for online audio encoding.

`hschg/data.py` - Cached-feature datasets and manifest conversion.

`hschg/train.py` - Training and checkpoint recovery.

`hschg/evaluate.py` and `hschg/predict.py` - Cached-feature evaluation and inference.

`hschg/metrics.py` and `hschg/official_metrics.py` - Seen/unseen/total aggregation and official metrics.

## Acknowledgement

We thank [OV-AVEL](https://github.com/jasongief/OV-AVEL) and [ImageBind](https://github.com/facebookresearch/ImageBind) for their code and pretrained representations. The upstream temporal implementation also credits [PEL4VAD](https://github.com/yujiangpu20/PEL4VAD).

The repository retains the upstream [license](LICENSE). Third-party code, models, and datasets remain subject to their respective licenses.
