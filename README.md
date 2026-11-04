# HSCHG
## Open-Vocabulary Audio-Visual Event Localization

The code is based on [OV-AVEL](https://github.com/jasongief/OV-AVEL) and tested on Ubuntu with Python 3.10 and PyTorch 2.0.1.

### Installation

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
  --index-url https://download.pytorch.org/whl/cu118
pip install numpy==1.24.4
```

### Preparing features

Follow [OV-AVEL](https://github.com/jasongief/OV-AVEL) for the dataset, class splits, and ImageBind features. Prepare `train.json`, `val.json`, and `test.json` using the cache manifest format in [hschg/data.py](hschg/data.py). Keep features and checkpoints outside this repository.

## Evaluation

### Pre-trained model

The trained model is provided separately as `hschg.pt` (252.6 MB).

### Evaluate HSCHG

```bash
python -m hschg.evaluate --checkpoint /path/to/hschg.pt \
  --manifest /path/to/test.json --device cpu
```

## Training

Install the additional dependencies and prepare the upstream audio encoder:

```bash
pip install timm==0.4.5 einops==0.6.1 iopath==0.1.10 fvcore==0.1.5.post20221221 \
  ftfy==6.1.1 regex==2023.6.3 pytorchvideo==0.1.5 scipy==1.10.1 Pillow==9.5.0
git clone https://github.com/jasongief/OV-AVEL.git /path/to/OV-AVEL
git -C /path/to/OV-AVEL checkout b5fe1d685d0c6d0d6fd80312b5ccde79f9b73ea6

CUDA_VISIBLE_DEVICES=0 python -m hschg.train \
  --checkpoint /path/to/hschg.pt \
  --train-manifest /path/to/train.json --val-manifest /path/to/val.json \
  --imagebind-source /path/to/OV-AVEL/proposed_method/ImageBind-main \
  --imagebind-weights /path/to/imagebind_huge.pth \
  --output /path/to/experiment --device cuda:0 --workers 4
```

Training starts from the checkpoint's baseline reference: 412 baseline updates followed by 412 HSCHG updates. Append `--resume` to continue an interrupted run.

## Results

### Test performance (%)

| Split | Accuracy | Segment F1 | Event F1 | Avg |
|---|---:|---:|---:|---:|
| Seen | 74.6094 | 64.1080 | 57.3519 | 65.3564 |
| Unseen | 67.3532 | 56.8744 | 49.1569 | 57.7948 |
| Total | 69.4278 | 58.9563 | 51.5111 | **59.9651** |

Results cover all 5,820 test videos using the seed-123 checkpoint and official batch-32 F1 aggregation. Avg is the mean of the three metrics. On validation, HSCHG achieves 59.8080% total Avg and the T=0.1 baseline achieves 59.8487%.

## Project Structure

`hschg/model.py`, `graph.py`, `geometry.py`, `temporal.py`, `heads.py`, `network.py`, `graph_config.py` - Model and configuration.

`hschg/data.py`, `backbone.py` - Cached inputs and audio encoding.

`hschg/train.py`, `evaluate.py`, `predict.py` - Training, evaluation, and inference.

`hschg/metrics.py`, `official_metrics.py` - Evaluation metrics.

## Acknowledgement

We thank [OV-AVEL](https://github.com/jasongief/OV-AVEL), [ImageBind](https://github.com/facebookresearch/ImageBind), and [PEL4VAD](https://github.com/yujiangpu20/PEL4VAD) for their code. Please follow the [license](LICENSE) and the terms of the upstream projects.
