# SDDFNet

**Structure-Aware Dynamic Difference Fusion Network for Building Damage Assessment**

SDDFNet studies building localization and damage classification from paired pre- and post-disaster satellite images. The project builds on [xView2 Strong Baseline](https://github.com/PaulBorneP/Xview2_Strong_Baseline) and uses a shared ResNeSt50d encoder with a Siamese U-Net architecture.

## Installation

Use Python 3.10 or newer. Install matching PyTorch and torchvision builds for your CUDA environment, then install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

Run commands from the repository root. Training and evaluation use PyTorch Lightning and Hydra; logs are written locally in CSV format.

## Project Structure

| Path | Purpose |
| --- | --- |
| `main.py` | Training, validation, testing and prediction |
| `conf/` | Hydra model, data and trainer configurations |
| `datasets/` | xBD and EBD loading and paired augmentation |
| `modules/supervised.py` | Loss, metrics and optimizer management |
| `legacy/zoo/models.py` | Runnable reference baseline implementations |
| `legacy/zoo/sddfnet.py` | Interface-level SDDFNet architecture outline |
| `legacy/losses.py` | Dice and focal losses |
| `tools/create_masks.py` | Pixel-annotation rasterization |
| `tools/check_dataset.py` | Prepared-data validation |
| `tests/` | Data, model and checkpoint smoke tests |

## Data Preparation

### xBD

Download the official train, tier3 and test subsets. Each subset should contain `images/` and `labels/`. The expected paired filenames are `<sample>_pre_disaster.png` and `<sample>_post_disaster.png`, with matching JSON annotation names under `labels/`.

Generate masks from the pixel-coordinate WKT annotations:

```bash
python tools/create_masks.py --roots data/xBD/train data/xBD/tier3 data/xBD/test
python tools/check_dataset.py --roots data/xBD/train data/xBD/tier3 data/xBD/test
```

The script creates single-channel PNG masks in each subset's `masks/` directory:

| Mask | Encoding |
| --- | --- |
| Pre-disaster localization | 0: background; 255: building |
| Post-disaster damage | 0: background; 1: no damage; 2: minor; 3: major; 4: destroyed |

Existing masks are retained unless `--overwrite` is specified. Following the supplied baseline preprocessing, `un-classified` annotations map to no damage by default; the script reports their count. Use `--unclassified error` to reject them instead. The converter supports Polygon and MultiPolygon geometries, including holes, and reads output dimensions from each image pair.

The experiment protocol merges train and tier3 (9,168 pairs), then uses a 90/10 disaster-stratified split with random state 23: 8,251 training and 917 validation pairs. The official test subset (933 pairs) is evaluated separately. This is an event-overlapping protocol; the holdout subset is not used. Generated split lists are saved under `logs/splits/`.

### EBD

EBD is used only for zero-shot evaluation. Prepared event folders may be arranged as `data/EBD/<event>/images/` and `data/EBD/<event>/masks/`, using the same paired filenames and mask encoding as xBD. Images are discovered recursively.

If the source annotations use the same pixel-coordinate WKT JSON schema as xBD, masks can be generated with:

```bash
python tools/create_masks.py --roots data/EBD --recursive
python tools/check_dataset.py --roots data/EBD --recursive
```

For other annotation schemas, convert them to the mask encoding above before evaluation. Do not treat the converter as a universal EBD annotation parser. The manuscript evaluates 18,215 pairs from 12 events, without target-domain training or checkpoint selection.

### Input Processing

The loader preserves the supplied code's OpenCV BGR channel order and normalization `image / 127 - 1`. Pre/post images and masks receive the same spatial augmentation. Training uses 608 × 608 crops/resizing, horizontal flips, and random affine transformations. Evaluation uses original spatial dimensions with no random augmentation. Missing or invalid samples raise an error instead of being silently replaced.

## Architecture

SDDFNet uses a shared ResNeSt50d encoder and combines structure-detail aggregation, dynamic difference guidance and multi-scale damage refinement. The model contract and high-level execution sequence are documented in `legacy/zoo/sddfnet.py`; the reference ResNeSt50 and ResNet34 Siamese U-Net baselines remain executable end to end.

## Reference Training

Train the ResNeSt50 reference baseline on one GPU with a global batch size of 16:

```bash
python main.py network=resnest50_baseline data_root=/path/to/xBD \
  name=resnest50_seed42 seed=42 trainer.devices=1 data.labeled_batch_size=16
```

To use two GPUs, use batch size 8 per GPU:

```bash
python main.py network=resnest50_baseline data_root=/path/to/xBD \
  name=resnest50_seed42 seed=42 \
  trainer.devices=2 trainer.strategy=ddp trainer.sync_batchnorm=true \
  data.labeled_batch_size=8
```

| Setting | Value |
| --- | --- |
| Encoder | ImageNet-pretrained ResNeSt50d |
| Epochs | 60 |
| Optimizer | AdamW |
| Learning rate | 0.0001 |
| Weight decay | 0.00001 |
| LR schedule | MultiStepLR, milestones 24/42/54, gamma 0.5 |
| Precision | 16-bit mixed precision |
| Gradient clipping | L2 norm, 0.5 |
| Seeds | 42, 43, 44 |
| Checkpoint selection | Highest validation xBD Score |

Pretrained encoder weights are obtained through timm by default. To load compatible local encoder weights, set `network.model.ckpt_path=/path/to/encoder.pth`. Use `network.model.pretrained=false` for a random-initialization smoke test. Full-checkpoint resume/evaluation does not download separate encoder weights.

Use `network=res34_baseline` to select the ResNet34 alternative, and supply a distinct `name` for every experiment. The ResNet34 option uses the common 60-epoch schedule here and should not be assumed to reproduce every upstream Strong Baseline experiment setting.

## Checkpoints and Resume

Checkpoints are saved in `outputs/checkpoints/<name>/`. `last.ckpt` includes model, optimizer and scheduler state for resuming training:

```bash
python main.py network=resnest50_baseline data_root=/path/to/xBD \
  name=resnest50_seed42 ckpt_path=/path/to/last.ckpt
```

Single-device training evaluates the best validation checkpoint after fitting by default. Multi-device training should be followed by a separate single-device evaluation command to avoid duplicated samples from distributed evaluation padding. Use the same architecture and experiment settings when resuming or evaluating a checkpoint.

## Evaluation

```bash
python main.py mode=test network=resnest50_baseline data_root=/path/to/xBD \
  ckpt_path=/path/to/best.ckpt trainer.devices=1 \
  name=resnest50_test
```

For zero-shot EBD evaluation:

```bash
python main.py mode=test network=resnest50_baseline data=ebd_generalization \
  ebd_root=/path/to/EBD ckpt_path=/path/to/best.ckpt \
  trainer.devices=1 name=resnest50_ebd
```

Use `mode=validate` to evaluate the fixed xBD validation split. Testing and prediction require an explicit full-model checkpoint. Local CSV logs and `test_metrics.json` are written under the experiment log directory.

## Prediction

```bash
python main.py mode=predict network=resnest50_baseline data=ebd_generalization \
  ebd_root=/path/to/EBD ckpt_path=/path/to/best.ckpt \
  trainer.devices=1 name=resnest50_predictions
```

Prediction uses the same prepared, labeled dataset layout as evaluation. Localization and damage PNGs are saved under `outputs/<name>/submission/`; event-folder prefixes avoid filename collisions across events.

## Metrics

Predicted localization uses sigmoid probability > 0.38. Damage predictions use the highest-probability damage channel and are masked by predicted localization, matching the supplied implementation.

| Metric | Definition |
| --- | --- |
| Localization F1 | Binary building segmentation F1 |
| Class-wise F1 | No damage, minor, major, destroyed |
| Damage F1 | Harmonic mean of the four damage-class F1 scores |
| Score | 0.3 × Localization F1 + 0.7 × Damage F1 |

Metrics are aggregated over the evaluated dataset. Run seeds 42, 43 and 44 separately when preparing repeated-run summaries.

## Tests

```bash
python -m unittest discover -s tests -v
```

The smoke tests check image/mask validation, split isolation, polygon holes, baseline model interfaces and checkpoint resume. They are not a replacement for full-dataset GPU experiments.

## Acknowledgments

This project builds on [xView2 Strong Baseline](https://github.com/PaulBorneP/Xview2_Strong_Baseline), with ResNeSt support from timm. Please cite the original method and dataset papers when using these resources. Existing third-party notices are retained in [LICENSE](LICENSE).
