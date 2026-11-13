# BEVFusion dual-head correction run

Final checkpoint from the 10-epoch correction run (2026-11-11/12).

## Result

- Checkpoint: `checkpoints/epoch_10.pth`
- SHA-256: `4ffefa802b5ccab4af289c0d578269ad52ef924b74c007159e407870318f3f7e`
- Final mIoU@max: **0.3819**
- Final detection mAP: **0.6549**
- Final detection NDS: **0.6973**

### Per-class mIoU

| Class | mIoU@max |
|---|---:|
| drivable_area | 0.7262 |
| ped_crossing | 0.3094 |
| walkway | 0.4317 |
| stop_line | 0.2047 |
| carpark_area | 0.3201 |
| divider | 0.2993 |

## Run configuration

- 2x RTX PRO 6000, batch 6 per GPU
- Full nuScenes trainval with CBGS
- Start checkpoint: merged detection + epoch-20 map checkpoint
- Trainable: camera ResNet50 backbone + neck, `heads.map`, `heads.object`
- Frozen: LiDAR encoder, fuser, decoder
- Loss scales: map 1.0, object 0.35
- Max epochs: 10
- First validation: epoch 5 (`mIoU 0.3810`)
- Final validation: epoch 10 (`mIoU 0.3819`)

## Files

- `checkpoints/epoch_10.pth` — final model
- `checkpoints/latest.pth` — symlink
- `model_config.yaml` — standalone model-only config with both heads
- `train_config.yaml` — resolved training config used by the run
- `logs/20261111_140503.log` — training log from run directory
- `logs/20261111_140503.log.json` — structured training log
- `logs/dualhead_correction.log` — full stdout/monitor log

## Comparison

| Run | Map mIoU | Detection mAP | Detection NDS |
|---|---:|---:|---:|
| Original map-only epoch 20 | 0.3913 | — | — |
| Equal-weight joint pilot, epoch 5 | 0.3572 | 0.6612 | 0.7021 |
| Correction run, epoch 5 | 0.3810 | 0.6577 | 0.6990 |
| Correction run, epoch 10 | 0.3819 | 0.6549 | 0.6973 |

The correction run recovered most of the map loss from the equal-weight pilot
but did not surpass the original map-only checkpoint.
