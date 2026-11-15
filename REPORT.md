## Ablation Study: Improving the Mobile Leaf Classifier

### 1. Results Table

#### Single-Seed Ablations (`seed = 0`)
| Variant | What Changed | Epochs | Train Acc | Val Acc | Camera Val Acc | Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **V0** | Baseline: Task 3 `LeafNetBN` + geometric augmentation | 15 | 0.8943 | 0.8917 | 0.6667 | 40.9 |
| **V1** | V0 + `ColorJitter(0.3, 0.3, 0.3, 0.08)` | 15 | 0.8329 | 0.8167 | 0.7333 | 48.4 |
| **V2** | V0 + `AdaptiveAvgPool2d(1)` & `Dropout(0.2)` instead of `Flatten` | 15 | 0.7714 | 0.8083 | 0.5750 | 41.3 |
| **V3** | V0 + `AdamW(weight_decay=1e-2)` instead of `Adam` | 15 | 0.8900 | 0.8750 | 0.6833 | 40.6 |
| **V4** | V0 + `CrossEntropyLoss(label_smoothing=0.1)` | 15 | 0.8957 | 0.8833 | 0.6750 | 39.6 |
| **V5** | V0 + best-epoch checkpoint (`deepcopy`) | 15 | 0.8943 | 0.8917 | 0.6667 | 40.8 |
| **V6** | Transfer: pretrained `MobileNetV3-Small`, backbone frozen | 15 | 0.9971 | 0.8083 | 0.7000 | 16.3 |
| **V7** | Transfer: full fine-tuning of `MobileNetV3-Small` (`lr=3e-4`) | 15 | 0.9971 | 0.8833 | 0.7917 | 44.3 |
| **V8** | V7 + geometric & `ColorJitter` + `CosineAnnealingLR` + 30 epochs + best epoch | 30 | 0.9429 | 0.9167 | 0.8583 | 110.0 |
| **V9** | V8 + `AdamW(weight_decay=1e-2)` + `label_smoothing=0.1` | 30 | 0.9757 | 0.9250 | 0.8833 | 110.4 |

#### Multi-Seed Verification (`seeds = 0, 1, 2`, Mean ± Sample SD)
| Variant | Description | Train Acc | Val Acc | Camera Val Acc |
| :--- | :--- | :---: | :---: | :---: |
| **V0** | Baseline `LeafNetBN` (15 ep) | 0.8733 ± 0.0315 | 0.8611 ± 0.0268 | 0.6750 ± 0.0382 |
| **V1** | `LeafNetBN` + `ColorJitter` (15 ep) | 0.8034 ± 0.0275 | 0.7639 ± 0.0459 | 0.6861 ± 0.0614 |
| **V7** | Full Fine-Tuned `MobileNetV3-Small` (15 ep) | 0.9971 ± 0.0000 | 0.9083 ± 0.0250 | 0.8222 ± 0.0336 |
| **V8** | `MobileNetV3` + Aug + Cosine LR + Best Epoch (30 ep) | 0.9457 ± 0.0117 | 0.9139 ± 0.0048 | 0.8805 ± 0.0210 |
| **V9** | **Best Combined:** V8 + `AdamW` + Label Smoothing (30 ep) | 0.9733 ± 0.0068 | **0.9167 ± 0.0084** | **0.8861 ± 0.0048** |

*(Note: Individual seed runs were: V0 `val_acc` = 0.8917 / 0.8500 / 0.8417, `camera_val_acc` = 0.6667 / 0.7167 / 0.6417; V1 `val_acc` = 0.8167 / 0.7333 / 0.7417, `camera_val_acc` = 0.7333 / 0.6167 / 0.7083; V7 `val_acc` = 0.8833 / 0.9333 / 0.9083, `camera_val_acc` = 0.7917 / 0.8583 / 0.8167; V8 `val_acc` = 0.9167 / 0.9167 / 0.9083, `camera_val_acc` = 0.8583 / 0.8833 / 0.9000; V9 `val_acc` = 0.9250 / 0.9083 / 0.9167, `camera_val_acc` = 0.8833 / 0.8833 / 0.8917.)*

2. What Each Change Brought

Baseline (V0): The custom LeafNetBN performs well on clean validation images (86.1% mean accuracy across 3 seeds), but struggles on camera-like images (67.5%), proving too sensitive to lighting and framing shifts.

Transfer Learning — Full Fine-Tuning (V7): Switching to a pretrained MobileNetV3-Small and unfreezing all layers produced the biggest improvement. Standard accuracy rose to 90.8% (+4.7 pp) and camera-like accuracy jumped to 82.2% (+14.7 pp).

Augmentation + Cosine Schedule + 30 Epochs (V8): Adding ColorJitter, geometric augmentations, a cosine learning rate schedule, and longer training (30 epochs) boosted camera-like accuracy to 88.1% (+5.8 pp) and made results more consistent across seeds (91.4% ± 0.5% standard accuracy).

Regularized Optimization (V9): Adding AdamW weight decay (1e-2) and label smoothing (0.1) further stabilized training, achieving the best overall scores on both standard (91.7%) and camera-like validation sets (88.6%).

3. Is the Difference Larger Than the Noise?
Small scratch-model tweaks (V3, V4, V5) are within the noise: On the 120-image validation set (1 image = 0.83 pp), the baseline (V0) varies by ±2.7 pp on standard accuracy and ±3.8 pp on camera-like accuracy across seeds. Minor single-seed gains of 0.8–1.7 pp in V3 and V4 fall well within this random variance and aren't meaningful on their own.

Transfer learning and the full recipe (V7, V8, V9) clearly beat the noise: Moving from V0 to V8/V9 boosts clean validation accuracy by +5.6 pp (86.1% → 91.7%) and camera-like accuracy by +21.1 pp (67.5% → 88.6%). Every individual seed in V8 and V9 (85.8%–90.0% camera accuracy) outperforms every seed in V0 (64.2%–71.7%), proving the gains are genuine. While V8 and V9 have nearly identical average accuracy (~0.6 pp apart), V9 is noticeably more consistent across seeds (±0.5 pp vs. ±2.1 pp).

4. What Did Not Help
Global Average Pooling + Dropout on the scratch CNN (V2): Replacing Flatten with AdaptiveAvgPool2d(1) and Dropout(0.2) caused severe underfitting in 15 epochs (val_acc dropped to 80.8% and camera_val_acc to 57.5%), as the small network lost spatial capacity and needed longer to converge.

ColorJitter on the 15-epoch scratch CNN (V1): While promising on a single seed, 3-seed runs showed that 15 epochs wasn't enough to learn under heavy color distortion—clean val_acc dropped by nearly 10 pp (86.1% → 76.4%), and camera_val_acc barely improved over V0 (68.6% vs. 67.5%).

Frozen Pretrained Backbone (V6): Training only the linear head of MobileNetV3-Small heavily overfit the training set (99.7% train_acc) while underperforming the scratch baseline (val_acc 80.8% vs. 89.2%). Adapting to 96×96 leaf images required unfreezing the backbone (V7).

Best-epoch checkpointing on 15-epoch V0 (V5): Yielded identical results to V0 (89.2% / 66.7%) because the final epoch was already the best one.


proof [Screenshot from phone](Screenshot_20261115_233737_Samsung Browser.jpg)


5. Which Model to Put on the Phone
I would deploy **`V9` (Full fine-tuned `MobileNetV3-Small` trained for 30 epochs with geometric + `ColorJitter` augmentation, `CosineAnnealingLR`, `AdamW(weight_decay=1e-2)`, `label_smoothing=0.1`, and best-epoch weights)**. 

It achieves the highest accuracy on both clean validation (`0.9167` mean; `0.9250` on `seed=0`) and the camera-like validation set (`0.8861` mean), improving robustness to real phone camera crops, rotations, and lighting shifts by **+21.1 percentage points** over the Task 3 baseline while remaining lightweight and fast enough for real-time in-browser ONNX inference on a phone.
