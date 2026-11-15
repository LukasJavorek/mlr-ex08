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

### 2. What Each Change Brought
* **Baseline (`V0`):** The scratch `LeafNetBN` achieves solid accuracy on clean validation images (`0.8611` mean across 3 seeds), but collapses on the camera-like validation set (`0.6750` mean), showing high sensitivity to lighting/color shifts and framing.
* **Transfer Learning — Full Fine-Tuning (`V7`):** Switching to a pretrained `MobileNetV3-Small` and unfreezing the entire backbone delivered the single largest jump in the study. Across 3 seeds, `val_acc` rose from `0.8611` to `0.9083` (+4.7 percentage points) and `camera_val_acc` jumped from `0.6750` to `0.8222` (+14.7 percentage points).
* **Stronger Augmentation + Cosine Schedule + 30 Epochs (`V8`):** While `ColorJitter` hurt the small scratch network in 15 epochs, combining geometric + `ColorJitter` augmentation with `CosineAnnealingLR`, best-epoch checkpointing, and 30 epochs on the higher-capacity `MobileNetV3-Small` pushed 3-seed `camera_val_acc` from `0.8222` (`V7`) up to `0.8805` (+5.8 percentage points) and reduced across-seed variance on `val_acc` (`0.9139 ± 0.0048`).
* **Regularized Optimization (`V9`):** Adding `AdamW(weight_decay=1e-2)` and `label_smoothing=0.1` on top of `V8` further stabilized training, yielding the highest 3-seed mean on both standard validation (`0.9167 ± 0.0084`) and camera-like validation (`0.8861 ± 0.0048`).

### 3. Is the Difference Larger Than the Noise?
* **Small scratch-model tweaks (`V3`, `V4`, `V5`) are inside the noise:** On the 120-image validation set (where 1 image = `0.0083` or 0.83 percentage points), the baseline `V0` has a 3-seed standard deviation of `±0.0268` on `val_acc` and `±0.0382` on `camera_val_acc`. Single-seed differences of 0.8–1.7 percentage points in `V3` (`AdamW`) and `V4` (`label_smoothing`) are much smaller than the seed-to-seed noise and cannot be considered real improvements in isolation.
* **Transfer learning and the combined recipe (`V7`, `V8`, `V9`) are well above the noise:** The gain from `V0` to `V8`/`V9` is **+5.6 percentage points** on clean validation (`0.8611` → `0.9167`) and **+21.1 percentage points** on camera-like validation (`0.6750` → `0.8861`). Every single seed of `V8` and `V9` (`0.8583–0.9000` on `camera_val_acc`) outperforms every single seed of `V0` (`0.6417–0.7167`) by a wide margin, confirming a genuine, reproducible improvement. Between `V8` (`0.8805 ± 0.0210`) and `V9` (`0.8861 ± 0.0048`), the mean difference is small (~0.6 percentage points, less than 1 image), but `V9` exhibits noticeably lower variance across seeds.

### 4. What Did Not Help
* **Global Average Pooling + Dropout on the scratch CNN (`V2`):** Replacing `Flatten` with `AdaptiveAvgPool2d(1)` and `Dropout(0.2)` caused severe underfitting in 15 epochs (`train_acc` dropped to `0.7714`, `val_acc` to `0.8083`, and `camera_val_acc` to `0.5750`), because the small scratch network lost spatial feature capacity and needed far more epochs to converge.
* **`ColorJitter` on the 15-epoch scratch CNN (`V1`):** Although `V1` looked promising on `camera_val_acc` in `seed=0` (`0.7333`), repeating across 3 seeds revealed that 15 epochs is too short for the scratch CNN to learn under heavy color distortion: 3-seed `val_acc` dropped by nearly 10 percentage points (`0.8611` → `0.7639`), and mean `camera_val_acc` (`0.6861 ± 0.0614`) barely beat `V0` (`0.6750 ± 0.0382`).
* **Frozen Pretrained Backbone (`V6`):** Training only the linear head of `MobileNetV3-Small` overfit the training features (`train_acc = 0.9971`) while underperforming the scratch baseline on `val_acc` (`0.8083` vs. `0.8917`). Domain adaptation to `96×96` leaf images required unfreezing the backbone (`V7`).
* **Best-epoch checkpointing on 15-epoch `V0` (`V5`):** Produced identical numbers to `V0` (`0.8917` / `0.6667`) because the final epoch (epoch 15) was already the best epoch.
proof [Screenshot from phone](Screenshot_20261115_233737_Samsung Browser.jpg)
### 5. Which Model to Put on the Phone
I would deploy **`V9` (Full fine-tuned `MobileNetV3-Small` trained for 30 epochs with geometric + `ColorJitter` augmentation, `CosineAnnealingLR`, `AdamW(weight_decay=1e-2)`, `label_smoothing=0.1`, and best-epoch weights)**. 

It achieves the highest accuracy on both clean validation (`0.9167` mean; `0.9250` on `seed=0`) and the camera-like validation set (`0.8861` mean), improving robustness to real phone camera crops, rotations, and lighting shifts by **+21.1 percentage points** over the Task 3 baseline while remaining lightweight and fast enough for real-time in-browser ONNX inference on a phone.
