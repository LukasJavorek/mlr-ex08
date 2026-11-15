# Ablation results

| variant | what changed | epochs | seed | train_acc | val_acc | camera_val_acc |
|---|---|---|---|---|---|---|
| V0 | baseline: Task 3 LeafNetBN + geometric augmentation (notebook) | 15 | 0 | 0.894 | 0.892 | 0.667 |
| V0-s1 | baseline: Task 3 LeafNetBN + geometric augmentation (notebook) | 15 | 1 | 0.889 | 0.850 | 0.717 |
| V0-s2 | baseline: Task 3 LeafNetBN + geometric augmentation (notebook) | 15 | 2 | 0.837 | 0.842 | 0.642 |
| **V0 mean±std** | seeds 0, 1, 2 |  | 0,1,2 |  | **0.861 +/- 0.027** | **0.675 +/- 0.038** |
| V1 | + ColorJitter(0.3,0.3,0.3,0.08) in the train transform | 15 | 0 | 0.833 | 0.817 | 0.733 |
| V1-s1 | + ColorJitter(0.3,0.3,0.3,0.08) in the train transform | 15 | 1 | 0.779 | 0.733 | 0.617 |
| V1-s2 | + ColorJitter(0.3,0.3,0.3,0.08) in the train transform | 15 | 2 | 0.799 | 0.742 | 0.708 |
| **V1 mean±std** | seeds 0, 1, 2 |  | 0,1,2 |  | **0.764 +/- 0.046** | **0.686 +/- 0.061** |
| V2 | + AdaptiveAvgPool2d(1) and Dropout(0.2) instead of Flatten | 15 | 0 | 0.771 | 0.808 | 0.575 |
| V3 | + AdamW with weight_decay=1e-2 instead of Adam | 15 | 0 | 0.890 | 0.875 | 0.683 |
| V4 | + CrossEntropyLoss(label_smoothing=0.1) | 15 | 0 | 0.896 | 0.883 | 0.675 |
| V5 | + best-epoch checkpoint (deepcopy) instead of last-epoch weights | 15 | 0 | 0.894 | 0.892 | 0.667 |
| V6 | transfer: pretrained MobileNetV3-Small, backbone frozen | 15 | 0 | 0.997 | 0.808 | 0.700 |
| V7 | transfer: full fine-tuning of MobileNetV3-Small, lr=3e-4 | 15 | 0 | 0.997 | 0.883 | 0.792 |
| V7-s1 | transfer: full fine-tuning of MobileNetV3-Small, lr=3e-4 | 15 | 1 | 0.997 | 0.933 | 0.858 |
| V7-s2 | transfer: full fine-tuning of MobileNetV3-Small, lr=3e-4 | 15 | 2 | 0.997 | 0.908 | 0.817 |
| **V7 mean±std** | seeds 0, 1, 2 |  | 0,1,2 |  | **0.908 +/- 0.025** | **0.822 +/- 0.034** |
| V8 | transfer + geometric & ColorJitter + CosineAnnealingLR + 30 epochs + best epoch | 30 | 0 | 0.943 | 0.917 | 0.858 |
| V8-s1 | transfer + geometric & ColorJitter + CosineAnnealingLR + 30 epochs + best epoch | 30 | 1 | 0.936 | 0.917 | 0.883 |
| V8-s2 | transfer + geometric & ColorJitter + CosineAnnealingLR + 30 epochs + best epoch | 30 | 2 | 0.959 | 0.908 | 0.900 |
| **V8 mean±std** | seeds 0, 1, 2 |  | 0,1,2 |  | **0.914 +/- 0.005** | **0.881 +/- 0.021** |
| V9 | best combined: V8 + AdamW(weight_decay=1e-2) + label_smoothing=0.1 | 30 | 0 | 0.976 | 0.925 | 0.883 |
| V9-s1 | best combined: V8 + AdamW(weight_decay=1e-2) + label_smoothing=0.1 | 30 | 1 | 0.979 | 0.908 | 0.883 |
| V9-s2 | best combined: V8 + AdamW(weight_decay=1e-2) + label_smoothing=0.1 | 30 | 2 | 0.966 | 0.917 | 0.892 |
| **V9 mean±std** | seeds 0, 1, 2 |  | 0,1,2 |  | **0.917 +/- 0.008** | **0.886 +/- 0.005** |

`hist_val_acc` and variant V10 are not available: no `archive.zip` (last year's train/val split) exists in this workspace, so no model was trained or evaluated on combined data.
