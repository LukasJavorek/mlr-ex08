# Leaf classifier — live camera classification in the browser

**Live page: <https://lukasjavorek.github.io/mlr-ex08/>**

Point your phone's rear camera at a leaf and the four class probabilities
(apple, cherry, chestnut, maple) update live. Everything runs on the device:
the model is `leaf_model.onnx`, executed by
[onnxruntime-web](https://github.com/microsoft/onnxruntime-web) (WebAssembly)
loaded from a CDN. No images are uploaded anywhere.

Open the page on a phone — there is a QR code at the bottom of the page that
encodes exactly the URL above, generated in the browser by the vendored
[qrcode-generator](https://www.npmjs.com/package/qrcode-generator) library
(no third-party QR image service).

## Files

| File | Purpose |
| --- | --- |
| `index.html` | the whole app: camera, preprocessing, inference, UI, photo fallback, QR code |
| `leaf_model.onnx` | MobileNetV3-Small classifier (ablation variant V9), input `image` (1×3×96×96 float32), output `logits` (4 raw logits) |
| `model_info.json` | class names, input size and channel mean/std — read by the page at run time |
| `check_image.jpg` | 96×96 verification image |
| `check_expected.json` | expected probabilities for `check_image.jpg` |
| `ablation.py`, `export_model.py`, `make_table.py` | the ablation study, the export step, the table generator — not part of the app |
| `ablation_results.csv`, `ablation_report.md` | every measured variant and the summary table |

## Preprocessing (follows `model_info.json` exactly)

1. take the **central square** of the frame (`side = min(width, height)`, centred),
2. **resize** it to `input_size × input_size` (96×96),
3. **scale** RGB samples to 0–1 (`/255`),
4. **standardise** each channel with `mean = [0.485, 0.456, 0.406]`,
   `std = [0.229, 0.224, 0.225]`,
5. feed a `float32` tensor of shape **1 × 3 × 96 × 96** (channel-first).

The ONNX graph ends in a fully connected layer and contains no `Softmax` node,
so `logits` holds raw logits; the page applies softmax to get probabilities.

## Which model is in `leaf_model.onnx`

The shipped model is variant **V9** of the study in `ablation.py`: a pretrained
`torchvision.models.mobilenet_v3_small`, final classifier replaced by
`Linear(in, 4)` and fine-tuned on the 700 training leaves with random
crop/rotation/flip **plus** `ColorJitter(0.3, 0.3, 0.3, 0.08)`,
`AdamW(lr=3e-4, weight_decay=1e-2)`, `CrossEntropyLoss(label_smoothing=0.1)`,
`CosineAnnealingLR`, 30 epochs, keeping the best epoch's weights.

| model | `val_acc` | `camera_val_acc` |
| --- | --- | --- |
| baseline = Task 3 `LeafNetBN` + geometric augmentation | 0.861 ± 0.027 | 0.675 ± 0.038 |
| **V9, shipped** | **0.917 ± 0.008** | **0.886 ± 0.005** |

Mean ± std over the seeds 0, 1, 2. `camera_val_acc` is measured on a fixed
"camera-like" version of the validation set: each of the 120 validation images
pushed once through a random crop (0.5–1.0), a ±30° rotation and
`ColorJitter(0.3, 0.3, 0.3, 0.08)` with seed 42, generated once and reused for
every model, so the comparison is fair. The single changes that moved the
needle were colour jitter (+0.011 camera accuracy on its own, but only when
combined with the pretrained backbone) and transfer learning itself; `AdaptiveAvgPool2d`
+ `Dropout` alone made the small CNN clearly worse (V2–V5 were measured at seed 0
only, which is why they have no ± column). Full numbers for V0–V9, including the
seed spread, are in `ablation_report.md`.

One honest caveat about `check_image.jpg`: it is a 96×96 image saved as JPEG, so
its compression artefacts are large relative to the leaf. The baseline labels it
apple, V9 labels it *cherry* (p ≈ 0.64), while V9 labels the uncompressed source
of that picture (`leaves/val/apple/apple_000.jpg`) correctly as apple on all
three seeds. `check_expected.json` was regenerated from the shipped model, so
the self-test below still tests what it is meant to test: that the browser's
preprocessing and the exported graph agree, not that any class is right.

## Verify the numbers

Append `?selftest=1` to the URL — the page then classifies `check_image.jpg`
with exactly the same code path as the camera and prints a report in a
`#selftest` block: the probabilities it got, the ones from
`check_expected.json`, the maximum absolute difference, and the URL encoded in
the QR code. The self-test passes when the maximum difference is below `1e-3`.

## Camera notes (Safari on iPhone)

* `getUserMedia` needs HTTPS — that is why this is served from GitHub Pages.
* The camera starts on a tap; iOS requires a user gesture.
* `<video playsinline muted>` so the preview stays in the page instead of
  going fullscreen.
* The stage is a square with `object-fit: cover`, so the visible area **is**
  the central square the model receives.
* onnxruntime-web runs single-threaded without a worker proxy, so it does not
  need `SharedArrayBuffer` / cross-origin isolation, which iOS Safari lacks.
* **"Use a photo"** is the fallback wherever the camera is blocked; the crop
  and the maths are identical.

## Deployment

Static site, served by GitHub Pages from the `main` branch (root of this
repository). `README.md` and the tooling used while developing are not part of
the app.
