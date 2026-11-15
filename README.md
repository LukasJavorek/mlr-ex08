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
| `leaf_model.onnx` | MobileNetV3-based classifier, input `image` (1×3×96×96 float32), output `linear_1` (4 logits) |
| `model_info.json` | class names, input size and channel mean/std — read by the page at run time |
| `check_image.jpg` | 96×96 verification image |
| `check_expected.json` | expected probabilities for `check_image.jpg` |

## Preprocessing (follows `model_info.json` exactly)

1. take the **central square** of the frame (`side = min(width, height)`, centred),
2. **resize** it to `input_size × input_size` (96×96),
3. **scale** RGB samples to 0–1 (`/255`),
4. **standardise** each channel with `mean = [0.485, 0.456, 0.406]`,
   `std = [0.229, 0.224, 0.225]`,
5. feed a `float32` tensor of shape **1 × 3 × 96 × 96** (channel-first).

The ONNX graph ends in a fully connected layer and contains no `Softmax` node,
so `linear_1` holds raw logits; the page applies softmax to get probabilities.

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
