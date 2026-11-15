1.
Application works on mobile as intended. Proof ![Screenshot from phone](Screenshot_20261115_200528_Samsung Browser.jpg)

2.
The results are: "apple": 0.9991209796102658,
  "cherry": 0.0008336547403291935,
  "chestnut": 0.000043643820398252224,
  "maple": 0.0000017218290068847934
which is the same as in the check_expected.json

3.
In index.html, preprocessFrom() center-crops the input frame to a square and resizes it to 96 × 96 pixels on an off-screen canvas. It then normalizes the RGB values using standard ImageNet statistics (mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) and arranges them into a planar NCHW float32 tensor ([1, 3, 96, 96]). Finally, classify() runs the model and applies softmax() to the output. This setup matches both model_info.json and the PyTorch validation pipeline.

4.
All image processing, QR code generation, and model inference happen 100% locally in the browser—no image data is ever uploaded to a server. The page only fetches the ONNX Runtime Web library (ort.min.js + Wasm) from the jsDelivr CDN, along with the local model weights (leaf_model.onnx) and metadata (model_info.json). Test files (check_image.jpg and check_expected.json) are loaded only when ?selftest=1 is used.
