"""Export one trained variant for the phone app, exactly as notebook cells 49 and 51 do.

    .venv/Scripts/python.exe export_model.py --variant V9 --seed 0

Writes leaf_model.onnx (input `image`, output `logits`), model_info.json and
check_expected.json next to this script, then checks with onnxruntime that the exported
file reproduces the PyTorch probabilities on check_image.jpg before it prints them.
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import datasets

import ablation   # reuses the same model builders, transforms and VARIANTS table

HERE = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="V9")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    what_changed, cfg = ablation.VARIANTS[args.variant]
    model = ablation.build_model(cfg, args.seed)
    weights = torch.load(ablation.RUNS / f"{args.variant}_seed{args.seed}.pt", map_location="cpu")
    model.load_state_dict(weights)
    model.eval().cpu()
    print(f"exporting {args.variant} (seed {args.seed}): {what_changed}")

    classes = datasets.ImageFolder(ablation.DATA / "train").classes
    model_info = {"classes": classes, "input_size": 96, "mean": ablation.mean, "std": ablation.std}

    torch.onnx.export(model, (torch.zeros(1, 3, 96, 96),), str(HERE / "leaf_model.onnx"),
                      input_names=["image"], output_names=["logits"], external_data=False)

    import onnx
    onnx_model = onnx.load(str(HERE / "leaf_model.onnx"))
    onnx_model.metadata_props.add(key="model_info", value=json.dumps(model_info))
    onnx.save(onnx_model, str(HERE / "leaf_model.onnx"))
    print(f"leaf_model.onnx: {(HERE / 'leaf_model.onnx').stat().st_size / 1e6:.1f} MB")

    # The app's own check image and expected probabilities.
    x = ablation.transform(Image.open(HERE / "check_image.jpg").convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        torch_probs = torch.softmax(model(x), dim=1)[0]

    import onnxruntime as ort
    session = ort.InferenceSession(str(HERE / "leaf_model.onnx"), providers=["CPUExecutionProvider"])
    in_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name
    onnx_out = session.run(None, {in_name: x.numpy()})[0]
    onnx_probs = torch.softmax(torch.tensor(onnx_out), dim=1)[0]
    diff = (torch_probs - onnx_probs).abs().max().item()
    print(f"onnx vs pytorch max abs prob diff: {diff:.2e}  "
          f"(input name: {in_name}, output name: {out_name})")
    if diff > 1e-4:
        raise SystemExit("exported ONNX does not match the PyTorch model, not writing the json files")
    shape = session.get_inputs()[0].shape                 # batch dim may be a symbolic name
    if (in_name, out_name) != ("image", "logits") or list(shape)[1:] != [3, 96, 96]:
        raise SystemExit(f"unexpected graph signature {(in_name, out_name, shape)}, "
                         "the web app expects image -> logits at 1x3x96x96")

    check_expected = dict(zip(classes, torch_probs.tolist()))
    (HERE / "check_expected.json").write_text(json.dumps(check_expected, indent=1))
    (HERE / "model_info.json").write_text(json.dumps(model_info))
    print("check_expected:", json.dumps(check_expected))
    print("model_info:   ", json.dumps(model_info))


if __name__ == "__main__":
    main()
