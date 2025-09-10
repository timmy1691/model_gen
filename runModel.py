# run_model.py
import argparse, json, csv, os, torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="Path to model weights .pt/.pth")
    p.add_argument("--data", required=True, help="Path to dataset (e.g., ImageFolder)")
    p.add_argument("--output", default="preds.csv", help="Where to write predictions")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--labels", default=None, help="Optional labels.json mapping idx->name")
    p.add_argument("--half", action="store_true", help="Use FP16 if device supports it")
    return p.parse_args()

@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)

    # 1) Build model & load weights
    model = onnx.args.model()                     # your architecture
    state = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(state if isinstance(state, dict) else state["state_dict"])
    model.eval().to(device)

    # 2) Data pipeline (ImageFolder example; adapt to your dataset)
    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    ds = datasets.ImageFolder(args.data, transform=tfm)
    loader = DataLoader(ds, batch_size=args.batch_size,
                        num_workers=args.num_workers, pin_memory=True)

    # 3) Optional label names
    idx2name = None
    if args.labels and os.path.exists(args.labels):
        with open(args.labels) as f:
            idx2name = {int(k): v for k, v in json.load(f).items()}

    # 4) Inference loop
    dtype = torch.float16 if (args.half and device.type == "cuda") else torch.float32
    results = []
    for xb, _ in loader:
        xb = xb.to(device=device, dtype=dtype, non_blocking=True)
        logits = model(xb)
        probs = torch.softmax(logits.float(), dim=1)
        conf, pred = probs.max(dim=1)
        for i in range(pred.size(0)):
            path = ds.samples[len(results)+i][0]  # filepath aligned with DataLoader’s order
            cls = pred[i].item()
            name = idx2name.get(cls, str(cls)) if idx2name else str(cls)
            results.append((path, name, float(conf[i])))

    # 5) Save outputs
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "prediction", "confidence"])
        w.writerows(results)

    print(f"Done. Wrote {len(results)} rows to {args.output}")

if __name__ == "__main__":
    main()
