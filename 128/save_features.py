import argparse
import os
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from datasets.custom_list import CustomListDataset
from models import get_backbone_class


def parse_args():
    parser = argparse.ArgumentParser(description="Extract MVST features")
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to CSV with columns: filepath,label[,mode,patient_id,split]")
    parser.add_argument("--data_split", type=str, default=None,
                        help="Optional: train/valid/test if CSV has 'split'")
    parser.add_argument("--resample_to", type=int, default=16000)
    parser.add_argument("--pad_seconds", type=float, default=10.0)
    parser.add_argument("--n_mels", type=int, default=128)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--hop_length", type=int, default=320)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--patch_size", type=int, required=True)
    parser.add_argument("--pretrained_ckpt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="ast")
    return parser.parse_args()


def load_model(args, input_tdim, device):
    ModelClass = get_backbone_class(args.model)
    model = ModelClass(
        label_dim=1,
        fstride=args.patch_size,
        tstride=args.patch_size,
        input_fdim=args.n_mels,
        input_tdim=input_tdim,
        imagenet_pretrain=False,
        audioset_pretrain=False,
        verbose=False,
    )
    ckpt = torch.load(args.pretrained_ckpt, map_location="cpu")
    state_dict = ckpt.get("model", ckpt)
    new_state = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "").replace("backbone.", "")
        new_state[k] = v
    model.load_state_dict(new_state, strict=False)
    model.to(device)
    model.eval()
    return model


def extract_embed(model, x):
    # Try common AST/MVST access points first
    if hasattr(model, "forward_features"):
        return model.forward_features(x)  # [B, D] or [B, T+1, D]
    # Fallback: hook last block and take CLS token
    last = {}
    def _hook(_, __, output):
        last["h"] = output
    # Try common attribute names; adjust to your model structure if needed
    handle = None
    for attr in ["blocks", "transformer", "encoder"]:
        if hasattr(model, attr):
            mod = getattr(model, attr)
            try:
                candidate = mod[-1] if isinstance(mod, (list, tuple)) else mod.blocks[-1]
            except Exception:
                candidate = None
            if candidate is not None and hasattr(candidate, "register_forward_hook"):
                handle = candidate.register_forward_hook(_hook)
                break
    _ = model(x)
    if handle is not None:
        handle.remove()
    h = last.get("h", None)
    if h is None:
        # as a last resort, return logits as embedding
        return model(x)
    # Expect [B, T+1, D]; take CLS at [:, 0, :]
    return h[:, 0, :]


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CustomListDataset(
        csv_path=args.csv,
        split=args.data_split,
        resample_to=args.resample_to,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        pad_seconds=args.pad_seconds,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=getattr(args, "num_workers", 4), pin_memory=True)

    pad_samples = int(round((args.pad_seconds or 0.0) * args.resample_to))
    input_tdim = 1 + max(0, (pad_samples - args.n_fft) // args.hop_length)
    model = load_model(args, input_tdim, device)

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    with torch.no_grad():
        for x, lab_id, meta in loader:
            x = x.to(device, non_blocking=True).float()
            emb = extract_embed(model, x)
            if emb.dim() == 3:  # [B, T, D] -> mean pool
                emb = emb.mean(dim=1)
            emb = emb.detach().cpu().numpy()  # [B, D]
            B, D = emb.shape
            for i in range(B):
                row = {
                    "filepath": meta["filepath"][i],
                    "patient_id": meta["patient_id"][i],
                    "mode": meta["mode"][i],
                    "label": meta["label"][i],
                }
                for j in range(D):
                    row[f"mvst_{args.patch_size}_f{j:04d}"] = float(emb[i, j])
                rows.append(row)

    df_view = pd.DataFrame(rows)
    out_csv = os.path.join(args.out_dir, f"features_mvst_{args.patch_size}.csv")
    df_view.to_csv(out_csv, index=False)
    print(f"[MVST] saved features: {out_csv} shape={df_view.shape}")


if __name__ == "__main__":
    main()
