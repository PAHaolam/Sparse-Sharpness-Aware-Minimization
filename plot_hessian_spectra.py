"""
Vẽ lại Figure 5 (Hessian spectrum của ResNet18 trên CIFAR10: SGD / SAM / SSAM)
cho checkpoint được tạo bởi train.py của repo Sparse-SAM.

Chuẩn bị (đặt TRONG thư mục gốc của repo Sparse-SAM, cùng cấp với train.py):
    - copy thư mục  PyHessian-master/pyhessian/        -> ./pyhessian/
    - copy file     PyHessian-master/density_plot.py   -> ./density_plot.py
    (KHÔNG copy utils.py của PyHessian, vì trùng tên với package utils/ của repo này)

Chạy:
    python plot_fig5.py \
        --ckpt SGD=logs/xxx_sgd/checkpoint.pth \
               SAM=logs/xxx_sam-sgd/checkpoint.pth \
               SSAM=logs/xxx_ssam-d-sgd/checkpoint.pth \
        --datadir /path/to/datasets \
        --out figure5.pdf
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

# import models để các hàm resnet18/... được đăng ký vào MODELS_REGISTRY
import models  # noqa: F401
from models.build import MODELS_REGISTRY

from pyhessian import hessian              # từ PyHessian
from density_plot import density_generate  # từ PyHessian

# mean/std giống hệt data/dataset.py (CIFAR10_base)
CIFAR10_MEAN = np.array([125.3, 123.0, 113.9]) / 255.0
CIFAR10_STD = np.array([63.0, 62.1, 66.7]) / 255.0


# ----------------------------------------------------------------------
# Load model đúng cách train.py đã lưu:
#   torch.save({'model': model_without_ddp.state_dict(), 'optimizer': ...,
#               'lr_scheduler': ..., 'epoch': ..., 'args': args}, 'checkpoint.pth')
# -> state_dict nằm ở key 'model', KHÔNG có tiền tố "module.";
#    'args' là argparse.Namespace nên cần weights_only=False.
# ----------------------------------------------------------------------
def load_model(path, device, model_name="resnet18", n_classes=10):
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch cũ chưa có tham số weights_only
        ckpt = torch.load(path, map_location="cpu")

    saved_args = ckpt.get("args", None)
    if saved_args is not None:
        # dùng đúng cấu hình model lúc train (args.model, args.n_classes, args.dataset)
        model = MODELS_REGISTRY.get(saved_args.model)(saved_args)
        info = (f"model={saved_args.model} opt={saved_args.opt} "
                f"epoch={ckpt.get('epoch')}")
    else:
        model = MODELS_REGISTRY.get(model_name)(
            argparse.Namespace(n_classes=n_classes, dataset="CIFAR10_base"))
        info = f"model={model_name} epoch={ckpt.get('epoch')}"

    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    print(f"Loaded {path}: {info}")
    return model


# ----------------------------------------------------------------------
# Dữ liệu cho Hessian: tập TRAIN, không augmentation, normalize như lúc train
# ----------------------------------------------------------------------
def get_hessian_batches(datadir, mini_bs, hessian_bs, seed):
    assert hessian_bs % mini_bs == 0
    tf = T.Compose([T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    train = torchvision.datasets.CIFAR10(root=datadir, train=True,
                                         transform=tf, download=True)
    g = torch.Generator().manual_seed(seed)  # cả 3 model dùng cùng 1 tập mẫu
    loader = torch.utils.data.DataLoader(train, batch_size=mini_bs,
                                         shuffle=True, generator=g)
    n = hessian_bs // mini_bs
    batches = []
    for i, (x, y) in enumerate(loader):
        batches.append((x, y))
        if i == n - 1:
            break
    return batches


def analyze(model, batches, device, lanczos_iter=100, n_v=1):
    criterion = nn.CrossEntropyLoss()
    cuda = device == "cuda"
    if len(batches) == 1:
        comp = hessian(model, criterion, data=batches[0], cuda=cuda)
    else:
        comp = hessian(model, criterion, dataloader=batches, cuda=cuda)

    # sanity check: loss / acc của model trên đúng các mẫu dùng tính Hessian
    with torch.no_grad():
        tot_loss, correct, cnt = 0.0, 0, 0
        for x, y in batches:
            x, y = x.to(device), y.to(device)
            out = model(x)
            tot_loss += criterion(out, y).item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            cnt += len(y)
    print(f"   loss={tot_loss / cnt:.4f} acc={correct / cnt:.4f} "
          f"(nếu acc ~10% nghĩa là load sai model/normalize)")

    top, _ = comp.eigenvalues(top_n=5)           # power iteration
    eigen, weight = comp.density(iter=lanczos_iter, n_v=n_v)  # Lanczos (SLQ)
    return top[0], top[0] / top[4], eigen, weight


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", nargs="+", required=True,
                   help="NAME=path/to/checkpoint.pth, ví dụ SGD=a.pth SAM=b.pth SSAM=c.pth")
    p.add_argument("--datadir", required=True, help="thư mục chứa CIFAR10 (như --datadir lúc train)")
    p.add_argument("--hessian_batch_size", type=int, default=200)
    p.add_argument("--mini_hessian_batch_size", type=int, default=200)
    p.add_argument("--lanczos_iter", type=int, default=100)
    p.add_argument("--n_v", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default="figure5.pdf")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    items = [s.split("=", 1) for s in args.ckpt]
    batches = get_hessian_batches(args.datadir, args.mini_hessian_batch_size,
                                  args.hessian_batch_size, args.seed)

    fig, axes = plt.subplots(1, len(items), figsize=(4.5 * len(items), 3.6))
    axes = np.atleast_1d(axes)
    for ax, (name, path) in zip(axes, items):
        model = load_model(path, device)
        lam1, ratio, eigen, weight = analyze(model, batches, device,
                                             args.lanczos_iter, args.n_v)
        print(f"[{name}] lambda1={lam1:.2f}  lambda1/lambda5={ratio:.2f}")
        density, grids = density_generate(eigen, weight)
        ax.semilogy(grids, density + 1e-7)
        ax.set_ylim(1e-6, 1e0)
        ax.set_xlabel("Eigenvalue")
        ax.set_ylabel("Density")
        ax.text(0.45, 0.85, rf"$\lambda_1$ = {lam1:.1f}", transform=ax.transAxes)
        ax.text(0.45, 0.70, rf"$\lambda_1/\lambda_5$ = {ratio:.1f}", transform=ax.transAxes)
        ax.set_title(f"{name} Hessian spectrum")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    plt.tight_layout()
    plt.savefig(args.out)
    print("Saved", args.out)


if __name__ == "__main__":
    main()