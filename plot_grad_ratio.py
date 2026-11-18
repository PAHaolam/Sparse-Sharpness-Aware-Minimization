"""
Vẽ phân bố của relative difference ratio  r = log | (g_SAM - g_SGD) / g_SGD |
tại các checkpoint lưu khi train bằng SGD (checkpoint_epoch20.pth, checkpoint_epoch40.pth, ...).

Với mỗi checkpoint (cùng một bộ trọng số w), trên cùng một batch:
    g_SGD = grad L(w)
    g_SAM = grad L(w + rho * g_SGD / ||g_SGD||)
r được tính cho từng tham số; histogram được cộng dồn qua `--num_batches` batch.

Ví dụ:
    python plot_grad_ratio.py \
        --ckpt "logs/CIFAR10_..._sgd_seed1234/checkpoint_epoch*.pth" \
        --datadir /path/to/data --rho 0.05 --num_batches 20 --out grad_ratio.png

Vẽ lại từ kết quả đã tính (không cần GPU / torch):
    python plot_grad_ratio.py --from_npz grad_ratio.npz --out grad_ratio.png
"""
import argparse
import glob
import os
import re

import numpy as np

R_MIN, R_MAX, N_BINS = -12.0, 12.0, 480  # histogram range of r


# --------------------------------------------------------------------------- #
# Compute
# --------------------------------------------------------------------------- #
def expand_ckpts(patterns):
    paths = []
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        paths.extend(hits if hits else [pat])

    def key(p):
        m = re.search(r'epoch(\d+)', os.path.basename(p))
        return (int(m.group(1)) if m else 10 ** 9, p)

    return sorted(dict.fromkeys(paths), key=key)


def load_ckpt(path):
    import torch
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:  # torch < 1.13 has no `weights_only`
        return torch.load(path, map_location='cpu')


def compute_ratio_hist(model, loader, rho, num_batches, log_base, seed):
    import math
    import torch

    criterion = torch.nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    hist = torch.zeros(N_BINS, dtype=torch.float64)
    n_neg, n_tot = 0, 0
    log_div = 1.0 if log_base == 'e' else math.log(float(log_base))

    torch.manual_seed(seed)  # same batches for every checkpoint
    for i, (images, targets) in enumerate(loader):
        if i >= num_batches:
            break
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        # ---- SGD gradient at w
        model.zero_grad(set_to_none=True)
        criterion(model(images), targets).backward()
        g_sgd = [p.grad.detach().clone() for p in params]

        # ---- SAM gradient at w + e(w)
        grad_norm = torch.sqrt(sum((g ** 2).sum() for g in g_sgd))
        scale = rho / (grad_norm + 1e-7)
        with torch.no_grad():
            for p, g in zip(params, g_sgd):
                p.add_(g * scale)
        model.zero_grad(set_to_none=True)
        criterion(model(images), targets).backward()
        with torch.no_grad():
            for p, g in zip(params, g_sgd):
                p.sub_(g * scale)  # go back to w
        g_sam = [p.grad.detach().clone() for p in params]
        model.zero_grad(set_to_none=True)

        # ---- r = log|(g_sam - g_sgd) / g_sgd|
        g1 = torch.cat([g.flatten() for g in g_sgd])
        g2 = torch.cat([g.flatten() for g in g_sam])
        r = torch.log(((g2 - g1) / g1).abs()) / log_div
        r = r[torch.isfinite(r)]  # drop params with g_sgd == 0 (nan/inf)

        n_neg += int((r < 0).sum().item())
        n_tot += r.numel()
        hist += torch.histc(r.clamp(R_MIN, R_MAX).double().cpu(), bins=N_BINS, min=R_MIN, max=R_MAX)
        print(f'    batch [{i + 1}/{num_batches}] done', flush=True)

    return hist.numpy(), n_neg / max(n_tot, 1)


def run_compute(cli):
    import models  # noqa: F401  (register models)
    import data    # noqa: F401  (register datasets)
    from models.build import build_model
    from data.build import build_dataset, build_train_dataloader

    ckpt_paths = expand_ckpts(cli.ckpt)
    assert ckpt_paths, 'No checkpoint found.'
    print('Checkpoints:', *ckpt_paths, sep='\n  ')

    # dataset / loader are built once from the args stored in the first checkpoint
    first = load_ckpt(ckpt_paths[0])
    base_args = first['args']
    base_args.distributed = False
    base_args.num_workers = cli.num_workers
    if cli.datadir:
        base_args.datadir = cli.datadir
    if cli.batch_size:
        base_args.batch_size = cli.batch_size
    train_data, _, n_classes = build_dataset(base_args)
    base_args.n_classes = n_classes
    loader = build_train_dataloader(train_dataset=train_data, args=base_args)

    labels, hists, fracs = [], [], []
    for path in ckpt_paths:
        ckpt = first if path == ckpt_paths[0] else load_ckpt(path)
        margs = ckpt['args']
        margs.n_classes = n_classes
        model = build_model(margs)
        model.load_state_dict(ckpt['model'])
        model.eval() if cli.eval_mode else model.train()

        label = 'epoch {}'.format(ckpt['epoch'] + 1)
        print(f'[{label}] {path}')
        hist, frac = compute_ratio_hist(model, loader, cli.rho, cli.num_batches, cli.log_base, cli.seed)
        print(f'[{label}] fraction of params with r < 0: {frac * 100:.1f}%')
        labels.append(label)
        hists.append(hist)
        fracs.append(frac)
        del model

    np.savez(cli.npz, labels=np.array(labels), hists=np.array(hists), fracs=np.array(fracs))
    print('Saved stats to', cli.npz)
    return labels, np.array(hists), np.array(fracs)


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
def smooth(y, sigma):
    k = np.arange(-4 * sigma, 4 * sigma + 1)
    k = np.exp(-0.5 * (k / sigma) ** 2)
    return np.convolve(y, k / k.sum(), mode='same')


def plot(labels, hists, fracs, out, title=None, xlim=(-5, 5), sigma=6):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    edges = np.linspace(R_MIN, R_MAX, N_BINS + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    bw = edges[1] - edges[0]

    n = len(labels)
    colors = (plt.cm.tab10(np.arange(n)) if n <= 10 else plt.cm.viridis(np.linspace(0, 1, n)))

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    peaks = []
    for i in range(n):
        dens = smooth(hists[i] / (hists[i].sum() * bw), sigma)
        ax.fill_between(centers, dens, color=colors[i], alpha=0.35, linewidth=0,
                        label='{} ({:.1f}%)'.format(labels[i], fracs[i] * 100))
        ax.plot(centers, dens, color=colors[i], linewidth=0.8)
        j = int(np.argmax(dens))
        peaks.append((centers[j], dens[j]))

    ax.axvline(0, color='red', linestyle='--', linewidth=1.2)
    ax.set_xlim(*xlim)
    ymax = max(p[1] for p in peaks)
    ax.set_ylim(0, ymax * 1.25)
    ax.set_xlabel('Ratio')
    ax.set_ylabel('Density')
    ax.tick_params(direction='in')
    if title:
        ax.set_title(title, fontsize=10)

    # percentage call-outs like in the paper's figure (only when there are few curves)
    if n <= 4:
        for i in range(n):
            px, py = peaks[i]
            ax.annotate('{:.1f}%'.format(fracs[i] * 100), xy=(px, py * 0.9),
                        xytext=(xlim[0] + 0.6, ymax * (1.12 - 0.12 * i)),
                        fontsize=9, ha='left', va='center',
                        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='black', lw=0.8),
                        arrowprops=dict(arrowstyle='->', color='black', lw=0.8))
    ax.legend(loc='upper right', fontsize=7, frameon=False, title='checkpoint (r<0)', title_fontsize=7)

    fig.tight_layout()
    fig.savefig(out, dpi=300)
    print('Saved figure to', out)


# --------------------------------------------------------------------------- #
def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', nargs='+', help='Checkpoint paths / glob patterns, e.g. "logs/xxx/checkpoint_epoch*.pth"')
    p.add_argument('--from_npz', type=str, default=None, help='Skip computing; re-plot from a saved .npz')
    p.add_argument('--datadir', type=str, default=None, help='Override datadir stored in the checkpoint args.')
    p.add_argument('--batch_size', type=int, default=None, help='Override batch size stored in the checkpoint args.')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--rho', type=float, default=0.05, help='SAM perturbation radius.')
    p.add_argument('--num_batches', type=int, default=20, help='Number of batches accumulated per checkpoint.')
    p.add_argument('--log_base', type=str, default='e', help="Base of the log in r: 'e' (default) or e.g. 10.")
    p.add_argument('--eval_mode', action='store_true', help='Use model.eval() (BN running stats) instead of train().')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--npz', type=str, default='grad_ratio.npz', help='Where to save the computed histograms.')
    p.add_argument('--out', type=str, default='grad_ratio.png')
    p.add_argument('--title', type=str, default=None)
    return p


if __name__ == '__main__':
    cli = get_parser().parse_args()
    if cli.from_npz:
        d = np.load(cli.from_npz)
        labels, hists, fracs = list(d['labels']), d['hists'], d['fracs']
    else:
        assert cli.ckpt, 'Please pass --ckpt (or --from_npz).'
        labels, hists, fracs = run_compute(cli)
    plot(labels, hists, fracs, cli.out, title=cli.title)
