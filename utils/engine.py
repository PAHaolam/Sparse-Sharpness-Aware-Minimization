import time
from collections import defaultdict
from typing import Iterable

import torch
import torch.distributed as dist
from utils.dist import is_dist_avail_and_initialized

def train_one_epoch(
    model: torch.nn.Module,
    train_loader : Iterable,
    criterion, optimizer, epoch, logger, log_freq, use_closure
):
    model.train()

    _memory = MetricLogger()
    _memory.add_meter('train_loss', Metric())
    _memory.add_meter('train_acc1', Metric())
    _memory.add_meter('train_acc5', Metric())
    for batch_idx, (images, targets) in enumerate(train_loader):
        batch_start = time.time()
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        def closure():
            output = model(images)
            loss = criterion(output, targets)
            loss.backward()

        output = model(images)
        loss = criterion(output, targets)
        
        optimizer.zero_grad()
        loss.backward()
        if use_closure: 
            optimizer.step(
                closure, 
                epoch=epoch,
                step=epoch*len(train_loader)+batch_idx,
                batch_idx=batch_idx,
                model=model, 
                train_data=train_loader.dataset, 
                logger=logger,
            ) 
        else: 
            optimizer.step()

        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        batch_num = images.shape[0]
        _memory.update_meter('train_loss', loss.item(), n=batch_num)
        _memory.update_meter('train_acc1', acc1.item(), n=batch_num)
        _memory.update_meter('train_acc5', acc5.item(), n=batch_num)
        
        msg = ' '.join([
            'Epoch: {epoch}',
            '[{batch_id}/{batch_len}]',
            'lr:{lr:.6f}',
            'Train Loss:{train_loss:.4f}',
            'Train Acc1:{train_acc1:.4f}',
            'Train Acc5:{train_acc5:.4f}',
            'Time:{batch_time:.3f}s'])
        if batch_idx % log_freq == 0:
            logger.log(
                msg.format(
                    epoch=epoch, 
                    batch_id=batch_idx, batch_len = len(train_loader),
                    lr=optimizer.param_groups[0]["lr"],
                    train_loss=_memory.meters["train_loss"].global_avg,
                    train_acc1=_memory.meters["train_acc1"].global_avg,
                    train_acc5=_memory.meters["train_acc5"].global_avg,
                    batch_time=time.time() - batch_start,
                )
            )
        _memory.synchronize_between_processes()
    return {
        name: meter.global_avg for name, meter in _memory.meters.items()
    }

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    val_loader: Iterable,
):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    _memory = MetricLogger()
    _memory.add_meter('test_loss', Metric())
    _memory.add_meter('test_acc1', Metric())
    _memory.add_meter('test_acc5', Metric())

    for images, targets in val_loader:
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        output = model(images)
        loss = criterion(output, targets)
        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        
        batch_num = images.shape[0]
        _memory.update_meter('test_loss', loss.item(), n=batch_num)
        _memory.update_meter('test_acc1', acc1.item(), n=batch_num)
        _memory.update_meter('test_acc5', acc5.item(), n=batch_num)
    _memory.synchronize_between_processes()
    return {
        name: meter.global_avg for name, meter in _memory.meters.items()
    }

def compute_r_neg_ratio(model, loader, criterion, rho):
    """
    Fraction of parameters whose relative difference ratio
        r = log | (g_SAM - g_SGD) / g_SGD |
    is < 0, measured over one full pass of `loader` (every batch: g_SGD at w,
    g_SAM at w + rho * g_SGD / ||g_SGD||, both on the same batch).

    The pass leaves training untouched: weights are restored exactly, BN buffers and
    RNG states are restored, and gradients are cleared. Parameters with g_SGD == 0
    (nan/inf ratio) are skipped. In distributed mode the counts are summed over ranks.
    `model` should be the un-wrapped model (model_without_ddp) to avoid DDP grad sync.
    """
    was_training = model.training
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    device = params[0].device

    # ---- backups
    w_backup = [p.detach().clone() for p in params]
    buf_backup = [b.detach().clone() for b in model.buffers()]
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

    counts = torch.zeros(2, dtype=torch.float64, device=device)  # [#(r<0), #valid]
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # g_SGD at w
        model.zero_grad(set_to_none=True)
        criterion(model(images), targets).backward()
        g_sgd = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]

        # g_SAM at w + e(w)
        grad_norm = torch.sqrt(sum((g ** 2).sum() for g in g_sgd))
        scale = rho / (grad_norm + 1e-7)
        with torch.no_grad():
            for p, g in zip(params, g_sgd):
                p.add_(g * scale)
        model.zero_grad(set_to_none=True)
        criterion(model(images), targets).backward()

        with torch.no_grad():
            for p, w in zip(params, w_backup):
                p.copy_(w)  # exact restore (add_/sub_ would leave rounding drift)
            for p, g1 in zip(params, g_sgd):
                g2 = p.grad if p.grad is not None else torch.zeros_like(p)
                r = torch.log(((g2 - g1) / g1).abs())
                r = r[torch.isfinite(r)]
                counts[0] += (r < 0).sum()
                counts[1] += r.numel()

    # ---- restore state
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        for b, bb in zip(model.buffers(), buf_backup):
            b.copy_(bb)
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state(cuda_rng)
    model.train(was_training)

    if is_dist_avail_and_initialized():
        dist.all_reduce(counts)
    return {'r_neg_ratio': (counts[0] / counts[1].clamp(min=1)).item()}


def append_r_curve(csv_path, epoch, ratio):
    """Append one `epoch,r_neg_ratio` row (header written when the file is new)."""
    import os
    new = not os.path.exists(csv_path)
    with open(csv_path, 'a') as f:
        if new:
            f.write('epoch,r_neg_ratio\n')
        f.write('{},{:.6f}\n'.format(epoch, ratio))


def plot_r_curve(csv_path, png_path, title=None):
    """Plot the (epoch, r<0 ratio) curve from the csv written by `append_r_curve`."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not found: skip plotting r curve.')
        return
    epochs, ratios = [], []
    with open(csv_path) as f:
        next(f)
        for line in f:
            e, r = line.strip().split(',')
            epochs.append(int(e))
            ratios.append(float(r) * 100)
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.plot(epochs, ratios, marker='o', markersize=2.5, linewidth=1.2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Params with r < 0 (%)')
    ax.grid(alpha=0.3)
    if title:
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)


def accuracy(output, targets, topk=(1,)):
    # output: [b, n]
    # targets: [b]
    batch_size, n_classes = output.size()
    maxk = min(max(topk), n_classes)
    _, pred = torch.topk(output, k=maxk, dim=1, largest=True, sorted=True)
    pred = pred.t() # pred: [b, maxk] -> [maxk, b]
    correct = pred.eq(targets.reshape(1, -1).expand_as(pred)) # targets: [b] -> [1, b] -> [maxk, b]; correct(bool): [maxk, b]
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


class Metric:
    def __init__(self) -> None:
        self.value = 0
        self.num = 0
    
    def update(self, value, n=1):
        self.num += n
        self.value += value * n
    
    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.num, self.value], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.num = int(t[0])
        self.value = t[1]
    
    @property
    def global_avg(self):
        return self.value / self.num

class MetricLogger:
    def __init__(self) -> None:
        self.meters = defaultdict(Metric)
    
    def add_meter(self, name, meter):
        self.meters[name] = meter
    
    def update_meter(self, name, value, n):
        self.meters[name].update(value, n)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()