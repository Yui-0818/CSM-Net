"""CSM-Net 训练/验证/评估入口。

子命令：
- smoke：一个 batch 的前向/反向，检查显存与四通道输入；
- fold：单个折训练/验证；
- cv：分层交叉验证（默认 3 折；未给 --folds-dir 时按种子现场生成）；
- eval：对单个或多个 checkpoint 做 softmax 概率平均，并在 test 上评估一次。

训练与 CV 默认只读 data_clean/train；test 只在显式 eval 时读取。
最终推荐配方见 README：center_crop、90 epoch、强增强、head128、
--no-stain-gate --no-aux。
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import pathlib
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset

from data import make_dataset
from model import CSMNet

CODE = pathlib.Path(__file__).resolve().parents[1]
EXP = CODE.parent
DEFAULT_DATA = EXP / "data_clean"
DEFAULT_OUT = CODE / "run/csmnet"


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _worker_init(worker_id: int):
    s = torch.initial_seed() % (2 ** 32)
    np.random.seed(s)
    random.seed(s)


def lr_at(ep: int, base_lr: float, warmup: int, epochs: int) -> float:
    if warmup > 0 and ep < warmup:
        return base_lr * float(ep + 1) / warmup
    t = ep - warmup
    T = epochs - warmup
    if T <= 0:
        return base_lr
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * t / T))


def save_atomic(obj, path: pathlib.Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def compute_metrics(ts, ps):
    return {
        "acc": float(accuracy_score(ts, ps)),
        "macro_f1": float(f1_score(ts, ps, average="macro")),
        "weighted_f1": float(f1_score(ts, ps, average="weighted")),
        "per_class_f1": f1_score(ts, ps, average=None, zero_division=0).tolist(),
        "cm": confusion_matrix(ts, ps).tolist(),
    }


@torch.no_grad()
def evaluate(net, dl, crit):
    net.eval()
    ps, ts = [], []
    total, n = 0.0, 0
    for x, y in dl:
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        with torch.amp.autocast("cuda"):
            logits = net(x)
        loss = crit(logits.float(), y)
        ps += logits.argmax(1).cpu().tolist()
        ts += y.cpu().tolist()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return np.asarray(ps), np.asarray(ts), total / max(n, 1)


class FocalLoss(nn.Module):
    """带类别权重的 focal loss；gamma=0 时退化为加权 CE。"""

    def __init__(self, weight=None, gamma: float = 1.5,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.weight = weight
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        ce = F.cross_entropy(
            logits, target, weight=self.weight,
            label_smoothing=self.label_smoothing, reduction="none")
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


def make_criterion(weight, args):
    if getattr(args, "focal_gamma", 0.0) > 0:
        return FocalLoss(weight=weight, gamma=args.focal_gamma,
                         label_smoothing=args.ls)
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=args.ls)


class ModelEMA:
    """只对参数做 EMA；BN buffer 直接跟随当前模型。"""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = float(decay)

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for e, p in zip(self.module.parameters(), model.parameters()):
            e.mul_(d).add_(p, alpha=1.0 - d)
        for e, b in zip(self.module.buffers(), model.buffers()):
            e.copy_(b)


def build_model(args) -> CSMNet:
    in_ch = 4 if args.input_mode == "stain" else 3
    return CSMNet(
        in_ch=in_ch,
        num_classes=3,
        drop_path=args.drop_path,
        use_stain_gate=args.stain_gate,
        use_aux=args.aux,
        head_dim=args.head_dim,
        head_dropout=args.head_dropout,
    )


def build_loaders(args, fold: int):
    root = pathlib.Path(args.data_root)
    train_ds = make_dataset(root, "train", args.res, True,
                            args.input_mode, args.erase_red,
                            args.strong_aug, args.ratio_mode)
    val_ds = make_dataset(root, "train", args.res, False,
                          args.input_mode, args.erase_red, False,
                          args.ratio_mode)

    fold_dir = pathlib.Path(args.folds_dir) if args.folds_dir else None
    if fold_dir is not None:
        fold_path = fold_dir / f"fold{fold}.json"
        if not fold_path.exists():
            raise SystemExit("折文件不存在：%s；可省略 --folds-dir 让脚本按种子重新生成"
                             % fold_path)
        split = json.loads(fold_path.read_text(encoding="utf-8"))
        fit_idx = [int(i) for i in split["fit_idx"]]
        val_idx = [int(i) for i in split["val_idx"]]
    else:
        # 未给固定折文件时，用与 ConvNeXt 相同的分层规则现场生成；
        # 同一数据顺序、同一 seed 下与历史 run 的 fold*.json 等价。
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True,
                              random_state=args.folds_seed)
        splits = list(skf.split(np.zeros(len(train_ds.targets)),
                                train_ds.targets))
        if fold >= len(splits):
            raise SystemExit("fold=%d 超出 --folds=%d" % (fold, args.folds))
        fit_idx = [int(i) for i in splits[fold][0]]
        val_idx = [int(i) for i in splits[fold][1]]
        print("未给 --folds-dir：按 StratifiedKFold(%d, seed=%d) 生成 fold%d"
              % (args.folds, args.folds_seed, fold), flush=True)
    fit_ds = Subset(train_ds, fit_idx)
    val_te_ds = Subset(val_ds, val_idx)

    counts = np.bincount([train_ds.targets[i] for i in fit_idx],
                         minlength=3).astype(np.int64)
    val_counts = np.bincount([train_ds.targets[i] for i in val_idx],
                             minlength=3).astype(np.int64)

    gen = torch.Generator().manual_seed(args.seed + fold)
    dl_fit = DataLoader(
        fit_ds, batch_size=args.bs, shuffle=True, num_workers=args.workers,
        pin_memory=True, drop_last=False, generator=gen,
        worker_init_fn=_worker_init,
        persistent_workers=args.workers > 0)
    dl_val = DataLoader(
        val_te_ds, batch_size=args.bs, shuffle=False, num_workers=args.workers,
        pin_memory=True, drop_last=False,
        persistent_workers=args.workers > 0)
    return dl_fit, dl_val, counts, val_counts, len(fit_idx), len(val_idx)


def train_one(args, fold: int, out_dir: pathlib.Path):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed + fold)

    dl_fit, dl_val, counts, val_counts, n_fit, n_val = build_loaders(args, fold)
    print("运行目录:", out_dir, flush=True)
    print("fold%d | fit=%d %s | val=%d %s"
          % (fold, n_fit, counts.tolist(), n_val, val_counts.tolist()),
          flush=True)

    net = build_model(args).cuda()
    n_par = sum(p.numel() for p in net.parameters())
    print("CSM-Net | 参数量 %.3f M | in_ch=%d | stain_gate=%s | aux=%s"
          % (n_par / 1e6, 4 if args.input_mode == "stain" else 3,
             net.use_stain_gate, net.use_aux), flush=True)

    w = torch.tensor(
        counts.sum() / (len(counts) * np.maximum(counts, 1)),
        dtype=torch.float32, device="cuda")
    crit = make_criterion(w, args)
    pos0 = float(counts[0]) / max(1.0, float(counts[1] + counts[2]))
    pos12 = float(counts[1]) / max(1.0, float(counts[2]))
    bce0 = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos0], device="cuda"))
    bce12 = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos12], device="cuda"))
    print("类别权重:", [round(float(x), 4) for x in w.tolist()],
          "| aux0 pos_weight=%.3f | aux12 pos_weight=%.3f"
          % (pos0, pos12), flush=True)

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr,
                            weight_decay=args.wd)
    scaler = torch.amp.GradScaler("cuda")
    ema = ModelEMA(net, args.ema_decay)

    cfg = vars(args).copy()
    cfg.update({
        "fold": fold,
        "fit_counts": counts.tolist(),
        "val_counts": val_counts.tolist(),
        "n_params": int(n_par),
        "stamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    })
    (out_dir / "config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    log_path = out_dir / "epochs.csv"
    log_fh = open(log_path, "w", newline="", encoding="utf-8-sig")
    log_w = csv.writer(log_fh)
    log_w.writerow(["epoch", "train_loss", "val_loss", "val_acc",
                    "val_macro_f1", "ema_val_acc", "ema_val_macro_f1",
                    "lr", "sec"])

    best_f1, best_ep, best_src = -1.0, 0, ""
    last_acc = last_f1 = 0.0
    t_start = time.time()
    for ep in range(args.epochs):
        base_lr = lr_at(ep, args.lr, args.warmup, args.epochs)
        for g in opt.param_groups:
            g["lr"] = base_lr

        net.train()
        t0, tot, nb = time.time(), 0.0, 0
        for x, y in dl_fit:
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                if args.aux:
                    logits, a0, a12 = net(x, return_aux=True)
                else:
                    logits = net(x)
                loss = crit(logits.float(), y)
                if args.aux:
                    t0_aux = (y > 0).float().unsqueeze(1)
                    loss = loss + args.aux0_w * bce0(a0.float(), t0_aux)
                    mask = y > 0
                    if mask.any():
                        t12 = (y[mask] == 2).float().unsqueeze(1)
                        loss = loss + args.aux12_w * bce12(
                            a12[mask].float(), t12)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            ema.update(net)
            tot += loss.item()
            nb += 1

        ps, ts, v_loss = evaluate(net, dl_val, crit)
        m = compute_metrics(ts, ps)
        ps_e, ts_e, v_loss_e = evaluate(ema.module, dl_val, crit)
        me = compute_metrics(ts_e, ps_e)
        sec = time.time() - t0
        last_acc, last_f1 = m["acc"], m["macro_f1"]
        print("[ep %2d/%d] loss %.4f | raw val_loss %.4f acc %.4f 宏F1 %.4f "
              "| ema acc %.4f 宏F1 %.4f | lr %.2e | %.1fs"
              % (ep + 1, args.epochs, tot / max(nb, 1), v_loss,
                 m["acc"], m["macro_f1"], me["acc"], me["macro_f1"],
                 base_lr, sec), flush=True)
        log_w.writerow([ep + 1, round(tot / max(nb, 1), 4),
                        round(v_loss, 4), round(m["acc"], 4),
                        round(m["macro_f1"], 4), round(me["acc"], 4),
                        round(me["macro_f1"], 4), round(base_lr, 8),
                        round(sec, 1)])
        log_fh.flush()

        # P1 短跑里 EMA 只有 ~120 步，远未收敛；raw/EMA 同时评估，
        # 谁在 val 上高就用谁，避免把随机初始化的 EMA 当最终模型。
        cand = [(m["macro_f1"], "raw", net),
                (me["macro_f1"], "ema", ema.module)]
        cand_f1, cand_src, cand_net = max(cand, key=lambda t: t[0])
        if cand_f1 > best_f1:
            best_f1, best_ep, best_src = cand_f1, ep + 1, cand_src
            save_atomic(cand_net.state_dict(), out_dir / "best.pth")
            print("    -> 保存最佳 (%s, 宏F1 %.4f)"
                  % (best_src, best_f1), flush=True)

    if args.save_last:
        save_atomic(net.state_dict(), out_dir / "last.pth")
        save_atomic(ema.module.state_dict(), out_dir / "last_ema.pth")
    log_fh.close()
    total_sec = time.time() - t_start
    print("训练完成：best ep%d %s 宏F1 %.4f | 末轮 raw acc %.4f / 宏F1 %.4f "
          "| 总 %.1f 分钟"
          % (best_ep, best_src, best_f1, last_acc, last_f1,
             total_sec / 60), flush=True)
    return {
        "fold": fold,
        "n_fit": n_fit,
        "n_val": n_val,
        "best_ep": best_ep,
        "best_src": best_src,
        "best_f1": best_f1,
        "last_acc": last_acc,
        "last_f1": last_f1,
        "total_sec": total_sec,
    }


def cmd_smoke(args):
    seed_all(args.seed)
    root = pathlib.Path(args.data_root)
    ds = make_dataset(root, "train", args.res, True,
                      args.input_mode, args.erase_red, args.strong_aug,
                      args.ratio_mode)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=0,
                    pin_memory=True, drop_last=False)
    x, y = next(iter(dl))
    print("batch", tuple(x.shape), "| labels", y.tolist(), flush=True)
    for i, name in enumerate(["R", "G", "B", "S"][:x.shape[1]]):
        print("  %s: min %.3f max %.3f mean %.3f"
              % (name, x[:, i].min().item(), x[:, i].max().item(),
                 x[:, i].mean().item()), flush=True)

    net = build_model(args).cuda()
    n_par = sum(p.numel() for p in net.parameters())
    counts = np.bincount(ds.targets, minlength=3)
    w = torch.tensor(
        counts.sum() / (len(counts) * np.maximum(counts, 1)),
        dtype=torch.float32, device="cuda")
    crit = make_criterion(w, args)
    bce0 = nn.BCEWithLogitsLoss()
    bce12 = nn.BCEWithLogitsLoss()

    x = x.cuda(non_blocking=True)
    y = y.cuda(non_blocking=True)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr,
                            weight_decay=args.wd)
    opt.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda"):
        if args.aux:
            logits, a0, a12 = net(x, return_aux=True)
        else:
            logits = net(x)
        loss = crit(logits.float(), y)
        if args.aux:
            loss = loss + args.aux0_w * bce0(
                a0.float(), (y > 0).float().unsqueeze(1))
            mask = y > 0
            if mask.any():
                loss = loss + args.aux12_w * bce12(
                    a12[mask].float(), (y[mask] == 2).float().unsqueeze(1))
    loss.backward()
    opt.step()
    torch.cuda.synchronize()
    print("SMOKE | 参数量 %.3f M | loss %.4f | %.2fs | 峰值显存 %.2f GB"
          % (n_par / 1e6, loss.item(), time.time() - t0,
             torch.cuda.max_memory_allocated() / 2 ** 30), flush=True)
    if args.aux:
        print("  logits", tuple(logits.shape), "| aux0", tuple(a0.shape),
              "| aux12", tuple(a12.shape), flush=True)


def cmd_fold(args):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else (
        DEFAULT_OUT / f"fold{args.fold}_{stamp}")
    train_one(args, args.fold, out_dir)


def cmd_cv(args):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = pathlib.Path(args.out_dir) if args.out_dir else (
        DEFAULT_OUT / f"cv_{stamp}")
    rows = []
    for fold in range(args.folds):
        res = train_one(args, fold, root / f"fold{fold}")
        rows.append(res)
    summary = root / "cv_summary.csv"
    with open(summary, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["fold", "n_fit", "n_val", "best_ep", "best_val_macro_f1",
                    "last_val_acc", "last_val_macro_f1", "total_sec"])
        for r in rows:
            w.writerow([r["fold"], r["n_fit"], r["n_val"], r["best_ep"],
                        round(r["best_f1"], 4), round(r["last_acc"], 4),
                        round(r["last_f1"], 4), round(r["total_sec"], 1)])
    best = np.asarray([r["best_f1"] for r in rows], dtype=float)
    last = np.asarray([r["last_f1"] for r in rows], dtype=float)
    print("CV best-val 宏F1 %.4f ± %.4f | 末轮 %.4f ± %.4f"
          % (best.mean(), best.std(ddof=1) if len(best) > 1 else 0.0,
             last.mean(), last.std(ddof=1) if len(last) > 1 else 0.0),
          flush=True)
    print("CV 汇总:", summary, flush=True)


@torch.no_grad()
def _predict_probs(net, dl):
    net.eval()
    probs, ts = [], []
    for x, y in dl:
        x = x.cuda(non_blocking=True)
        with torch.amp.autocast("cuda"):
            logits = net(x)
        probs.append(torch.softmax(logits.float(), 1).cpu())
        ts += y.tolist()
    return torch.cat(probs).numpy(), np.asarray(ts)


def cmd_eval(args):
    ckpts = list(args.checkpoints) if args.checkpoints else []
    if args.checkpoint:
        ckpts.insert(0, args.checkpoint)
    if not ckpts:
        raise SystemExit("eval 需要 --checkpoint 或 --checkpoints")
    root = pathlib.Path(args.data_root)
    ds = make_dataset(root, "test", args.res, False,
                      args.input_mode, args.erase_red, False, args.ratio_mode)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=False,
                    num_workers=args.workers, pin_memory=True)

    prob_sum, ts = None, None
    for ckpt in ckpts:
        net = build_model(args).cuda()
        state = torch.load(ckpt, map_location="cuda", weights_only=True)
        if any(k.startswith("module.") for k in state):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        net.load_state_dict(state)
        p, t = _predict_probs(net, dl)
        prob_sum = p if prob_sum is None else prob_sum + p
        ts = t
    probs = prob_sum / len(ckpts)
    ps = probs.argmax(1)
    m = compute_metrics(ts, ps)
    print("test n=%d | acc %.4f | 宏F1 %.4f | 加权F1 %.4f"
          % (len(ts), m["acc"], m["macro_f1"], m["weighted_f1"]), flush=True)
    print("per-class F1", [round(x, 4) for x in m["per_class_f1"]],
          "| CM", m["cm"], flush=True)

    out_dir = pathlib.Path(args.out_dir) if args.out_dir else (
        DEFAULT_OUT / "eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "eval_test.csv", "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "checkpoints", "n_checkpoints", "res",
                    "test_acc", "test_macro_f1", "test_weighted_f1",
                    "per_class_f1", "confusion_matrix", "n"])
        w.writerow(["csmnet", "|".join(ckpts), len(ckpts), args.res,
                    round(m["acc"], 4), round(m["macro_f1"], 4),
                    round(m["weighted_f1"], 4),
                    "|".join(str(round(x, 4)) for x in m["per_class_f1"]),
                    "|".join("|".join(str(v) for v in row) for row in m["cm"]),
                    len(ts)])
    with open(out_dir / "test_predictions.csv", "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "label", "pred", "p0", "p1", "p2"])
        for i, (path, label) in enumerate(ds.samples):
            w.writerow([path, int(label), int(ps[i]),
                        round(float(probs[i, 0]), 6),
                        round(float(probs[i, 1]), 6),
                        round(float(probs[i, 2]), 6)])
    print("写出:", out_dir / "eval_test.csv", flush=True)


def add_common(p):
    p.add_argument("--data-root", default=str(DEFAULT_DATA))
    p.add_argument("--folds-dir", default="",
                   help="固定折文件目录；省略时按 --folds/--folds-seed 现场生成")
    p.add_argument("--folds", type=int, default=3,
                   help="未给 --folds-dir 时的分层折数")
    p.add_argument("--folds-seed", type=int, default=0,
                   help="未给 --folds-dir 时的折划分种子")
    p.add_argument("--input-mode", choices=["stain", "rgb"], default="stain")
    p.add_argument("--ratio-mode",
                   choices=["stretch", "letterbox", "center_crop"],
                   default="stretch",
                   help="几何预处理：直接拉伸 / 保比例 pad / 保比例中心裁剪")
    p.add_argument("--erase-red", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--stain-gate", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--aux", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--res", type=int, default=224)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--ls", type=float, default=0.05)
    p.add_argument("--focal-gamma", type=float, default=0.0,
                   help=">0 时主损失改用 focal loss")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--drop-path", type=float, default=0.05)
    p.add_argument("--strong-aug", action=argparse.BooleanOptionalAction,
                   default=False)
    p.add_argument("--head-dim", type=int, default=256)
    p.add_argument("--head-dropout", type=float, default=0.3)
    p.add_argument("--save-last", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="是否保存 last/last_ema；默认只留 best.pth + CSV")
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--aux0-w", type=float, default=0.3)
    p.add_argument("--aux12-w", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--tag", default="")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="CSM-Net：红线擦除 + 荧光素先验 + 可消融的纯卷积三分类")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_smoke = sub.add_parser("smoke", help="一个 batch 的前向/反向")
    add_common(p_smoke)

    p_fold = sub.add_parser("fold", help="单个折的训练/验证")
    add_common(p_fold)
    p_fold.add_argument("--fold", type=int, default=0)
    p_fold.add_argument("--out-dir", default="")

    p_cv = sub.add_parser("cv", help="完整 3 折交叉验证")
    add_common(p_cv)
    p_cv.add_argument("--out-dir", default="")

    p_eval = sub.add_parser("eval", help="在 data_clean/test 上评估")
    add_common(p_eval)
    p_eval.add_argument("--checkpoint", default="")
    p_eval.add_argument("--checkpoints", nargs="+", default=[])
    p_eval.add_argument("--out-dir", default="")

    args = parser.parse_args(argv)
    torch.backends.cudnn.benchmark = True
    if args.cmd == "smoke":
        cmd_smoke(args)
    elif args.cmd == "fold":
        cmd_fold(args)
    elif args.cmd == "cv":
        cmd_cv(args)
    elif args.cmd == "eval":
        cmd_eval(args)


if __name__ == "__main__":
    main()
