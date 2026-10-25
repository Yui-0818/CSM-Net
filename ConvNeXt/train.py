"""ConvNeXt 三分类训练（实验一第二种模型，纯 CNN 对照）。

模型与预训练权重来自 facebookresearch/ConvNeXt（MIT License）：
  https://github.com/facebookresearch/ConvNeXt
本文件只保留上游 `models/convnext.py` 的模型定义，训练/验证/测试流程按本实验
统一的协议重写，以便与 MedMamba 使用同一份数据划分和同一套评估口径。

协议要点：
  - 只使用 Experiment1/data/train 做训练与验证；data/test 默认不读。
  - stage1 的 fit/val 划分用 torch.random_split(seed=split_seed)，与 MedMamba
    阶段一完全相同；cv 用 StratifiedKFold(seed=folds_seed)，与 MedMamba CV 相同。
  - 测试集只在显式执行 `eval` 子命令时读取一次，避免调参期间泄漏。
  - 每个 epoch 保存 epN.pth 与 epN_opt.pth，支持断点续跑与 STOP 文件优雅暂停。

常用命令（在 Experiment1/code/ConvNeXt 下执行）：
  python train.py stage1 --arm transfer --save-last
  python train.py cv --folds 3 --epochs 30
  python train.py eval --checkpoint run/.../last.pth

本文件不自动启动任何训练。
"""
import argparse
import csv
import datetime
import json
import math
import os
import pathlib
import random
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

HERE = pathlib.Path(__file__).resolve().parent          # Experiment1/code/ConvNeXt
CODE = HERE.parent                                      # Experiment1/code
EXP = HERE.parents[1]                                   # Experiment1
sys.path.insert(0, str(HERE))

# 上游文件用 @register_model 注册同名模型，timm 已内置 convnext_*，导入时会提示覆盖；
# 这里直接调用工厂函数，不依赖 timm registry，屏蔽这条无害警告。
warnings.filterwarnings("ignore", message=r"Overwriting convnext_.* in registry")

from models import (convnext_base, convnext_large, convnext_small,  # noqa: E402
                    convnext_tiny, convnext_xlarge)

DEFAULT_DATA_ROOT = EXP / "data_clean"
DEFAULT_PRETRAINED = EXP / "weights" / "ConvNeXt" / "convnext_tiny_1k_224_ema.pth"

MODEL_FACTORY = {
    "convnext_tiny": convnext_tiny,
    "convnext_small": convnext_small,
    "convnext_base": convnext_base,
    "convnext_large": convnext_large,
    "convnext_xlarge": convnext_xlarge,
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def save_atomic(obj, path):
    """先写 .tmp 再 os.replace，中断时不会留下半个 checkpoint。"""
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, str(path))


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(name, num_classes, drop_path, pretrained=""):
    """构造 ConvNeXt；pretrained 给定时只加载 backbone，分类头重新初始化。"""
    net = MODEL_FACTORY[name](num_classes=num_classes, drop_path_rate=drop_path)
    if pretrained:
        p = pathlib.Path(pretrained)
        if not p.exists():
            raise SystemExit(
                "预训练权重不存在：%s\n"
                "请从 https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth 下载，"
                "放到 %s" % (p, p)
            )
        # 官方 checkpoint 含 model/optimizer/epoch 等字段；weights_only=False 仅为兼容
        # 官方的 args 字段，文件来源固定为 facebookresearch/ConvNeXt 的发布地址。
        sd = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(sd, nn.Module):
            sd = sd.state_dict()
        if isinstance(sd, dict) and isinstance(sd.get("model"), dict):
            sd = sd["model"]
        if isinstance(sd, dict) and isinstance(sd.get("state_dict"), dict):
            sd = sd["state_dict"]
        sd = {str(k).replace("module.", "", 1): v for k, v in sd.items()}
        body = {k: v for k, v in sd.items() if not k.startswith("head.")}
        missing, unexpected = net.load_state_dict(body, strict=False)
        bad = [k for k in missing if not k.startswith("head.")]
        if bad or unexpected:
            raise SystemExit("权重键不匹配：missing=%s unexpected=%s"
                             % (bad[:5], list(unexpected)[:5]))
        nn.init.trunc_normal_(net.head.weight, std=0.02)
        nn.init.zeros_(net.head.bias)
        print("迁移初始化：%s | 已加载 body 参数 %d 个，head 重新初始化"
              % (p.name, len(body)), flush=True)
    return net


def make_transforms(res, train):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(
                res, scale=(0.7, 1.0), ratio=(0.75, 1.333),
                interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        # 与 MedMamba 阶段一相同：直接拉伸到方形，保持输入口径一致
        transforms.Resize((res, res)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def lr_at(ep, base_lr, warmup, epochs):
    """线性 warmup（可选）+ 余弦衰减；warmup=0 时就是普通余弦。"""
    if warmup > 0 and ep < warmup:
        return base_lr * float(ep + 1) / warmup
    t = ep - warmup
    T = epochs - warmup
    if T <= 0:
        return base_lr
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * t / T))


@torch.no_grad()
def predict_probs(net, dl):
    net.eval()
    probs, ts = [], []
    for x, y in dl:
        x = x.cuda(non_blocking=True)
        with torch.amp.autocast("cuda"):
            logits = net(x)
        probs.append(torch.softmax(logits.float(), 1).cpu())
        ts += y.tolist()
    return torch.cat(probs).numpy(), np.asarray(ts)


def evaluate(net, dl):
    probs, ts = predict_probs(net, dl)
    return probs.argmax(1), ts


@torch.no_grad()
def evaluate_with_loss(net, dl, crit):
    """一次前向同时得到预测、标签与验证损失，用于观察过拟合。"""
    net.eval()
    ps, ts = [], []
    total, n = 0.0, 0
    for x, y in dl:
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        with torch.amp.autocast("cuda"):
            logits = net(x)
        # AMP 下 logits 可能是 fp16，而 CE 的类别权重是 fp32；先转 fp32 再算损失
        loss = crit(logits.float(), y)
        ps += logits.argmax(1).cpu().tolist()
        ts += y.cpu().tolist()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return np.asarray(ps), np.asarray(ts), total / max(n, 1)


def compute_metrics(ts, ps):
    return {
        "acc": float(accuracy_score(ts, ps)),
        "macro_f1": float(f1_score(ts, ps, average="macro")),
        "weighted_f1": float(f1_score(ts, ps, average="weighted")),
        "per_class_f1": f1_score(ts, ps, average=None, zero_division=0).tolist(),
        "cm": confusion_matrix(ts, ps).tolist(),
    }


def stage1_split(n, val_frac, split_seed):
    """与 MedMamba stage1 完全一致的 random_split 划分。"""
    n_val = max(1, int(n * val_frac))
    fit, val = random_split(
        range(n), [n - n_val, n_val],
        generator=torch.Generator().manual_seed(split_seed))
    return list(fit), list(val)


def build_folds(n_splits, seed, targets):
    """按 ImageFolder 顺序做分层折划分，索引与 MedMamba CV 完全一致。"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(fit.tolist(), val.tolist())
            for fit, val in skf.split(np.zeros(len(targets)), targets)]


def read_epochs(path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, encoding="utf-8-sig") as fh:
        return [r for r in csv.DictReader(fh) if str(r.get("epoch", "")).isdigit()]


def train_one(args, fit_idx, val_idx, out_dir, stamp, save_last=False):
    """训练一个 fit/val 划分；返回末轮与最佳轮的验证指标。"""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("运行目录:", out_dir, flush=True)

    tf_fit = make_transforms(args.res, True)
    tf_ev = make_transforms(args.res, False)
    train_root = pathlib.Path(args.data_root) / "train"
    ds_fit_full = datasets.ImageFolder(str(train_root), tf_fit)
    ds_val_full = datasets.ImageFolder(str(train_root), tf_ev)
    ds_fit = Subset(ds_fit_full, fit_idx)
    ds_val = Subset(ds_val_full, val_idx)
    y_fit = np.asarray([ds_fit_full.targets[i] for i in fit_idx])
    n_classes = len(ds_fit_full.classes)
    print("类别映射:", ds_fit_full.class_to_idx,
          "| fit=%d val=%d" % (len(fit_idx), len(val_idx)), flush=True)

    counts = np.bincount(y_fit, minlength=n_classes).astype(float)
    w = torch.tensor(counts.sum() / (len(counts) * counts),
                     dtype=torch.float32, device="cuda")
    print("fit 类别计数:", counts.tolist(),
          "-> 损失权重:", [round(x, 3) for x in w.tolist()], flush=True)

    net = build_model(args.model, n_classes, args.drop_path,
                      args.pretrained if args.arm == "transfer" else "").cuda()
    n_par = sum(p.numel() for p in net.parameters())
    print("模型 %s | 参数量 %.2f M | res=%d bs=%d lr=%.1e epochs=%d"
          % (args.model, n_par / 1e6, args.res, args.bs, args.lr, args.epochs),
          flush=True)

    def set_backbone_grad(flag):
        for name, p in net.named_parameters():
            if not name.startswith("head."):
                p.requires_grad = flag

    if args.freeze_epochs > 0 and args.arm == "transfer":
        set_backbone_grad(False)
        print("冻结 backbone %d 轮，只训练分类头" % args.freeze_epochs, flush=True)

    head_params = [p for n, p in net.named_parameters() if n.startswith("head.")]
    backbone_params = [p for n, p in net.named_parameters() if not n.startswith("head.")]
    opt = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr},
        {"params": head_params, "lr": args.lr * args.head_lr_mult},
    ], weight_decay=args.wd)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=args.ls)
    scaler = torch.amp.GradScaler("cuda")

    dl_fit = DataLoader(ds_fit, batch_size=args.bs, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=args.bs, shuffle=False,
                        num_workers=args.num_workers)

    if getattr(args, "smoke", False):
        net.train()
        x, y = next(iter(dl_fit))
        x, y = x.cuda(), y.cuda()
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            out = net(x)
        loss = crit(out.float(), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        torch.cuda.synchronize()
        ps, ts = evaluate(net, dl_val)
        print("SMOKE: 1 step %.1fs | loss=%.3f | 输出%s | val(随机/预训练头) acc=%.3f | 峰值显存 %.2fGB"
              % (time.time() - t0, loss.item(), tuple(out.shape),
                 accuracy_score(ts, ps),
                 torch.cuda.max_memory_allocated() / 2**30), flush=True)
        return {"smoke": True}

    log_path = out_dir / "epochs.csv"
    need_header = not log_path.exists() or log_path.stat().st_size == 0
    if not need_header:
        with open(log_path, encoding="utf-8-sig") as fh:
            header = fh.readline()
        if "val_loss" not in header:
            raise SystemExit("该目录的 epochs.csv 是旧版（无 val_loss），"
                             "不能与新代码续跑；请换一个新的 --out-dir")
    log_fh = open(log_path, "a", newline="", encoding="utf-8-sig")
    log_w = csv.writer(log_fh)
    if need_header:
        log_w.writerow(["epoch", "train_loss", "val_loss", "val_acc",
                        "val_macro_f1", "lr", "sec"])

    prev = read_epochs(log_path)
    done = 0
    best_f1, best_ep = -1.0, -1
    last_acc = last_f1 = 0.0
    if prev:
        done = int(prev[-1]["epoch"])
        best_row = max(prev, key=lambda r: float(r["val_macro_f1"]))
        best_ep, best_f1 = int(best_row["epoch"]), float(best_row["val_macro_f1"])
        last_acc, last_f1 = float(prev[-1]["val_acc"]), float(prev[-1]["val_macro_f1"])
        if done >= args.epochs:
            ck = out_dir / ("ep%d.pth" % done)
            if not ck.exists():
                raise SystemExit("已完成 %d 轮但缺少 %s" % (done, ck.name))
            if save_last and not (out_dir / "last.pth").exists():
                net.load_state_dict(torch.load(ck, map_location="cuda", weights_only=True))
                save_atomic(net.state_dict(), out_dir / "last.pth")
            print("该划分已完成 %d/%d 轮，跳过" % (done, args.epochs), flush=True)
            log_fh.close()
            return {"best_ep": best_ep, "best_f1": best_f1,
                    "last_acc": last_acc, "last_f1": last_f1,
                    "done": done, "stopped": False}
        ck = out_dir / ("ep%d.pth" % done)
        ok = out_dir / ("ep%d_opt.pth" % done)
        if not (ck.exists() and ok.exists()):
            raise SystemExit("epochs.csv 已有 %d 轮，但缺少 %s 或 %s，无法续跑"
                             % (done, ck.name, ok.name))
        net.load_state_dict(torch.load(ck, map_location="cuda", weights_only=True))
        opt.load_state_dict(torch.load(ok, map_location="cuda", weights_only=False))
        if args.freeze_epochs > 0 and done >= args.freeze_epochs:
            set_backbone_grad(True)
        print("resume: 已完成 %d 轮，从 ep%d 起" % (done, done + 1), flush=True)
        if not (out_dir / "best.pth").exists() and best_ep > 0:
            bp = out_dir / ("ep%d.pth" % best_ep)
            if bp.exists():
                net.load_state_dict(torch.load(bp, map_location="cuda", weights_only=True))
                save_atomic(net.state_dict(), out_dir / "best.pth")
                net.load_state_dict(torch.load(ck, map_location="cuda", weights_only=True))

    stopped = False
    completed = done
    for ep in range(done, args.epochs):
        if args.stop_file and pathlib.Path(args.stop_file).exists():
            stopped = True
            print("检测到 STOP 文件（%s），在第 %d 轮开始前暂停"
                  % (args.stop_file, ep + 1), flush=True)
            break
        if args.freeze_epochs > 0 and ep == args.freeze_epochs and args.arm == "transfer":
            set_backbone_grad(True)
            print("ep%d：解冻 backbone，全模型微调" % (ep + 1), flush=True)

        base_lr = lr_at(ep, args.lr, args.warmup, args.epochs)
        opt.param_groups[0]["lr"] = base_lr
        opt.param_groups[1]["lr"] = base_lr * args.head_lr_mult

        # 每轮重播 RNG，断点续跑与一次性跑完的批次顺序/增强一致
        seed_all(args.seed * 1000 + ep)
        net.train()
        t0, tot, nb = time.time(), 0.0, 0
        for x, y in dl_fit:
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = net(x)
            loss = crit(logits.float(), y)
            scaler.scale(loss).backward()
            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            tot += loss.item()
            nb += 1
        ps, ts, v_loss = evaluate_with_loss(net, dl_val, crit)
        m = compute_metrics(ts, ps)
        el = time.time() - t0
        print("[ep %2d/%d] loss %.3f | val_loss %.3f | val_acc %.3f | val_宏F1 %.3f | lr %.2e | %.1f 分钟"
              % (ep + 1, args.epochs, tot / max(nb, 1), v_loss, m["acc"], m["macro_f1"],
                 base_lr, el / 60), flush=True)
        log_w.writerow([ep + 1, round(tot / max(nb, 1), 4), round(v_loss, 4),
                        round(m["acc"], 4), round(m["macro_f1"], 4),
                        round(base_lr, 8), round(el, 1)])
        log_fh.flush()
        save_atomic(net.state_dict(), out_dir / ("ep%d.pth" % (ep + 1)))
        save_atomic(opt.state_dict(), out_dir / ("ep%d_opt.pth" % (ep + 1)))
        last_acc, last_f1 = m["acc"], m["macro_f1"]
        completed = ep + 1
        if m["macro_f1"] > best_f1:
            best_f1, best_ep = m["macro_f1"], ep + 1
            save_atomic(net.state_dict(), out_dir / "best.pth")
            print("    -> 保存最佳 (宏F1 %.3f)" % best_f1, flush=True)

    if stopped:
        if save_last:
            save_atomic(net.state_dict(), out_dir / "last.pth")
        print("已暂停：完成 %d/%d 轮；删除 STOP 文件后重跑同一命令即可续上"
              % (completed, args.epochs), flush=True)
        log_fh.close()
        return {"best_ep": best_ep, "best_f1": best_f1,
                "last_acc": last_acc, "last_f1": last_f1,
                "done": completed, "stopped": True}

    if save_last:
        save_atomic(net.state_dict(), out_dir / "last.pth")
        print("已保存 last.pth", flush=True)
    log_fh.close()
    print("训练完成：best ep%d 宏F1 %.4f | 末轮 acc %.4f / 宏F1 %.4f"
          % (best_ep, best_f1, last_acc, last_f1), flush=True)
    return {"best_ep": best_ep, "best_f1": best_f1,
            "last_acc": last_acc, "last_f1": last_f1,
            "done": args.epochs, "stopped": False}


def cmd_stage1(args, stamp):
    ds = datasets.ImageFolder(str(pathlib.Path(args.data_root) / "train"))
    if args.split_json:
        split = json.loads(pathlib.Path(args.split_json).read_text(encoding="utf-8"))
        fit_idx = [int(i) for i in split["fit_idx"]]
        val_idx = [int(i) for i in split["val_idx"]]
    else:
        fit_idx, val_idx = stage1_split(len(ds), args.val_frac, args.split_seed)
    print("划分: fit=%d val=%d | split_seed=%d | arm=%s"
          % (len(fit_idx), len(val_idx), args.split_seed, args.arm), flush=True)
    if args.dry_run:
        return

    suffix = ("_" + args.tag) if args.tag else ""
    out_dir = (pathlib.Path(args.out_dir) if args.out_dir
               else CODE / "run" / "convnext"
               / ("stage1_%s_%s%s" % (args.arm, stamp, suffix)))
    res = train_one(args, fit_idx, val_idx, out_dir, stamp,
                    save_last=args.save_last)
    if args.smoke or res.get("stopped"):
        return

    if args.test:
        ckpt = out_dir / "best.pth"
        print("\n===== 测试集评估（仅此一次）=====", flush=True)
        test_ds = datasets.ImageFolder(str(pathlib.Path(args.data_root) / "test"),
                                       make_transforms(args.res, False))
        net = build_model(args.model, len(test_ds.classes), args.drop_path).cuda()
        net.load_state_dict(torch.load(ckpt, map_location="cuda", weights_only=True))
        dl = DataLoader(test_ds, batch_size=args.bs, shuffle=False,
                        num_workers=args.num_workers)
        ps, ts = evaluate(net, dl)
        m = compute_metrics(ts, ps)
        print("测试 acc %.4f | 宏F1 %.4f | 加权F1 %.4f"
              % (m["acc"], m["macro_f1"], m["weighted_f1"]), flush=True)
        for i, cname in enumerate(test_ds.classes):
            print("  类 %s: F1 %.3f" % (cname, m["per_class_f1"][i]), flush=True)
        print("混淆矩阵(行=真值):\n", np.asarray(m["cm"]), flush=True)
        rows = [[args.model, args.arm, str(args.pretrained) if args.arm == "transfer" else "-",
                 args.seed, args.split_seed, args.res, args.bs, args.lr, args.wd, args.ls,
                 args.epochs, res["best_ep"], round(res["best_f1"], 4),
                 round(m["acc"], 4), round(m["macro_f1"], 4),
                 "|".join(str(round(float(v), 3)) for v in m["per_class_f1"]),
                 "|".join(str(int(v)) for v in np.asarray(m["cm"]).ravel()), stamp]]
        summ = out_dir / "summary.csv"
        new = not summ.exists() or summ.stat().st_size == 0
        with open(summ, "a", newline="", encoding="utf-8-sig") as fh:
            ww = csv.writer(fh)
            if new:
                ww.writerow(["model", "arm", "pretrained", "seed", "split_seed", "res", "bs",
                             "lr", "wd", "ls", "epochs", "best_epoch", "best_val_macro_f1",
                             "test_acc", "test_macro_f1", "per_class_f1",
                             "confusion_matrix", "timestamp"])
            ww.writerows(rows)
        print("日志:", out_dir / "epochs.csv", "\n汇总:", summ, flush=True)


def cmd_cv(args, stamp):
    ds = datasets.ImageFolder(str(pathlib.Path(args.data_root) / "train"))
    folds = build_folds(args.folds, args.folds_seed, ds.targets)
    suffix = ("_" + args.tag) if args.tag else ""
    root = (pathlib.Path(args.out_dir) if args.out_dir
            else CODE / "run" / "convnext"
            / ("cv_%s_ep%d_lr%g_ls%g_w%d%s"
               % (stamp, args.epochs, args.lr, args.ls, args.warmup, suffix)))
    print("CV 根目录:", root, flush=True)

    for k, (fit_idx, val_idx) in enumerate(folds):
        fit_counts = np.bincount([ds.targets[i] for i in fit_idx],
                                 minlength=len(ds.classes)).tolist()
        val_counts = np.bincount([ds.targets[i] for i in val_idx],
                                 minlength=len(ds.classes)).tolist()
        print("fold%d fit=%d %s | val=%d %s"
              % (k, len(fit_idx), fit_counts, len(val_idx), val_counts), flush=True)
        if args.dry_run:
            continue
        root.mkdir(parents=True, exist_ok=True)
        (root / "folds").mkdir(exist_ok=True)
        (root / "folds" / ("fold%d.json" % k)).write_text(
            json.dumps({"fit_idx": fit_idx, "val_idx": val_idx}), encoding="utf-8")
        fold_dir = root / ("fold%d" % k)
        prev = read_epochs(fold_dir / "epochs.csv")
        if (prev and int(prev[-1]["epoch"]) >= args.epochs
                and (fold_dir / "last.pth").exists()):
            print("fold%d 已完成，跳过" % k, flush=True)
            continue
        res = train_one(args, fit_idx, val_idx, fold_dir, stamp, save_last=True)
        if res.get("stopped"):
            print("收到 STOP，CV 在第 %d 折暂停" % k, flush=True)
            break

    if args.dry_run:
        return

    rows = []
    for k in range(args.folds):
        fold_dir = root / ("fold%d" % k)
        prev = read_epochs(fold_dir / "epochs.csv")
        if not prev or int(prev[-1]["epoch"]) < args.epochs:
            continue
        best_row = max(prev, key=lambda r: float(r["val_macro_f1"]))
        fit_idx, val_idx = folds[k]
        rows.append([
            k, len(fit_idx),
            "|".join(str(x) for x in np.bincount([ds.targets[i] for i in fit_idx],
                                                minlength=len(ds.classes)).tolist()),
            len(val_idx),
            "|".join(str(x) for x in np.bincount([ds.targets[i] for i in val_idx],
                                                minlength=len(ds.classes)).tolist()),
            int(prev[-1]["epoch"]), round(float(prev[-1]["val_acc"]), 4),
            round(float(prev[-1]["val_macro_f1"]), 4),
            int(best_row["epoch"]), round(float(best_row["val_macro_f1"]), 4),
            round(sum(float(r["sec"]) for r in prev), 1),
        ])
    summ = root / "cv_summary.csv"
    with open(summ, "w", newline="", encoding="utf-8-sig") as fh:
        ww = csv.writer(fh)
        ww.writerow(["fold", "n_fit", "fit_counts", "n_val", "val_counts",
                     "last_epoch", "last_val_acc", "last_val_macro_f1",
                     "best_epoch", "best_val_macro_f1", "total_sec"])
        ww.writerows(rows)
    print("\nCV 汇总：%d/%d 折完成" % (len(rows), args.folds), flush=True)
    if rows:
        last = np.asarray([r[7] for r in rows], dtype=float)
        best = np.asarray([r[9] for r in rows], dtype=float)
        print("末轮 val 宏F1 %.4f ± %.4f | best-val 宏F1 %.4f ± %.4f"
              % (last.mean(), last.std(ddof=1) if len(last) > 1 else 0.0,
                 best.mean(), best.std(ddof=1) if len(best) > 1 else 0.0), flush=True)
        print("明细:", summ, flush=True)


def cmd_eval(args):
    ckpts = list(args.checkpoints)
    if args.checkpoint:
        if ckpts:
            raise SystemExit("--checkpoint 与 --checkpoints 二选一")
        ckpts = [args.checkpoint]
    if not ckpts:
        raise SystemExit("需要 --checkpoint 或 --checkpoints")

    test_ds = datasets.ImageFolder(str(pathlib.Path(args.data_root) / "test"),
                                   make_transforms(args.res, False))
    net = build_model(args.model, len(test_ds.classes), 0.0).cuda()
    dl = DataLoader(test_ds, batch_size=args.bs, shuffle=False,
                    num_workers=args.num_workers)
    probs, ts = None, None
    for c in ckpts:
        ckpt = pathlib.Path(c)
        if not ckpt.exists():
            raise SystemExit("checkpoint 不存在：%s" % ckpt)
        sd = torch.load(ckpt, map_location="cuda", weights_only=True)
        if isinstance(sd, dict) and isinstance(sd.get("model"), dict):
            sd = sd["model"]
        net.load_state_dict(sd)
        p, ts = predict_probs(net, dl)
        probs = p if probs is None else probs + p
    probs = probs / len(ckpts)
    ps = probs.argmax(1)
    m = compute_metrics(ts, ps)
    print("checkpoints:", " | ".join(str(c) for c in ckpts), flush=True)
    print("测试 acc %.4f | 宏F1 %.4f | 加权F1 %.4f"
          % (m["acc"], m["macro_f1"], m["weighted_f1"]), flush=True)
    pre = precision_score(ts, ps, average=None, zero_division=0)
    rec = recall_score(ts, ps, average=None, zero_division=0)
    for i, cname in enumerate(test_ds.classes):
        print("  类 %s: precision %.3f | recall %.3f | F1 %.3f"
              % (cname, pre[i], rec[i], m["per_class_f1"][i]), flush=True)
    print("混淆矩阵(行=真值):\n", np.asarray(m["cm"]), flush=True)
    out = (pathlib.Path(args.out) if args.out
           else pathlib.Path(ckpts[0]).parent
           / ("eval_test.csv" if len(ckpts) == 1 else "eval_test_ensemble.csv"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        ww = csv.writer(fh)
        ww.writerow(["model", "checkpoints", "n_checkpoints", "res", "test_acc",
                     "test_macro_f1", "test_weighted_f1", "per_class_f1",
                     "confusion_matrix", "n"])
        ww.writerow([args.model, "|".join(str(c) for c in ckpts), len(ckpts),
                     args.res, round(m["acc"], 4), round(m["macro_f1"], 4),
                     round(m["weighted_f1"], 4),
                     "|".join(str(round(float(v), 3)) for v in m["per_class_f1"]),
                     "|".join(str(int(v)) for v in np.asarray(m["cm"]).ravel()),
                     len(ts)])
    print("已写:", out, flush=True)
    pred_path = out.parent / ("test_predictions.csv" if len(ckpts) == 1
                              else "test_predictions_ensemble.csv")
    with open(pred_path, "w", newline="", encoding="utf-8-sig") as fh:
        ww = csv.writer(fh)
        ww.writerow(["path", "label", "pred", "p0", "p1", "p2"])
        for (path, label), pred, p in zip(test_ds.samples, ps, probs):
            ww.writerow([path, int(label), int(pred)] +
                        [round(float(v), 6) for v in p])
    print("逐样本预测:", pred_path, flush=True)


def add_common(p):
    p.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT),
                   help="数据根目录（含 train/test）；默认 Experiment1/data_clean")
    p.add_argument("--model", default="convnext_tiny", choices=sorted(MODEL_FACTORY))
    p.add_argument("--pretrained", default=str(DEFAULT_PRETRAINED),
                   help="预训练权重路径；arm=transfer 时使用")
    p.add_argument("--arm", choices=["transfer", "scratch"], default="transfer")
    p.add_argument("--res", type=int, default=224)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--ls", type=float, default=0.1, help="label smoothing")
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--drop-path", type=float, default=0.1)
    p.add_argument("--freeze-epochs", type=int, default=0)
    p.add_argument("--head-lr-mult", type=float, default=1.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--stop-file", default=str(CODE / "STOP"),
                   help="存在该文件时在下一轮开始前正常退出（默认 code\\STOP）")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--dry-run", action="store_true")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="ConvNeXt 三分类：stage1 训练、3 折 CV、测试集一次性评估")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("stage1", help="单次 fit/val 训练")
    add_common(p1)
    p1.add_argument("--split-json", default="", help="固定 fit/val 划分 JSON")
    p1.add_argument("--out-dir", default="")
    p1.add_argument("--test", action="store_true", help="训练结束后评估 data/test（默认不读）")
    p1.add_argument("--save-last", action="store_true")
    p1.add_argument("--smoke", action="store_true")

    p2 = sub.add_parser("cv", help="分层 3 折交叉验证（不读 data/test）")
    add_common(p2)
    p2.add_argument("--folds", type=int, default=3)
    p2.add_argument("--folds-seed", type=int, default=0)
    p2.add_argument("--out-dir", default="")

    p3 = sub.add_parser("eval", help="在 data/test 上评估单个或多个 checkpoint")
    p3.add_argument("--checkpoint", default="", help="单个 checkpoint")
    p3.add_argument("--checkpoints", nargs="+", default=[],
                    help="多个 checkpoint，softmax 概率平均")
    p3.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT),
                    help="数据根目录（含 train/test）；默认 Experiment1/data_clean")
    p3.add_argument("--model", default="convnext_tiny", choices=sorted(MODEL_FACTORY))
    p3.add_argument("--res", type=int, default=224)
    p3.add_argument("--bs", type=int, default=16)
    p3.add_argument("--num-workers", type=int, default=0)
    p3.add_argument("--out", default="")

    args = parser.parse_args(argv)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.backends.cudnn.benchmark = True
    seed_all(args.seed if hasattr(args, "seed") else 0)

    if args.cmd == "stage1":
        cmd_stage1(args, stamp)
    elif args.cmd == "cv":
        cmd_cv(args, stamp)
    elif args.cmd == "eval":
        cmd_eval(args)


if __name__ == "__main__":
    main()
