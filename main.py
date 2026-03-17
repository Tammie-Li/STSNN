"""
六分类实验：单日（每日前80%训练、后20%测试，结果取 Day1 与 Day2 准确率平均）；
跨日（Day1 训练，Day2 测试）。评估指标：准确率。
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from model import STSNN, EMGNet


# 默认数据根目录与结构： data/SubXX/DayY/segments.npy, segments_labels.npy
DATA_ROOT = "data"
SAMPLE_LENGTH = 250
IN_CHANNELS = 8
NUM_CLASSES = 6
C_H = 16


def build_model(args):
    """根据 args.model 构建模型；STSNN 与 EMGNet 使用相同拓扑与超参以便对比。"""
    common = dict(
        channel=IN_CHANNELS,
        time_length=SAMPLE_LENGTH,
        num_classes=NUM_CLASSES,
        drop_out=getattr(args, "dropout", 0.4),
        time_point=9,
        N_t=8,
        N_s=16,
    )
    if getattr(args, "model", "emgnet").lower() == "stsnn":
        return STSNN(**common)
    return EMGNet(**common)


def load_segments(data_root, sub_id, day, return_raw=False):
    """
    加载某被试某天的片段与标签。
    返回 (X, y_mapped) 或 (X, y_mapped, y_raw)。y_mapped 为 0..num_classes-1。
    """
    base = os.path.join(data_root, f"Sub{sub_id:02d}", f"Day{day}")
    seg_path = os.path.join(base, "segments.npy")
    label_path = os.path.join(base, "segments_labels.npy")
    if not os.path.isfile(seg_path) or not os.path.isfile(label_path):
        return (None, None, None) if return_raw else (None, None)
    X = np.load(seg_path).astype(np.float32)   # (N, C, T)
    y_raw = np.load(label_path)
    uniq = np.unique(y_raw)
    if len(uniq) > NUM_CLASSES:
        raise ValueError(f"Subject {sub_id} Day{day} 标签数 {len(uniq)} 超过 {NUM_CLASSES}")
    y_mapped = np.zeros_like(y_raw, dtype=np.int64)
    for i, u in enumerate(uniq):
        y_mapped[y_raw == u] = i
    if return_raw:
        return X, y_mapped, y_raw
    return X, y_mapped


def split_80_20(X, y, shuffle=True, seed=None):
    """先打乱（可选），再按 80% 训练、20% 测试划分。"""
    n = len(y)
    if shuffle and seed is not None:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        X, y = X[perm], y[perm]
    idx = int(np.ceil(n * 0.8))
    return (X[:idx], y[:idx]), (X[idx:], y[idx:])


def train_epoch(model, loader, criterion, optimizer, device, desc=None):
    model.train()
    total_loss, total_n = 0.0, 0
    reset_fn = getattr(model, "reset_net", None)
    pbar = tqdm(loader, desc=desc or "Train", leave=False, unit="batch")
    for xb, yb in pbar:
        xb, yb = xb.to(device), yb.to(device)
        if reset_fn is not None:
            reset_fn()
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        total_n += xb.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / total_n if total_n else 0.0


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    reset_fn = getattr(model, "reset_net", None)
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if reset_fn is not None:
                reset_fn()
            logits = model(xb)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += xb.size(0)
    return correct / total if total else 0.0


def run_single_day_experiment(args, device, subjects):
    """
    单日实验：每个被试的 Day1、Day2 分别先打乱数据，再按 80% 训练、20% 测试划分，
    该被试结果为 Day1 与 Day2 测试准确率的平均值。
    """
    results = {}  # sub_id -> (acc_day1, acc_day2, avg) 或 None
    for sub_id in tqdm(subjects, desc="单日实验-被试", unit="人"):
        accs = []
        for day in (1, 2):
            X, y = load_segments(args.data_root, sub_id, day)
            if X is None or len(y) < 10:
                tqdm.write(f"  跳过 Sub{sub_id:02d} Day{day}（无数据或样本过少）")
                accs.append(np.nan)
                continue
            # 划分前先打乱，种子由 args.seed + 被试/天 确定，保证可复现
            split_seed = (args.seed + sub_id * 10 + day) % (2 ** 32)
            (X_tr, y_tr), (X_te, y_te) = split_80_20(X, y, shuffle=True, seed=split_seed)
            train_ds = TensorDataset(
                torch.from_numpy(X_tr), torch.from_numpy(y_tr)
            )
            test_ds = TensorDataset(
                torch.from_numpy(X_te), torch.from_numpy(y_te)
            )
            train_loader = DataLoader(
                train_ds, batch_size=args.batch_size, shuffle=True,
                num_workers=0, pin_memory=False
            )
            test_loader = DataLoader(
                test_ds, batch_size=args.batch_size, shuffle=False,
                num_workers=0
            )
            model = build_model(args).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            for epoch in tqdm(range(args.epochs), desc=f"Sub{sub_id:02d} Day{day}", leave=False, unit="epoch"):
                train_epoch(model, train_loader, criterion, optimizer, device, desc=f"Sub{sub_id:02d} D{day}")
            acc = evaluate(model, test_loader, device)
            accs.append(acc)
        if len(accs) == 2:
            if np.isnan(accs[0]) and np.isnan(accs[1]):
                results[sub_id] = None
            else:
                valid = [a for a in accs if not np.isnan(a)]
                avg = np.mean(valid)
                results[sub_id] = (accs[0], accs[1], avg)
                tqdm.write(f"  [单日] Sub{sub_id:02d}: Day1={accs[0]:.4f}, Day2={accs[1]:.4f}, 平均={avg:.4f}")
        else:
            results[sub_id] = None
    return results


def run_cross_day_experiment(args, device, subjects):
    """
    跨日实验：每个被试用 Day1 全部数据训练，Day2 全部数据测试。
    标签以 Day1 的类别为基准，Day2 的原始标签按 Day1 的类别顺序映射。
    """
    results = {}  # sub_id -> acc
    for sub_id in tqdm(subjects, desc="跨日实验-被试", unit="人"):
        X1, y1, y1_raw = load_segments(args.data_root, sub_id, 1, return_raw=True)
        X2, y2_mapped, y2_raw = load_segments(args.data_root, sub_id, 2, return_raw=True)
        if X1 is None or X2 is None or len(y1) < 5 or len(y2_mapped) < 5:
            tqdm.write(f"  跳过 Sub{sub_id:02d} 跨日（Day1/Day2 缺数据或样本过少）")
            results[sub_id] = np.nan
            continue
        uniq1 = np.unique(y1_raw)
        # 将 Day2 标签映射到 Day1 的类别下标
        idx2 = np.searchsorted(uniq1, y2_raw)
        idx2 = np.clip(idx2, 0, len(uniq1) - 1)
        y2_aligned = np.where(uniq1[idx2] == y2_raw, idx2, 0).astype(np.int64)
        train_ds = TensorDataset(
            torch.from_numpy(X1), torch.from_numpy(y1)
        )
        test_ds = TensorDataset(
            torch.from_numpy(X2), torch.from_numpy(y2_aligned)
        )
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=0, pin_memory=False
        )
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=0
        )
        model = build_model(args).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in tqdm(range(args.epochs), desc=f"Sub{sub_id:02d} 跨日", leave=False, unit="epoch"):
            train_epoch(model, train_loader, criterion, optimizer, device, desc="跨日 Train")
        acc = evaluate(model, test_loader, device)
        results[sub_id] = acc
        tqdm.write(f"  [跨日] Sub{sub_id:02d}: 准确率={acc:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="STSNN 六分类：单日 / 跨日实验")
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    parser.add_argument("--subjects", type=str, default="1,2,3,4,5,6,7,8,9,10",
                        help="被试 ID 列表，逗号分隔，如 1,2,3")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="stsnn", choices=["emgnet", "stsnn"],
                        help="模型: emgnet (默认, 参考 sEMGNet 提升准确率) 或 stsnn")
    parser.add_argument("--dropout", type=float, default=0.4, help="EMGNet/STSNN 的 Dropout 比例")
    parser.add_argument("--single_day", action="store_true", help="仅运行单日实验")
    parser.add_argument("--cross_day", action="store_true", help="仅运行跨日实验")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subjects = [int(x.strip()) for x in args.subjects.split(",")]

    run_both = not args.single_day and not args.cross_day
    single_results = None
    cross_results = None

    # ----- 单日实验 -----
    if run_both or args.single_day:
        print("========== 单日实验（每日前 80% 训练、后 20% 测试，结果为 Day1 与 Day2 准确率平均）==========")
        single_results = run_single_day_experiment(args, device, subjects)
        valid_single = [
            (sid, single_results[sid][0], single_results[sid][1], single_results[sid][2])
            for sid in subjects
            if single_results.get(sid) is not None
        ]
        if valid_single:
            mean_avg = np.mean([x[3] for x in valid_single])
            print(f"\n  单日实验 平均准确率（被试内 Day1/Day2 平均再对被试求均）: {mean_avg:.4f}")
        else:
            print("无有效被试结果。请先生成 Day1、Day2 的 segments.npy（运行 get_data.py 并包含 Day2）。")

    # ----- 跨日实验 -----
    if run_both or args.cross_day:
        print("\n========== 跨日实验（Day1 训练，Day2 测试）==========")
        cross_results = run_cross_day_experiment(args, device, subjects)
        valid_cross = [
            (sid, cross_results[sid]) for sid in subjects
            if not np.isnan(cross_results.get(sid, np.nan))
        ]
        if valid_cross:
            mean_cross = np.mean([x[1] for x in valid_cross])
            print(f"\n  跨日实验 平均准确率: {mean_cross:.4f}")
        else:
            print("  无有效被试结果。请确保存在 Day1 与 Day2 的 segments 数据。")

    # ----- 实验结果汇总（输出每个实验的完整结果） -----
    print("\n" + "=" * 60)
    print("实验结果汇总")
    print("=" * 60)
    if single_results is not None:
        print("\n【单日实验】每日前 80% 训练、后 20% 测试，被试结果 = Day1 与 Day2 准确率平均")
        print("-" * 60)
        for sid in subjects:
            r = single_results.get(sid)
            if r is not None:
                a1, a2, avg = r[0], r[1], r[2]
                print(f"  Sub{sid:02d}:  Day1 准确率 = {a1:.4f},  Day2 准确率 = {a2:.4f},  平均 = {avg:.4f}")
            else:
                print(f"  Sub{sid:02d}:  无有效结果（跳过）")
        valid_single = [single_results[sid] for sid in subjects if single_results.get(sid) is not None]
        if valid_single:
            mean_s = np.mean([x[2] for x in valid_single])
            print("-" * 60)
            print(f"  单日实验 平均准确率: {mean_s:.4f}")
    if cross_results is not None:
        print("\n【跨日实验】Day1 训练、Day2 测试")
        print("-" * 60)
        for sid in subjects:
            acc = cross_results.get(sid, np.nan)
            if not np.isnan(acc):
                print(f"  Sub{sid:02d}:  准确率 = {acc:.4f}")
            else:
                print(f"  Sub{sid:02d}:  无有效结果（跳过）")
        valid_cross = [cross_results[sid] for sid in subjects if not np.isnan(cross_results.get(sid, np.nan))]
        if valid_cross:
            mean_c = np.mean(valid_cross)
            print("-" * 60)
            print(f"  跨日实验 平均准确率: {mean_c:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
