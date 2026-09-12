"""作业一：真实运行、固定划分、全批量 PyTorch MLP。AI 辅助编写。
运行：conda run -n ml_lab python mlp_digits.py
"""
from pathlib import Path
import csv
import json
import os
import platform
import random
import sys
import time

os.environ.setdefault('MPLCONFIGDIR', '/tmp/ml_lab_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent / 'results'
RAW = ROOT / 'records'
FIG = ROOT / 'figures'
for folder in (RAW, FIG):
    folder.mkdir(parents=True, exist_ok=True)
font = Path('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')
if font.exists():
    from matplotlib import font_manager
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family'] = ['DejaVu Sans', FontProperties(fname=str(font)).get_name()]
plt.rcParams['axes.unicode_minus'] = False
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
BASE = dict(hidden=[32], act='sigmoid', dropout=0., use_bn=False,
            opt='sgd', lr=.1, wd=0., epochs=30, seed=42)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MLP(nn.Module):
    def __init__(self, hidden, act, dropout, use_bn):
        super().__init__()
        activation = {'sigmoid': nn.Sigmoid, 'relu': nn.ReLU}[act]
        layers, previous = [], 64
        for width in hidden:
            layers.append(nn.Linear(previous, width))
            if use_bn:
                layers.append(nn.BatchNorm1d(width))
            layers.append(activation())
            if dropout:
                layers.append(nn.Dropout(dropout))
            previous = width
        layers.append(nn.Linear(previous, 10))  # 交叉熵直接接收 logits。
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


X, y = load_digits(return_X_y=True)
X = torch.tensor(X.astype(np.float32) / 16.)
y = torch.tensor(y, dtype=torch.long)
train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=.2,
                                     stratify=y.numpy(), random_state=42)
assert len(set(train_idx) & set(test_idx)) == 0
assert len(train_idx) == 1437 and len(test_idx) == 360
np.savez(RAW / 'data_split.npz', train=train_idx, test=test_idx)


def run(name, changes=None, tr=train_idx, te=test_idx, plot=True, title=None, save=True):
    cfg = BASE | (changes or {})
    seed_all(cfg['seed'])  # 每组重新初始化，避免组间随机状态漂移。
    model = MLP(**{k: cfg[k] for k in ('hidden', 'act', 'dropout', 'use_bn')})
    optimizer = {'sgd': torch.optim.SGD, 'adam': torch.optim.Adam}[cfg['opt']](
        model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    loss_fn = nn.CrossEntropyLoss()
    xt, yt, xe, ye = X[tr], y[tr], X[te], y[te]
    hist = []
    start = time.perf_counter()
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(xt), yt)
        loss.backward()
        grad_norm = model.net[0].weight.grad.norm().item()
        optimizer.step()
        model.eval()  # 评估时关闭 Dropout，BN 使用运行统计量。
        with torch.no_grad():
            pred = model(xe)
            acc = (pred.argmax(1) == ye).double().mean().item() * 100
        hist.append(dict(epoch=epoch, train_loss=loss.item(), test_accuracy=acc,
                         first_layer_gradient_norm=grad_norm))
    elapsed = time.perf_counter() - start  # 含逐轮评估，不含初始化、绘图和保存。
    with torch.no_grad():
        train_logits = model(xt)
        train_acc = (train_logits.argmax(1) == yt).double().mean().item() * 100
        train_loss = loss_fn(train_logits, yt).item()
        test_loss = loss_fn(pred, ye).item()
    result = dict(name=name, config=cfg, accuracy=acc, train_accuracy=train_acc,
                  seconds=elapsed, final_train_loss_eval=train_loss,
                  final_test_loss=test_loss, parameters=sum(p.numel() for p in model.parameters()),
                  history=hist, train_samples=len(tr), evaluation_samples=len(te))
    if save:
        (RAW / f'{name}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
        with (RAW / f'{name}.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(hist[0]))
            writer.writeheader()
            writer.writerows(hist)
        print(f'{name:25s} test={acc:6.2f}% train={train_acc:6.2f}% time={elapsed:.6f}s', flush=True)
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), constrained_layout=True)
        epochs = [r['epoch'] for r in hist]
        axes[0].plot(epochs, [r['train_loss'] for r in hist], color='#236b8e')
        axes[1].plot(epochs, [r['test_accuracy'] for r in hist], color='#c05b22')
        for ax, ylabel in zip(axes, ['训练交叉熵（更新前）', '测试准确率（%，更新后）']):
            ax.set(xlabel='训练轮次（全批量）', ylabel=ylabel)
            ax.grid(alpha=.25)
        axes[1].set_ylim(0, 100)
        fig.suptitle((title or name) + f'\n最终准确率 {acc:.2f}% | 30 轮 | 随机种子 42')
        fig.savefig(FIG / f'result_{name}.png', dpi=180)
        plt.close(fig)
    return result, model


def main():
    env = dict(os=platform.platform(), python=sys.version, executable=sys.executable,
               conda_environment=os.environ.get('CONDA_DEFAULT_ENV'), torch=torch.__version__,
               numpy=np.__version__, sklearn=sklearn.__version__, matplotlib=matplotlib.__version__,
               cuda_available=torch.cuda.is_available(), device='CPU', threads=torch.get_num_threads(),
               timing='perf_counter; training + per-epoch test evaluation; excludes setup and plotting')
    assert env['conda_environment'] == 'ml_lab', '请在 conda 环境 ml_lab 中运行'
    (RAW / 'environment.json').write_text(json.dumps(env, ensure_ascii=False, indent=2))
    print(json.dumps(env, ensure_ascii=False, indent=2), flush=True)
    experiments = [
        ('baseline', {}, '图1 基线：32 / Sigmoid / SGD 0.1'),
        ('exp1', {'act': 'relu'}, '图2 实验1：ReLU'),
        ('exp2', {'hidden': [128, 128]}, '图3 实验2：两层 × 128 / Sigmoid'),
        ('exp3', {'opt': 'adam', 'lr': .01}, '图4 实验3：Adam 0.01（样板配置）'),
        ('exp4', {'lr': .01}, '图5 实验4：SGD 0.01'),
        ('exp5', {'wd': 1e-4}, '图6 实验5：L2 正则化'),
        ('exp6', {'dropout': .2}, '图7 实验6：Dropout 0.2'),
        ('exp7', {'use_bn': True}, '图8 实验7：BatchNorm'),
        ('lr_too_big', {'lr': 10.}, '图10 失败实验：SGD 10.0'),
    ]
    results = [run(n, c, title=t)[0] for n, c, t in experiments]
    # 复现报告中已确定的组合，不执行额外候选搜索。
    best_config = dict(hidden=[128, 128], act='relu', use_bn=True,
                       opt='adam', lr=.01)
    for i in range(1, 4):
        result, _ = run(f'exp8_run{i}', best_config, title=f'图9 最优组合：复测 {i}')
        results.append(result)
    repeats = results[-3:]
    assert repeats[0]['history'] == repeats[1]['history'] == repeats[2]['history']
    assert all(np.isfinite(h['train_loss']) for r in results for h in r['history'])
    (RAW / 'all_results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
    with (RAW / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'accuracy', 'train_accuracy', 'seconds', 'parameters', 'config'])
        writer.writeheader()
        writer.writerows({k: r[k] for k in writer.fieldnames} for r in results)
    repeat_summary = dict(seed=42, runs=3,
                          mean_accuracy=float(np.mean([r['accuracy'] for r in repeats])),
                          mean_seconds=float(np.mean([r['seconds'] for r in repeats])))
    (RAW / 'repeat_summary.json').write_text(
        json.dumps(repeat_summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(repeat_summary, ensure_ascii=False), flush=True)
    print('检查通过：固定分层划分、标签类型、有限损失、同种子三次复测完全一致。', flush=True)


if __name__ == '__main__':
    main()
