# 实验作业一

使用 PyTorch 手写 MLP，在 scikit-learn 内置 `load_digits` 数据集上完成报告要求的实验。

## 运行

在已有的 conda 环境 `ml_lab` 中运行：

```bash
conda activate ml_lab
python mlp_digits.py
```

如需新建环境：

```bash
conda create -n ml_lab python=3.10
conda activate ml_lab
python -m pip install -r requirements.txt
python mlp_digits.py
```

使用 CPU 单线程，数据集无需下载。程序输出环境版本及每组准确率、耗时，并生成 `results/figures/` 曲线和 `results/records/` 原始记录、汇总表与三次复测均值。生成文件不纳入版本控制。

## 实验清单

所有实验使用同一份 80% / 20% 分层划分（1437 / 360 个样本），种子为42。像素值除以16，标签转换为 `torch.long`。每次重新设定随机种子，全批量训练30轮，使用交叉熵，报告最后一轮测试准确率。

| 报告组别 | 相对基线的配置 | 曲线文件 |
| --- | --- | --- |
| 0 基线 | 64→32（Sigmoid）→10，SGD，lr=0.1，无正则化 | result_baseline.png |
| 1 激活函数 | ReLU | result_exp1.png |
| 2 网络规模 | 两个隐藏层，每层128，Sigmoid | result_exp2.png |
| 3 优化器 | Adam，lr=0.01 | result_exp3.png |
| 4 学习率 | SGD，lr=0.01 | result_exp4.png |
| 5 L2 | weight_decay=1e-4 | result_exp5.png |
| 6 Dropout | p=0.2 | result_exp6.png |
| 7 BatchNorm | Linear后、Sigmoid前插入BatchNorm1d | result_exp7.png |
| 8 最优组合 | 两层128、ReLU、BatchNorm、Adam(0.01)，无L2/Dropout，固定种子复测3次 | result_exp8_run1.png 至 result_exp8_run3.png |
| 失败实验 | 基线学习率改为10.0 | result_lr_too_big.png |

共10种配置、12次训练。严格按报告样板保留这些实验，不执行额外学习率扫描、补充对照或候选组合搜索。第8组固定复现此前报告已确定的组合，本脚本不重新寻找最优配置。

样板第2组同时改变深度与宽度，第3组同时改变优化器与学习率，因此这两组不能用于严格分离单个因素的作用；这里保留样板指定配置。

## 指标说明

- 训练损失在参数更新前、`train()` 模式下计算；测试准确率在更新后、`eval()` 模式下计算。测试集不参与梯度更新。
- 耗时包含训练循环和每轮测试评估，不含初始化、绘图和保存；本脚本不增加预热训练。小数据集耗时易受系统调度及首次执行影响，不应过度比较毫秒级差异。
- 同种子三次复测用于核验可复现性，不代表不同随机种子下的统计稳定性。
- 参考原实验环境：Python 3.10.20、PyTorch 2.10.0+cu128、scikit-learn 1.7.2、NumPy 2.2.6、Matplotlib 3.10.8，实际计算使用CPU。
- 参考准确率：基线22.22%，第8组三次均98.33%。程序实际运行后打印真实指标，不硬编码结果；耗时可能变化。

图表默认使用系统可用字体；Linux 上若安装 Droid Sans Fallback，则自动启用中文字体回退。
