# Virtual Disembodiment｜虚拟去具身化

[English](README.md) · [代码版本 v0.1.0](https://github.com/M0M0KO/virtual-disembodiment/releases/tag/v0.1.0) · [方法](docs/methods.md) · [复现](docs/reproduction.md)

论文 *Decodable physiological information and predictive dependence in a brain–body model of propofol sedation* 的配套代码。

## 研究问题

身体输入怎样改变脑电预测模型的内部表征？表征中能够读出的信息，与模型预测未来脑电时实际使用的信息，是否一致？

我们使用丙泊酚镇静期间的 EEG、外周生理信号和指定给药情景，比较身体输入对内部信息组织、EEG 可读性和未来 EEG 预测的影响。读出器在校准参与者上拟合，在独立评价参与者上评分。

## 主要发现

| 问题 | 结果 |
| --- | --- |
| 身体输入是否改变表征？ | 真实身体输入提高了慢身体信号的线性可读性；原未来 EEG 预测比较未达到预设的 5% 改善门槛。 |
| 历史通路改变后，EEG 可读性能否恢复？ | 重新拟合同容量读出器后，定向干预条件下当前 EEG 的诊断 MAE 相对固定读出器平均下降 **12.59%**，95% 区间为 **10.82%–14.42%**，主要评价范围为 **16 名参与者**。 |
| 原预测器是否依赖 EEG 可读方向？ | 减弱四个由校准数据确定的 EEG 可读方向后，相对秩和位移范数匹配的随机对照，当前 EEG 读出 MAE 增加 **0.07635**；对应的 1 秒预测 CRPS 差为 **+0.00000349**，95% 区间为 **−0.00001698 至 +0.00002374**。 |

读出 MAE 衡量模型状态中近期已观测 EEG 的可访问程度；CRPS 衡量原预测器对未来 EEG 的概率预测误差。12.59% 对应读出恢复。直接状态干预的 CRPS 区间跨零，尚未检验正式等效性。

直接状态干预的[汇总源表](source_tables/eeg_subspace_effects.csv)覆盖 16 人、1,437 个匹配窗口和 5 个冻结模型种子。统计先在人内汇总种子，再计算参与者级配对效应与区间。

## 快速上手

需要 Python 3.10 或更高版本。下载源码、进入仓库根目录，并创建和激活虚拟环境后运行：

```bash
python -m pip install -e ".[dynamics]"
python -m unittest discover -s tests -v
python -m examples.check_readout
python -m examples.check_subspace
```

| 示例 | 内容 | 输出 |
| --- | --- | --- |
| `examples.check_readout` | 在生成数据上比较固定读出和条件匹配读出，校准与评价参与者分离 | 两种读出器的 MAE |
| `examples.check_subspace` | 检查子空间的秩、正交性和逐状态位移范数匹配 | 最大位移范数差 |

两个示例验证读出与干预算子的实现，无需真实记录或模型检查点。

## 复现范围

| 分析 | 仓库提供 | 完整研究另需 |
| --- | --- | --- |
| 未来 EEG 预测 | 因果特征、窗口、原生动力学和概率评分代码 | 原始信号、冻结参与者角色、给药情景、检查点及评价数组 |
| 表征几何与读出恢复 | 几何和 ridge 读出工具、生成数据示例 | 各条件表征、匹配窗口及逐人结果 |
| EEG 可读子空间干预 | 子空间拟合、状态编辑、生成数据示例和四项汇总效应 | 五个冻结模型、评价数组及逐窗干预记录 |

原研究评价未来 1、5、15 秒 EEG，CRPS 权重分别为 0.2、0.3、0.5。开发读出与模型内干预属于探索性分析；原验证结论和 5% 门槛保持不变。完整方法与输入要求见[复现指南](docs/reproduction.md)。

## 仓库结构

| 位置 | 内容 |
| --- | --- |
| [`docs/`](docs/) | 方法、复现与数据可用性说明 |
| [`src/virtual_disembodiment/`](src/virtual_disembodiment/) | 特征、窗口、模型、干预和评分 |
| [`scripts/dev_representation_metrics.py`](scripts/dev_representation_metrics.py) | 表征几何与诊断读出 |
| [`scripts/dev_eeg_subspace.py`](scripts/dev_eeg_subspace.py) | EEG 可读子空间与状态编辑 |
| [`examples/`](examples/) · [`tests/`](tests/) | 生成数据示例与软件检查 |
| [`source_tables/`](source_tables/) | 汇总结果源表 |

## 数据

原始数据来自 [DOSE-I v1](https://doi.org/10.5281/zenodo.18483292)，获取和使用遵循数据提供方的条款。本仓库发布研究代码与汇总源表；参与者级记录、训练好的模型和完整研究的输入清单未随本版提供。具体范围见[数据可用性说明](docs/data-availability.md)。

## 引用

引用研究时，请使用本页开头的论文题目。引用代码时，请注明[仓库地址](https://github.com/M0M0KO/virtual-disembodiment)及所用版本；现有研究代码版本为 `v0.1.0`。

## 许可

代码与软件文档采用 [MIT 许可证](LICENSE)。原始数据遵循提供方条款，数据许可与软件许可分别适用。
