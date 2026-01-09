# Analysis Tools Usage Guide

本文档说明如何使用已创建的分析工具进行模型对比和策略分析。

## 工具概览

已创建的5个核心分析工具：

1. **load_human_data.py** - 人类行为数据加载
2. **analyze_errors.py** - 模型错误样本对比分析
3. **compare_performance.py** - 性能对比可视化
4. **extract_attention.py** - ViT注意力提取
5. **evaluate_by_occlusion.py** - 按遮挡级别评估（已有）

## 使用流程

### 第0步：准备人类数据（可选）

如果你有OIID数据集的人类行为数据：

```bash
# 加载人类数据
python scripts/load_human_data.py \
  --dataset-root E:/Dataset/ds005226 \
  --output-dir data/human_performance
```

**输出：**
- `data/human_performance/human_all_trials.csv` - 所有试验数据
- `data/human_performance/human_performance_by_image.csv` - 按图像汇总
- `data/human_performance/human_performance_by_occlusion.csv` - 按遮挡级别汇总
- `data/human_performance/human_performance_by_image.json` - JSON格式

---

### 第1步：评估ViT性能（已完成）

```bash
# 评估ViT-B/16（已完成，结果在scripts/experiments/vit_b16/quick_test/metrics/）
python scripts/evaluate_by_occlusion.py \
  --checkpoint scripts/experiments/vit_b16/quick_test/checkpoints/best_model.pth \
  --config configs/vit_b16_quick_test.yaml
```

**输出：**
- `scripts/experiments/vit_b16/quick_test/metrics/performance_by_occlusion.csv`

---

### 第2步：评估ResNet-50性能（等ResNet训练完成）

```bash
# 评估ResNet-50（ResNet训练完成后运行）
python scripts/evaluate_by_occlusion.py \
  --checkpoint scripts/experiments/resnet50/quick_test/checkpoints/best_model.pth \
  --config configs/resnet50_quick_test.yaml
```

**输出：**
- `scripts/experiments/resnet50/quick_test/metrics/performance_by_occlusion.csv`

---

### 第3步：错误样本对比分析

当ViT和ResNet都训练完成后：

```bash
# 分析错误重叠
python scripts/analyze_errors.py \
  --config configs/vit_b16_quick_test.yaml \
  --vit-checkpoint scripts/experiments/vit_b16/quick_test/checkpoints/best_model.pth \
  --resnet-checkpoint scripts/experiments/resnet50/quick_test/checkpoints/best_model.pth \
  --output-dir scripts/experiments/analysis/error_analysis
```

**输出：**
- `error_samples.csv` - 所有样本的预测和错误标记
- `error_overlap.json` - Jaccard相似度、错误相关性等统计
- `vit_only_errors.csv` - 只有ViT错的样本
- `resnet_only_errors.csv` - 只有ResNet错的样本
- `both_error_errors.csv` - 两个模型都错的样本

**关键指标：**
- **Jaccard similarity**: 错误重叠度 (|交集| / |并集|)
- **Error correlation**: 错误模式相关性
- **Only ViT error**: 只有ViT错误的样本数
- **Only ResNet error**: 只有ResNet错误的样本数
- **Both error**: 两个模型都错的样本数

---

### 第4步：性能对比可视化

```bash
# 生成对比图表
python scripts/compare_performance.py \
  --vit-results scripts/experiments/vit_b16/quick_test/metrics/performance_by_occlusion.csv \
  --resnet-results scripts/experiments/resnet50/quick_test/metrics/performance_by_occlusion.csv \
  --human-results data/human_performance/human_performance_by_occlusion.csv \
  --output-dir scripts/experiments/analysis/comparison
```

**输出：**
- `accuracy_comparison.png` - 按遮挡级别的准确率对比曲线
- `robustness_gap.png` - 鲁棒性差距（10%-90%准确率下降）
- `model_human_gap.png` - 模型与人类性能差距
- `performance_summary.csv` - 汇总表格

---

### 第5步：提取注意力图（ViT策略分析）

#### 方式A：随机样本注意力提取

```bash
# 提取20个随机样本的注意力图
python scripts/extract_attention.py \
  --checkpoint scripts/experiments/vit_b16/quick_test/checkpoints/best_model.pth \
  --config configs/vit_b16_quick_test.yaml \
  --output-dir scripts/experiments/analysis/attention/random \
  --num-samples 20 \
  --layer -1
```

#### 方式B：对比正确vs错误样本的注意力

```bash
# 对比正确和错误预测的注意力模式
python scripts/extract_attention.py \
  --checkpoint scripts/experiments/vit_b16/quick_test/checkpoints/best_model.pth \
  --config configs/vit_b16_quick_test.yaml \
  --output-dir scripts/experiments/analysis/attention/correct_vs_error \
  --compare-errors \
  --error-csv scripts/experiments/analysis/error_analysis/error_samples.csv \
  --layer -1
```

**输出：**
- `correct/` - 正确预测样本的注意力图
- `error/` - 错误预测样本的注意力图
- 每个图像包含：原图、注意力热图、叠加图

---

## 实验决策点（根据研究计划）

### 决策点1（Week 2结束）

运行完步骤3后，检查：

```python
# 检查error_overlap.json中的指标
{
  "jaccard_similarity": 0.XX,  # 如果 < 0.5，说明错误模式不同
  "error_correlation": 0.XX,   # 如果 < 0.7，说明策略可能不同
}
```

**判断标准：**
- ✅ **继续深入分析**：如果性能差距 > 5% 或 Jaccard < 0.5
- ⚠️ **简化分析**：如果性能相近且错误高度重叠

---

### 决策点2（Week 4结束）

提取注意力图后，定性观察：

- ViT是否关注不同区域？
- 正确vs错误样本的注意力模式有何不同？
- 是否需要训练更多模型（MAE、Swin等）？

---

## 快速检查清单

在ResNet-50训练完成后，按此顺序运行：

```bash
# 1. 评估ResNet性能
python scripts/evaluate_by_occlusion.py --checkpoint scripts/experiments/resnet50/quick_test/checkpoints/best_model.pth --config configs/resnet50_quick_test.yaml

# 2. 错误分析
python scripts/analyze_errors.py

# 3. 性能对比
python scripts/compare_performance.py

# 4. 注意力提取（对比错误）
python scripts/extract_attention.py --compare-errors

# 5. 检查生成的图表和CSV，做出决策
```

---

## 注意事项

1. **路径问题**：所有脚本都支持相对于项目根目录的路径
2. **GPU内存**：注意力提取时batch_size固定为1，无需担心内存
3. **人类数据**：如果没有人类数据，可以跳过`--human-results`参数
4. **层级选择**：`--layer -1`表示最后一层，可以改成0-11提取特定层

---

## 下一步（Week 3+）

如果发现ViT和ResNet策略显著不同：

1. **Week 3**：人类-模型错误对比
2. **Week 4**：ResNet Grad-CAM提取（需要新脚本）
3. **Week 5+**：可选训练MAE/Swin等模型

所有工具都已准备好，等ResNet训练完成即可开始对比分析！
