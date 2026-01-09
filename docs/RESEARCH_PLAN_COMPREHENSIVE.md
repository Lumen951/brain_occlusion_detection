# 多模型对比研究计划：人类vs AI遮挡识别策略的深度分析

## 项目概述

**研究题目**：Beyond Vision Transformers: A Comprehensive Comparison of Human and Multiple AI Architectures on Occluded Object Recognition

**研究期限**：12周（3个月）

**核心目标**：
1. 系统比较人类与多种AI架构（Transformer、CNN、混合模型）在遮挡识别上的表现
2. 揭示不同架构的处理策略差异
3. 分析架构设计对遮挡鲁棒性的影响
4. 探索如何设计更接近人类认知的AI系统

---

## 1. 研究背景与动机

### 1.1 问题定义

**核心问题**：
```
不同架构的深度学习模型（ViT、CNN、混合等）
在处理遮挡物体时是否使用了不同的策略？
它们与人类的认知策略有何异同？
能否通过对比分析，指导下一代架构设计？
```

**研究意义**：
- **理论层面**：理解不同架构的内在机制
- **应用层面**：指导实际场景的模型选择
- **认知层面**：通过AI对比，理解人类视觉
- **设计层面**：为更robust的架构提供设计原则

### 1.2 为什么需要多模型对比？

**现有研究的局限**：
- ❌ 只对比单一模型（如ViT）vs 人类
- ❌ 无法区分是"深度学习的共性"还是"特定架构的特性"
- ❌ 难以指导架构设计

**多模型对比的优势**：
- ✅ 揭示架构vs 性能的关系
- ✅ 发现哪些设计更接近人类
- ✅ 为架构选择提供依据
- ✅ 故事更完整，说服力更强

---

## 2. 模型选择与分类

### 2.1 对比模型清单

#### 类别A：纯Transformer架构

| 模型 | 预训练 | 参数量 | 特点 | 预期表现 |
|------|--------|--------|------|---------|
| **ViT-B/16** | ImageNet-21K | 86M | 标准ViT，全局注意力 | 高遮挡表现好 |
| **ViT-L/16** | ImageNet-21K | 304M | 大型ViT，更强表征 | 90%遮挡最优 |
| **Swin Transformer** | ImageNet-1K | 88M | 层级式窗口注意力 | 平衡性能 |
| **DeiT-III** | ImageNet-1K | 86M | Data-efficient训练 | 数据效率高 |
| **MaxViT** | ImageNet-1K | 64M | 多尺度注意力 | 细节捕捉好 |

**关键对比点**：
- ViT-B vs ViT-L：模型规模的影响
- ViT vs Swin：全局 vs 局部注意力的差异
- ViT vs DeiT：预训练策略的影响

#### 类别B：纯CNN架构

| 模型 | 预训练 | 参数量 | 特点 | 预期表现 |
|------|--------|--------|------|---------|
| **ResNet-50** | ImageNet-1K | 26M | 经典CNN，残差连接 | 基线模型 |
| **ResNet-101** | ImageNet-1K | 45M | 深层ResNet | 更强表征 |
| **EfficientNet-V2** | ImageNet-21K | 119M | 复合缩放 | 效率最优 |
| **ConvNeXt** | ImageNet-1K | 89M | 现代CNN设计 | 接近ViT性能 |
| **RegNetY-16GF** | ImageNet-1K | 111M | 网络正则化 | 设计鲁棒 |

**关键对比点**：
- ResNet-50 vs ResNet-101：深度的影响
- ResNet vs EfficientNet：设计效率的影响
- 经典CNN vs 现代CNN（ConvNeXt）

#### 类别C：混合架构

| 模型 | 预训练 | 参数量 | 特点 | 预期表现 |
|------|--------|--------|------|---------|
| **CoAtNet-1** | ImageNet-1K | 71M | Conv + Attention | 结合两者优势 |
| **CMT-Tiny** | ImageNet-1K | 18M | 轻量级混合 | 效率与性能平衡 |
| **MT-HybridNet** | ImageNet-1K | 55M | Multi-task混合 | 任务泛化强 |
| **PVT-v2-B2** | ImageNet-1K | 43M | Pyramid Vision | 多尺度特征 |

**关键对比点**：
- 混合架构是否真的结合优势？
- 与纯Transformer/CNN的差异

#### 类别D：专门设计的robust模型

| 模型 | 预训练 | 参数量 | 特点 | 预期表现 |
|------|--------|--------|------|---------|
| **MAE-ViT** | ImageNet-1K | 86M | Masked自编码 | 遮挡鲁棒性强 |
| **BEiT-Base** | ImageNet-21K | 87M | Bidirectional encoder | 自监督学习 |
| **iGPT-XL** | ImageNet-1K | 693M | Generative预训练 | 生成式理解 |

**关键对比点**：
- 自监督预训练是否提升遮挡鲁棒性？
- MAE的masked modeling是否特别适合遮挡？

#### 类别E：大规模预训练模型

| 模型 | 预训练 | 参数量 | 特点 | 预期表现 |
|------|--------|--------|------|---------|
| **CLIP-ViT** | LAION-400M | 151M | 视觉-语言对齐 | 多模态理解 |
| **DINOv2-ViT** | LVD-142M | 304M | 自监督蒸馏 | 强表征学习 |

**关键对比点**：
- 大规模预训练是否带来更强的泛化？

### 2.2 最终选择的模型集（平衡工作量）

**主要对比集（10个模型）**：
```python
必选模型（7个）：
1. ViT-B/16        # Transformer代表
2. Swin-B          # 层级Transformer
3. ResNet-50       # CNN基线
4. ResNet-101      # 深层CNN
5. EfficientNet-V2 # 高效CNN
6. ConvNeXt-B      # 现代CNN
7. MAE-ViT         # 自监督预训练

可选模型（3-5个，根据时间）：
8. CoAtNet-1       # 混合架构
9. DeiT-III        # Data-efficient
10. CLIP-ViT       # 多模态预训练
11. DINOv2         # 强表征
12. PVT-v2         # 轻量级
```

**选择理由**：
- 覆盖主要架构类型
- 预训练权重可获取
- 参数量范围合理（26M-304M）
- 可以对比多个维度

### 2.3 渐进式模型扩展策略（⭐推荐方案）

**核心理念**：
```
不是一次性训练所有10个模型
而是从2个核心模型开始，逐步扩展
每一步都有独立成果，降低初期风险
```

#### 为什么采用渐进式策略？

**优势1: 降低初期风险**
```
一次性训练10个模型：
├─ 工作量：2-3周
├─ 风险：高（如果发现问题，时间已浪费）
└─ 灵活性：低

渐进式（2个模型开始）：
├─ 工作量：2-3天（第一个里程碑）
├─ 风险：低（快速验证可行性）
└─ 灵活性：高（每步都可以调整方向）
```

**优势2: 快速获得初步结果**
```
Week 1: 训练ViT和ResNet-50
Week 2: 初步对比和错误分析
Week 3: 决策：是否值得继续添加更多模型？

如果有价值 → 继续扩展
如果没价值 → 及时止损，调整方向
```

**优势3: 每个阶段都可独立成文**
```
阶段1（2个模型）：ViT vs ResNet-50 vs 人类
  → 投稿：Workshop或会议短文

阶段2（5个模型）：添加Swin, MAE, ConvNeXt
  → 投稿：AAAI/IJCAI

阶段3（10个模型）：完整对比
  → 投稿：CVPR/ICCV
```

#### 渐进式扩展的三阶段计划

**阶段1: 核心对比（Week 1-4）**

```python
目标：建立核心baseline，验证可行性

模型集（2个）：
1. ViT-B/16
   - 代表：Transformer家族
   - 预期：高遮挡表现好
   - 训练时间：~2-3天

2. ResNet-50
   - 代表：CNN家族
   - 预期：低遮挡表现好，高遮挡下降
   - 训练时间：~1-2天

对比维度：
├─ 性能曲线（3个遮挡等级）
├─ 错误样本分析（找出差异）
├─ Attention vs Grad-CAM
└─ 初步的fMRI对比（如果时间允许）

预期发现：
┌─────────────────────────────────────┐
│ 10%遮挡: ViT=98%, ResNet=96%       │
│ 70%遮挡: ViT=88%, ResNet=78%       │
│ 90%遮挡: ViT=85%, ResNet=65%       │
│                                     │
│ 关键差异：Transformer在高遮挡上    │
│           明显优于CNN               │
└─────────────────────────────────────┘

决策点（Week 4结束）：
✅ 如果差异明显（>5%）
   → 继续阶段2，添加更多模型

⚠️ 如果差异很小（<3%）
   → 调整策略：
     - 换其他模型（如MAE, Swin）
     - 或转向纯分析路线（不做多模型）
```

**阶段2: 架构家族扩展（Week 5-8）**

```python
前提：阶段1发现有价值的差异

添加模型（3个）：
3. Swin-B
   - 目的：层级式Transformer vs 全局ViT
   - 对比：局部窗口 vs 全局attention

4. MAE-ViT
   - 目的：自监督预训练的影响
   - 对比：masked modeling的robustness

5. ConvNeXt-B
   - 目的：现代CNN vs 经典ResNet
   - 对比：设计理念的影响

新的对比维度：
├─ Transformer内部：ViT vs Swin
│  └─ 全局 vs 局部attention
│
├─ CNN内部：ResNet vs ConvNeXt
│  └─ 经典 vs 现代
│
├─ 预训练：ViT vs MAE-ViT
│  └─ 监督 vs 自监督
│
└─ 架构家族：Transformer vs CNN
   └─ 更全面的对比

预期发现：
┌──────────────────────────────────────────┐
│ 架构对比（90%遮挡准确率）：              │
│ ViT:    85%  (全局attention基准)         │
│ Swin:   82%  (局部attention略差)         │
│ MAE:    87%  (自监督最优!)               │
│ ConvNX: 72%  (现代CNN优于ResNet)         │
│ ResNet: 65%  (经典CNN基线)               │
└──────────────────────────────────────────┘

新的分析：
1. Attention熵：Swin是否更聚焦？
2. 表征稳定性：MAE是否最稳定？
3. 错误模式：ConvNeXt是否比ResNet更像人类？

决策点（Week 8结束）：
✅ 如果发现足够丰富（3-5个）
   → 可以撰写论文投AAAI/IJCAI
   → 或继续阶段3冲顶会

⚠️ 如果发现一般（1-2个）
   → 加强分析，或尝试阶段3
```

**阶段3: 全面扩展（Week 9-12，可选）**

```python
前提：阶段2成果不错，有时间继续

添加模型（5个）：
6. ResNet-101      - 深度影响
7. EfficientNet-V2 - 效率设计
8. CoAtNet-1       - 混合架构
9. DeiT-III        - Data-efficient
10. DINOv2         - 大规模预训练（如果时间）

或者：
├─ 聚焦特定方向（如只加MAE变体）
└─ 或做模型创新（基于发现设计新模块）

最终成果：
- 10个模型的完整对比
- 深入的多维度分析
- 可能的模型创新

投稿目标：CVPR/ICCV
```

#### 时间分配对比

```
一次性训练所有模型：
┌────────────────────────────────────┐
│ Week 1-2:  训练10个模型            │
│ Week 3-8:  深入分析                │
│ Week 9-12: 论文撰写                │
│                                     │
│ 风险：如果2周后发现问题，         │
│       8周分析已经白费              │
└────────────────────────────────────┘

渐进式扩展：
┌────────────────────────────────────┐
│ Week 1-2:  训练2个模型 ✓           │
│ Week 3-4:  初步分析，决策点        │
│   ↓ 有价值，继续                  │
│ Week 5-6:  训练3个模型 ✓           │
│ Week 7-8:  扩展分析，决策点        │
│   ↓ 成果不错，继续或写成论文      │
│ Week 9-10: 训练5个模型或模型创新   │
│ Week 11-12: 论文撰写               │
│                                     │
│ 风险：可控，每个决策点都可调整    │
└────────────────────────────────────┘
```

#### 每个决策点的评估标准

**决策点1（Week 4）：是否继续？**

```python
评估指标：

1. 性能差异
   ✅ >5%  → 强烈推荐继续
   ⚠️ 3-5% → 可以继续
   ❌ <3%  → 考虑换方向

2. 错误模式
   ✅ 清晰的差异模式（如CNN错A类，ViT错B类）
   ⚠️ 部分差异
   ❌ 没有明显模式

3. 故事完整性
   ✅ 可以讲述完整故事
   ⚠️ 故事不完整
   ❌ 无法形成故事

决策矩阵：
┌──────────┬─────────┬─────────┬─────────┐
│ 性能差异 │ 强(>5%) │ 中(3-5%)│ 弱(<3%) │
├──────────┼─────────┼─────────┼─────────┤
│ 错误模式 │ 清晰    │ 清晰    │ 清晰    │
│ 决策     │ 继续✅  │ 继续✅  │ 考虑⚠️  │
├──────────┼─────────┼─────────┼─────────┤
│ 错误模式 │ 部分    │ 部分    │ 模糊    │
│ 决策     │ 继续⚠️  │ 考虑⚠️  │ 换方向❌ │
├──────────┼─────────┼─────────┼─────────┤
│ 错误模式 │ 模糊    │ 模糊    │ 模糊    │
│ 决策     │ 考虑⚠️  │ 换方向❌ │ 换方向❌ │
└──────────┴─────────┴─────────┴─────────┘
```

**决策点2（Week 8）：撰写还是扩展？**

```python
评估标准：

1. 当前成果质量
   ✅ 优秀（5+发现，多个深入分析）
      → 可以投AAAI/IJCAI
      → 或继续冲CVPR/ICCV
   ⚠️ 良好（3-4发现，分析充分）
      → 建议写论文投AAAI/IJCAI
      → 也可扩展后投CVPR/ICCV（风险较高）
   ❌ 一般（1-2发现）
      → 必须继续扩展或加强分析

2. 剩余时间
   ✅ 还有4周+：可以扩展
   ⚠️ 只有2-3周：建议写论文
   ❌ <2周：只能写论文

3. 投稿目标deadline
   ✅ 有充足时间：冲顶会
   ⚠️ 时间紧张：稳妥会议
```

#### 灵活调整策略

**策略A: 保守路线（推荐）**
```
Week 1-2:  ViT + ResNet-50
Week 3-4:  深入分析这2个模型
Week 5-6:  决策：如果好，加3个模型
Week 7-8:  分析5个模型
Week 9-10: 撰写论文
Week 11-12: 修改投稿

目标：AAAI/IJCAI（稳妥）
概率：70%
```

**策略B: 积极路线**
```
Week 1-2:  ViT + ResNet-50
Week 3-4:  分析 + 加3个模型
Week 5-6:  训练3个模型
Week 7-8:  分析5个模型
Week 9-10: 加5个模型或模型创新
Week 11-12: 撰写论文

目标：CVPR/ICCV
概率：30%（需要效率和运气）
```

**策略C: 快速保底**
```
Week 1-2:  ViT + ResNet-50
Week 3-4:  深入分析
Week 5-8:  超强分析（attention, 表征, fMRI）
Week 9-10: 撰写详细的分析型论文
Week 11-12: 修改投稿

目标：WACV/ICPR或workshop
概率：90%（保底）
```

#### 模型扩展的优先级

**如果时间有限，优先加哪些模型？**

```
优先级1（最高）：MAE-ViT
├─ 理由：自监督预训练，可能最robust
├─ 预期：性能提升3-5%
└─ 训练时间：2-3天

优先级2（高）：Swin-B
├─ 理由：对比全局vs局部attention
├─ 预期：性能略低于ViT但更高效
└─ 训练时间：2-3天

优先级3（中）：ConvNeXt-B
├─ 理由：现代CNN设计
├─ 预期：性能优于ResNet
└─ 训练时间：2天

优先级4（低）：ResNet-101
├─ 理由：深度影响（但不是核心）
├─ 预期：略优于ResNet-50
└─ 训练时间：2天

优先级5（最低）：其他模型
├─ CoAtNet, DeiT, DINOv2等
├─ 只在时间充裕时考虑
└─ 训练时间：2-4天
```

#### 实际执行建议

**Week 1-2: 立即开始**
```python
Day 1-2: 环境搭建
Day 3-5: 训练ViT-B/16
Day 6-7: 训练ResNet-50
Day 8-10: 加载人类数据，初步对比
Day 11-12: 错误样本识别
Day 13-14: 决策：差异是否明显？
```

**Week 3-4: 深入分析2个模型**
```python
Week 3:
├─ 错误样本聚类
├─ Attention vs Grad-CAM对比
└─ 初步fMRI分析（如果可以做）

Week 4:
├─ 总结发现
├─ 决策：是否继续加模型？
└─ 如果继续，规划加哪3个
```

**Week 5+: 根据Week 4决策**
```python
如果继续：
  ├─ Week 5-6: 训练3个模型（MAE, Swin, ConvNeXt）
  ├─ Week 7-8: 深入分析5个模型
  └─ Week 9+:  根据成果决定

如果停止：
  ├─ Week 5-8: 超强分析2个模型
  ├─ Week 9-10: 撰写论文
  └─ Week 11-12: 修改投稿
```

#### 预期成果（渐进式）

**最坏情况（Week 4停止）**
```
模型：2个（ViT, ResNet-50）
发现：2-3个
投稿：WACV/ICPR或workshop
时间：8周
价值：保底，有论文产出
```

**中等情况（Week 8停止）**
```
模型：5个
发现：4-5个
投稿：AAAI/IJCAI
时间：12周
价值：良好，平衡投入产出
```

**最好情况（Week 12完整）**
```
模型：10个
发现：6-8个
投稿：CVPR/ICCV
时间：16周（需要延期）
价值：优秀，冲击顶会
```

---

## 3. 研究问题与假设

### 3.1 核心研究问题

**RQ1: Performance Gap**
```
问题：不同架构在遮挡识别上的性能差异有多大？
假设：
  - Transformer在极高遮挡（90%）上优于CNN
  - 混合架构在中等遮挡（70%）上表现最优
  - MAE等自监督模型在所有遮挡等级上更robust
```

**RQ2: Error Pattern Divergence**
```
问题：不同架构的错误模式是否相同？
假设：
  - CNN和Transformer在不同样本上犯错
  - CNN错误：过度依赖局部纹理
  - Transformer错误：过度关注背景噪声
  - 混合架构：错误最少
```

**RQ3: Human-AI Similarity**
```
问题：哪些架构更像人类？
假设：
  - ResNet最不接近人类（过度依赖纹理）
  - ViT在注意力模式上接近人类
  - MAE-ViT在行为上最接近人类
```

**RQ4: Architecture-Feature Relationship**
```
问题：架构设计如何影响特征表征？
假设：
  - CNN：局部特征主导，缺乏全局上下文
  - Transformer：全局attention，但过度分散
  - 混合：平衡局部和全局
```

**RQ5: Scalability Effect**
```
问题：模型规模是否影响遮挡鲁棒性？
假设：
  - ViT-L > ViT-B（大模型更robust）
  - ResNet-101 > ResNet-50
  - 但边际收益递减
```

### 3.2 预期发现（假设驱动）

```
预期结果矩阵：

架构类型      │ 10%遮挡 │ 70%遮挡 │ 90%遮挡 │ 人类相似度
─────────────────────────────────────────────
ResNet-50     │   96%   │   78%   │   65%   │    低
ResNet-101    │   97%   │   81%   │   68%   │    低
EfficientNet  │   97%   │   83%   │   70%   │    中
ConvNeXt      │   97%   │   84%   │   72%   │    中
─────────────────────────────────────────────
ViT-B         │   98%   │   88%   │   85%   │    高
Swin-B        │   98%   │   86%   │   82%   │    高
MAE-ViT       │   98%   │   90%   │   87%   │   最高
─────────────────────────────────────────────
CoAtNet       │   98%   │   89%   │   84%   │    高
─────────────────────────────────────────────
人类          │   95%   │   80%   │   68%   │    -
```

---

## 4. 研究方法与实验设计

### 4.1 数据准备

#### 4.1.1 主要数据集：OIID

```
训练集：
- 50个受试者 × 2 runs ≈ 5000-7500样本
- 包含：图像、标签、遮挡等级、fMRI、行为数据

验证集：
- 8个受试者 ≈ 800-1200样本

测试集：
- 7个受试者 ≈ 700-1050样本

划分策略：
- 按受试者划分（避免信息泄露）
- 保证每个遮挡等级的样本均衡
```

#### 4.1.2 扩展数据集（可选，验证泛化性）

```
人造遮挡数据集：
1. COCO-Occlusion：在COCO验证集上添加人造遮挡
2. PascalVOC-Occlusion：同上
3. ImageNet-Occlusion：随机采样的ImageNet类 + 遮挡

遮挡类型：
- 方块遮挡（10%, 70%, 90%）
- 随机噪声遮挡
- 自然遮挡（用其他物体遮挡）
```

### 4.2 模型训练与评估

#### 4.2.1 训练策略

```python
# 统一的训练配置
training_config = {
    'epochs': 100,
    'batch_size': 32,
    'optimizer': 'AdamW',
    'lr': 3e-4,
    'weight_decay': 0.05,
    'scheduler': 'cosine',
    'warmup_epochs': 5,
    'augmentation': [
        'RandomHorizontalFlip',
        'RandomRotation',
        'ColorJitter',
    ],
}

# 特殊配置（根据架构调整）
architecture_specific = {
    'ViT': {
        'image_size': 224,
        'patch_size': 16,
    },
    'Swin': {
        'image_size': 224,
        'window_size': 7,
    },
    'ResNet': {
        'image_size': 224,
    },
}
```

#### 4.2.2 评估指标

```python
# 主要指标
metrics = {
    # 性能指标
    'accuracy': '准确率',
    'precision': '精确率',
    'recall': '召回率',
    'f1': 'F1分数',

    # 遮挡鲁棒性
    'occlusion_robustness': '遮挡鲁棒性指标',
    'robustness_gap': '90%-10%准确率差',

    # 人类相似度
    'error_overlap': '错误样本重叠度',
    'attention_correlation': '注意力相关性',
    'decision_correlation': '决策相关性',

    # 模型特性
    'confidence_calibration': '置信度校准',
    'uncertainty': '不确定性估计',
}
```

### 4.3 对比实验设计

#### 实验1：性能对比（Performance Comparison）

```
目的：对比不同架构在不同遮挡等级的表现

设计：
- 自变量：架构类型（10个模型）× 遮挡等级（10%, 70%, 90%）
- 因变量：准确率、F1、置信度

分析方法：
- 重复测量ANOVA
- 多重比较校正
- 效应量计算（Cohen's d）

可视化：
- 性能曲线图（x=遮挡等级，y=准确率，line=模型）
- 热力图（模型×遮挡等级）
- 箱线图（模型间差异）
```

#### 实验2：错误模式分析（Error Pattern Analysis）

```
目的：揭示不同架构的错误样本分布

设计：
- 识别三类错误样本：
  A. 人类对，模型错（模型弱点）
  B. 模型对，人类错（模型优势）
  C. 两者都对（正确机制）

分析方法：
- 聚类分析：错误样本的特征聚类
- 可视化：t-SNE/UMAP降维
- 统计：错误类型分布的差异

关键问题：
- CNN和Transformer的错误样本重叠度？
- MAE的错误样本是否与其他模型不同？
```

#### 实验3：注意力机制对比（Attention Analysis）

```
目的：比较不同架构的注意力/关注区域

方法：
- Transformer: 提取attention maps
- CNN: 使用Grad-CAM/Integrated Gradients
- 混合: 两者结合

对比维度：
1. 空间分布：聚焦 vs 分散
2. 遮挡响应：如何处理遮挡区域
3. 人类相似：与fMRI激活图的相关性

定量指标：
- Attention entropy（熵，衡量分散度）
- Spatial overlap（与人类fMRI的重叠度）
- Focus ratio（聚焦程度）
```

#### 实验4：fMRI对比分析（fMRI Analysis）

```
目的：理解模型错误与人类神经活动的关系

设计：
- 对比不同错误试次的fMRI激活
- 错误类型 × 模型类型 × 脑区

分析：
1. 模型错但人类对的试次：
   - 这些试次的人类脑激活模式？
   - 哪些脑区激活强？

2. 模型对但人类错的试次：
   - 模型学到了什么人类不具备的能力？
   - 是否只是overfit？

3. 所有模型都错的试次：
   - 这些是最困难的样本
   - 人类的神经机制如何？
```

#### 实验5：架构消融实验（Ablation Study）

```
目的：理解哪些设计元素影响遮挡鲁棒性

对比维度：
1. 注意力范围
   - 局部 vs 全局 vs 层级
   - Swin vs ViT vs ConvNeXt

2. 模型规模
   - ViT-B vs ViT-L
   - ResNet-50 vs ResNet-101

3. 预训练方式
   - 监督 vs 自监督（MAE）
   - ImageNet-1K vs ImageNet-21K

4. 混合策略
   - 纯CNN vs 纯Transformer vs 混合

预期发现：
- 全局attention在高遮挡上更优
- 自监督预训练提升鲁棒性
- 混合架构平衡性能和效率
```

---

## 5. 实施计划与时间表

### 5.1 总体时间线（12周）

```
Week 1-2:  阶段0 - 准备与Baseline
Week 3-8:  阶段1 - 深度对比分析
Week 9-12: 阶段2 - 模型创新（可选）
```

### 5.2 详细周计划

#### Week 1-2: 阶段0 - 准备与Baseline

**Week 1: 模型准备与训练**

```
Day 1-2: 环境搭建
├─ 安装依赖：timm, torch, nilearn等
├─ 下载预训练权重
└─ 准备数据加载器

Day 3-4: 训练第一批模型（5个）
├─ ViT-B/16
├─ ResNet-50
├─ ResNet-101
├─ EfficientNet-V2
└─ ConvNeXt-B

Day 5-6: 训练第二批模型（5个）
├─ Swin-B
├─ MAE-ViT
├─ CoAtNet-1
├─ DeiT-III
└─ PVT-v2（可选）

Day 7: 初步评估
├─ 统计所有模型的baseline性能
├─ 绘制初步对比图
└─ 决策点：继续 or 调整？
```

**Week 2: 人类数据加载与初步对比**

```
Day 1-2: 人类行为数据
├─ 加载行为数据（derivatives/Behavioral_data/）
├─ 计算人类在各遮挡等级的准确率
└─ 准备人类错误样本列表

Day 3-4: 初步性能对比
├─ 绘制：人类vs 10个模型的性能曲线
├─ 统计分析：哪些模型显著优于人类？
└─ 识别：最有希望的模型（Top 3）

Day 5-6: 错误样本准备
├─ 识别：人类对但模型错的样本（每个模型20-30个）
├─ 识别：模型对但人类错的样本
└─ 整理：用于后续深入分析

Day 7: 阶段0总结
├─ 评估：是否有足够差异继续？
├─ 规划：阶段1的重点分析哪些模型？
└─ 决策：聚焦Top 5模型，还是全部分析？
```

**输出成果**：
- ✅ 10个模型的训练结果
- ✅ 人类vs AI的性能对比图表
- ✅ 错误样本列表
- ✅ 阶段1的分析计划

---

#### Week 3-4: 阶段1A - 错误模式深度分析

**Week 3: 错误样本聚类与可视化**

```
Day 1-2: 特征提取
├─ 提取所有错误样本的模型特征
├─ CNN: 使用中间层特征
├─ Transformer: 使用CLS token或patch embeddings
└─ 降维：PCA/t-SNE/UMAP

Day 3-4: 聚类分析
├─ K-means聚类错误样本
├─ 分析：不同聚类的共同特征
│  - 聚类1：70%遮挡 + 关键特征可见
│  - 聚类2：90%遮挡 + 只有噪声
│  - 聚类3：背景复杂 + 分散注意
└─ 可视化：聚类图

Day 5-6: 模型间错误重叠分析
├─ 计算错误样本的重叠度
│  - ViT vs ResNet: Jaccard相似度
│  - Transformer vs CNN: 总体重叠
│  - MAE vs 其他: 是否独特？
├─ 假设检验：
│  - CNN和Transformer的错误样本分布是否相同？
│  - Chi-square test
└─ 可视化：韦恩图

Day 7: 中期检查
├─ 是否有清晰的错误模式？
├─ 故事是否成型？
└─ 调整后续分析重点
```

**Week 4: 典型错误样本的深入分析**

```
Day 1-2: 选择代表性样本
├─ 类型A：所有模型都错（10-15个）
├─ 类型B：只有MAE对（5-10个）
├─ 类型C：CNN错但Transformer对（10-15个）
└─ 类型D：Transformer错但CNN对（5-10个）

Day 3-4: 可视化分析
├─ 为每个样本生成：
│  - 原始图像
│  - 遮挡mask
│  - Grad-CAM (CNN)
│  - Attention maps (Transformer)
│  - 预测置信度
└─ 分析：为什么模型会错？

Day 5-6: 定量特征分析
├─ 提取错误样本的图像特征：
│  - 遮挡比例
│  - 可见区域的比例
│  - 边缘密度
│  - 纹理复杂度
│  - 对称性
└─ 回归分析：哪些特征预测错误？

Day 7: 整理发现
├─ 总结：3-5个关键发现
├─ 准备：用于论文的结果部分
└─ 可视化：高质量图表
```

**输出成果**：
- ✅ 错误样本的聚类结果
- ✅ 模型间错误重叠度分析
- ✅ 典型样本的可视化分析
- ✅ 错误预测特征的回归分析

---

#### Week 5-6: 阶段1B - Attention与表征分析

**Week 5: Attention机制对比**

```
Day 1-2: Attention提取
├─ Transformer: 提取多层attention
│  - 最后层的attention weights
│  - 12个head的attention
│  - 平均attention
├─ CNN: Grad-CAM, Integrated Gradients
└─ 保存：所有样本的attention maps

Day 3-4: 定量分析
├─ Attention entropy（熵）
│  - 计算每个attention map的熵
│  - 对比：熵 vs 遮挡等级
│  - 假设：高遮挡 → 熵增加（更分散）
│
├─ Focus ratio（聚焦度）
│  - 计算attention的集中程度
│  - 对比：不同架构的聚焦策略
│
└─ Spatial distribution
   - Attention的分布模式
   - 与遮挡区域的关系

Day 5-6: 架构间对比
├─ CNN vs Transformer:
│  - CNN: 局部，聚焦于边缘/纹理
│  - Transformer: 全局，但更分散
│
├─ ViT vs Swin:
│  - ViT: 全局attention
│  - Swin: 局部窗口 + 层级融合
│
├─ MAE vs others:
│  - MAE的attention是否更聚焦？
│  - 是否更接近人类？
│
└─ 可视化：attention对比图

Day 7: 分析总结
├─ 哪些attention策略更有效？
├─ 是否与人类fMRI相关？
└─ 为模型设计提供指导
```

**Week 6: 表征空间分析**

```
Day 1-2: 特征提取
├─ 提取所有模型的倒数第二层特征
├─ 对于Transformer: CLS token或pooler输出
├─ 对于CNN: GAP后的特征
└─ 维度: 768-2048维

Day 3-4: 表征相似度分析
├─ 计算模型间表征相关性
│  - CCA (Canonical Correlation Analysis)
│  - Procrustes analysis
│  - Representational Similarity Analysis (RSA)
│
├─ 问题：
│  - ViT和ResNet的表征是否相似？
│  - MAE的表征是否独特？
│  - 遮挡如何改变表征？
│
└─ 可视化：相似度矩阵

Day 5-6: 表征的鲁棒性分析
├─ 对比：10% vs 90%遮挡的表征变化
│  - 表征漂移（representation drift）
│  - 线性可分性（能否线性分离类别？）
│
├─ 分析：
│  - 哪些模型的表征更稳定？
│  - 表征稳定性 vs 性能的关系
│
└─ 预期：
   - MAE的表征最稳定（masked modeling）
   - Transformer > CNN

Day 7: 整合表征分析结果
├─ 表征差异如何解释性能差异？
├─ 是否有明确的"好表征"模式？
└─ 准备：用于论文的表征分析章节
```

**输出成果**：
- ✅ Attention机制的定量对比
- ✅ 表征空间的相似度分析
- ✅ 遮挡鲁棒性与表征的关系
- ✅ 2-3个关于表征的发现

---

#### Week 7: 阶段1C - fMRI深度分析

```
Day 1-2: fMRI数据加载与预处理
├─ 使用之前探索的代码
├─ 加载：错误试次的fMRI数据
├─ 分组：
│  - 组1: 人类对但ViT错
│  - 组2: 人类对但ResNet错
│  - 组3: 所有模型都对
│  - 组4: 所有模型都错
└─ 预处理：标准化、空间平滑

Day 3-4: 激活差异分析
├─ 组间对比：
│  - 组1 vs 组3: ViT错误试次的特征
│  - 组2 vs 组3: ResNet错误试次的特征
│  - 组4 vs 组3: 困难试次的特征
│
├─ 统计检验：
│  - 体素wise t-test
│  - 多重比较校正（FDR）
│
└─ 可视化：
   - 脑激活差异图
   - 重要脑区的放大图

Day 5-6: 功能连接分析（如果有时间）
├─ 计算不同错误类型的功能连接
│  - LOC ↔ IPS
│  - 视觉皮层 ↔ 前额叶
│
├─ 分析：
│  - ViT错误试次的连接模式
│  - ResNet错误试次的连接模式
│  - 与行为准确率的相关
│
└─ 可视化：
   - 连接强度图
   - 相关性散点图

Day 7: fMRI分析总结
├─ 整理：3-5个fMRI发现
├─ 对比：与attention分析的一致性
└─ 为模型设计提供神经科学依据
```

**输出成果**：
- ✅ 不同错误试次的脑激活差异
- ✅ 关键脑区的激活强度
- ✅ 功能连接模式（可选）
- ✅ 神经科学证据支持模型设计

---

#### Week 8: 阶段1总结与论文撰写

```
Day 1-2: 结果整理
├─ 汇总所有分析结果：
│  - 性能对比
│  - 错误模式
│  - Attention分析
│  - 表征分析
│  - fMRI分析
│
└─ 整理成：5-7个核心发现

Day 3-4: 图表制作
├─ Figure 1: 任务和数据集介绍
├─ Figure 2: 性能对比曲线（10个模型）
├─ Figure 3: 错误模式分析
│  - (a) 错误样本韦恩图
│  - (b) 典型错误样本可视化
│  - (c) 错误预测特征
├─ Figure 4: Attention对比
│  - (a) Attention entropy
│  - (b) 典型样本的attention maps
├─ Figure 5: 表征分析
│  - (a) 表征相似度矩阵
│  - (b) 表征稳定性
├─ Figure 6: fMRI分析
│  - (a) 脑激活差异图
│  - (b) 功能连接
└─ Table 1-2: 定量结果

Day 5-7: 论文初稿
├─ Abstract
├─ Introduction
├─ Related Work
├─ Methods（简要）
├─ Results（详细）
├─ Discussion
└─ Conclusion

Day 7: 决策点
├─ 评估：当前结果是否足够投会议？
│  - 如果很强 → 投分析型论文（AAAI/IJCAI）
│  - 如果一般 → 继续阶段2，加模型创新
│
└─ 规划：阶段2的模型设计方向
```

**输出成果**：
- ✅ 完整的对比分析结果
- ✅ 5-7张高质量图表
- ✅ 分析型论文初稿
- ✅ 阶段2的设计规划

---

#### Week 9-12: 阶段2 - 模型创新（可选）

**Week 9: 模块设计（基于阶段1发现）**

```
基于阶段1的发现，设计模块：

发现1: MAE的masked modeling最robust
→ 设计：遮挡自适应的masked attention
  class MaskedAdaptiveAttention:
      def __init__(self):
          self.mask_generator = MaskGenerator()
      def forward(self, x, occ_level):
          # 根据遮挡度生成mask
          if occ_level > 0.7:
              mask = self.generate_large_mask()
          else:
              mask = self.generate_small_mask()
          # 只对unmasked regions计算attention
          ...

发现2: Transformer过度分散attention
→ 设计：聚焦约束的attention
  class FocusedAttention:
      def forward(self, x):
          attn = self.standard_attention(x)
          # 添加熵正则化
          entropy_reg = -sum(attn * log(attn))
          loss = entropy_reg
          return attn, loss

发现3: 混合架构平衡性能
→ 设计：动态路径选择
  class DynamicPathSelection:
      def forward(self, x, occ_level):
          if occ_level < 0.5:
              # 低遮挡：快速CNN路径
              return self.cnn_path(x)
          else:
              # 高遮挡：Transformer路径
              return self.transformer_path(x)

发现4: fMRI显示LOC-IPS连接重要
→ 设计：多尺度特征融合
  class BrainInspiredFusion:
      def __init__(self):
          self.low_stream = ConvStream()  # 模拟V1
          self.high_stream = ViTStream()  # 模拟LOC/IPS
          self.fusion = AdaptiveFusion()
      def forward(self, x):
          low_feat = self.low_stream(x)
          high_feat = self.high_stream(x)
          # 根据遮挡度动态调整融合权重
          ...
```

**Week 10-11: 实验验证**

```
实验1: 消融实验
┌──────────────────────────────────────┐
│ Baseline (ViT-B)        │ 85%        │
├──────────────────────────────────────┤
│ + Masked Attention      │ 87% (+2%)  │
├──────────────────────────────────────┤
│ + Focused Attention     │ 86% (+1%)  │
├──────────────────────────────────────┤
│ + Dynamic Path          │ 87% (+2%)  │
├──────────────────────────────────────┤
│ + Brain Fusion          │ 86% (+1%)  │
├──────────────────────────────────────┤
│ All Combined (Ours)     │ 90% (+5%)  │
└──────────────────────────────────────┘

实验2: 错误样本改进
┌──────────────────────────────────────────┐
│ 类型A样本（人类对，基线ViT错）           │
│   Baseline: 30% (42/142)                │
│   Ours:      55% (78/142)  +25%         │
│                                          │
│ 类型B样本（基线ViT对，人类错）           │
│   保持性能或轻微提升                      │
└──────────────────────────────────────────┘

实验3: 泛化性测试（可选）
├─ COCO-Occlusion
├─ PascalVOC-Occlusion
└─ 验证：在新数据上是否仍然有效
```

**Week 12: 论文完善与投稿**

```
任务：
1. 重写论文（加入模型创新部分）
   - 新增：Proposed Method章节
   - 修改：Abstract和Introduction
   - 扩展：Experiments章节

2. 补充实验
   - 消融实验
   - 泛化性测试
   - 对比实验

3. 可视化改进
   - 模块示意图
   - 改进效果对比图
   - 更多attention可视化

4. 撰写补充材料
   - 更多实验结果
   - 错误样本分析详情
   - 代码链接

5. 最终检查
   - 故事线是否完整？
   - 贡献是否清晰？
   - 实验是否充分？
```

---

## 6. 预期成果与贡献

### 6.1 学术贡献

```
贡献1: 大规模多模型对比研究
- 首次系统对比10+架构在遮挡识别上的表现
- 揭示架构设计对鲁棒性的影响
- 为模型选择提供实证依据

贡献2: 错误模式的深入分析
- 发现CNN vs Transformer的根本差异
- 量化错误样本的重叠度
- 提出错误预测的图像特征

贡献3: Attention机制的对比研究
- 定量分析不同架构的attention策略
- 揭示attention熵与性能的关系
- 提出attention聚焦度的度量

贡献4: 表征空间的分析
- RSA分析揭示表征相似度
- 发现MAE的表征独特性
- 揭示表征稳定性与鲁棒性的关系

贡献5: 神经科学证据（如果有fMRI分析）
- 连接AI错误与人类神经活动
- 提供brain-inspired设计的依据
- 架桥认知科学与AI

贡献6（可选）: 模型创新
- 基于分析发现的设计原则
- 提升遮挡鲁棒性的新模块
- 在多个数据集上验证
```

### 6.2 论文产出

#### 投稿目标（按优先级）

```
理想情况（阶段1+2都成功）：
1. CVPR 2026 (deadline ~2025年11月)
   - 概率：30%
   - 理由：多模型对比 + 模型创新 + 实验充分

2. ICCV 2025 (deadline ~2025年3月，可能来不及)
   - 如果时间允许，可投ICCV 2027

3. NeurIPS 2025 (deadline ~2025年5月，已过)
   - 可投NeurIPS 2026

中等情况（阶段1强）：
1. AAAI 2026 (deadline ~2025年8月)
   - 概率：60%
   - 理由：扎实的分析 + 多模型对比

2. IJCAI 2026 (deadline ~2026年1月)
   - 概率：70%
   - 理由：同上

3. WACV 2026 (deadline ~2025年8月)
   - 概率：80%
   - 理由：应用导向，适合对比研究

保底情况：
1. BMVC 2026 (英国机器视觉会议)
2. ICPR 2026 (模式识别)
3. CVPR/ICCV Workshop
```

#### 预期影响

```
如果成功发表：
- 引用量预期：20-50（前2年）
- 可能被后续工作引用为：
  - "首个系统对比多架构的研究"
  - "遮挡识别的benchmark研究"
  - "brain-inspired AI的案例"

潜在应用：
- 指导实际场景的模型选择
- 启发新的robust架构设计
- 推动可解释AI的发展
```

### 6.3 数据与代码开源

```
开源计划：
1. GitHub仓库
   - 训练代码
   - 评估脚本
   - 可视化工具

2. 数据集
   - 标注的错误样本
   - Attention maps
   - 表征特征

3. 模型权重
   - 所有训练好的模型
   - 便于复现和比较

预期：
- 100+ GitHub stars（如果工作质量高）
- 被其他研究使用
```

---

## 7. 风险评估与应对策略

### 7.1 主要风险

```
风险1: 时间不够（12周完成10个模型）
├─ 影响：无法完成所有实验
├─ 概率：高（~60%）
└─ 应对：
   ├─ 优先训练核心模型（5-7个）
   ├─ 使用预训练模型（只在OIID上fine-tune）
   └─ 降低评估复杂度

风险2: 模型间差异不显著
├─ 影响：没有有趣发现，难发论文
├─ 概率：中（~30%）
└─ 应对：
   ├─ 加深分析（attention、表征）
   ├─ 聚焦于差异最大的模型对
   └─ 补充fMRI分析增加深度

风险3: fMRI分析困难
├─ 影响：神经科学部分做不好
├─ 概率：高（~50%，因为你没有经验）
└─ 应对：
   ├─ 简化fMRI分析（只做激活对比）
   ├─ 寻求学长/导师帮助
   ├─ 或者：不做fMRI，只做行为对比
   └─ 降低对fMRI的依赖

风险4: 模型创新无效
├─ 影响：阶段2白费时间
├─ 概率：中（~40%）
└─ 应对：
   ├─ 阶段1独立成文
   ├─ 阶段2作为bonus
   └─ 如果不行，及时止损

风险5: 导师不满意方向
├─ 影响：需要换方向
├─ 概率：低（~20%）
└─ 应对：
   ├─ 及早与导师沟通
   ├─ 展示初步结果
   └─ 灵活调整
```

### 7.2 风险缓解策略

```
策略1: 分阶段产出
- 阶段1（8周）结束后，必须有可发表的成果
- 阶段2（4周）是锦上添花，不是必需

策略2: 核心模型优先
- 优先训练和分析：ViT, ResNet, MAE
- 其他模型：根据时间灵活调整

策略3: 及早验证
- Week 2就检查是否有足够差异
- Week 6检查故事是否成型
- 每个检查点都可以调整方向

策略4: 寻求帮助
- 学长：fMRI分析经验
- 导师：宏观指导
- 论文：找同学帮忙修改

策略5: 灵活投稿
- 准备多个投稿目标
- 根据结果质量调整目标
```

---

## 8. 资源需求

### 8.1 计算资源

```
训练需求：
- GPU: RTX 3090/4090 或 A100
- 内存: ≥32GB GPU RAM
- 存储: ≥500GB SSD
- 时间: 10个模型 × 100 epochs ≈ 2-3周（并行）

优化：
- 使用混合精度训练
- 梯度累积（减小batch size）
- 多GPU并行（如果有）
```

### 8.2 数据资源

```
必需：
- OIID数据集（已获取）
- 行为数据（derivatives/Behavioral_data/）
- fMRI数据（derivatives/pre-processed_data/）

可选：
- COCO验证集（人造遮挡）
- PascalVOC（人造遮挡）
- ImageNet子集（人造遮挡）
```

### 8.3 软件资源

```
深度学习：
- PyTorch ≥ 2.0
- timm (用于加载预训练模型)
- torchvision
- transformers (Hugging Face)

fMRI分析：
- nilearn
- nibabel
- antspy (可选)

可视化：
- matplotlib, seaborn
- plotly (交互式图表)
- itkwidgets (3D脑激活可视化)

分析：
- scipy, scikit-learn
- statsmodels
- pingouin (统计检验)
```

---

## 9. 伦理与可复现性

### 9.1 伦理考虑

```
数据使用：
- OIID数据集已获伦理批准
- 遵循数据集的使用协议
- 不尝试识别受试者身份

实验伦理：
- 不进行有害实验
- 不生成误导性结果
- 诚实报告负结果
```

### 9.2 可复现性

```
代码开源：
- 所有训练代码
- 评估脚本
- 随机种子设置
- 超参数记录

数据记录：
- 训练日志
- 模型检查点
- 实验配置文件

报告标准：
- 遵循REPRODUCIBILITY checklist
- 详细描述实验设置
- 提供模型架构图
```

---

## 10. 成功标准与评估

### 10.1 最低标准（保底）

```
✅ 完成5-7个模型的对比
✅ 发现2-3个有意义的差异
✅ 完成错误模式分析
✅ 撰写完整的分析型论文
✅ 投稿一个会议（WACV/ICPR或workshop）
```

### 10.2 理想标准（目标）

```
✅ 完成10个模型的对比
✅ 发现5-7个深入的模式
✅ 完成attention、表征、fMRI分析
✅ 设计并验证新模块（提升>3%）
✅ 投稿顶会（CVPR/ICCV/AAAI）
```

### 10.3 卓越标准（理想）

```
✅ 所有理想标准 +
✅ 泛化性验证（多个数据集）
✅ 开源代码和数据被广泛使用
✅ 发表在顶会并产生影响力
✅ 后续工作引用
```

---

## 11. 时间管理与里程碑

### 11.1 关键里程碑

```
M1 (Week 2): Baseline完成
  └─ 10个模型训练完毕，初步对比结果

M2 (Week 4): 错误模式清晰
  └─ 发现明确的错误模式差异

M3 (Week 6): Attention分析完成
  └─ 定量分析attention机制差异

M4 (Week 8): 阶段1论文初稿
  └─ 分析型论文完成，决策是否继续阶段2

M5 (Week 10): 模型设计验证
  └─ 新模块有效（>3%提升）

M6 (Week 12): 论文投稿
  └─ 完整论文提交
```

### 11.2 每周检查点

```
每周五下午：
- 检查本周任务完成度
- 评估下周计划是否合理
- 及时调整（如果遇到困难）
- 与导师/学长沟通
```

---

## 12. 后续研究方向

### 12.1 短期延伸（6个月内）

```
方向1: 扩展到更多架构
├─ Vision Transformers变体（DINO, BEiTv2）
├─ 状态空间模型（Mamba, Vision Mamba）
└─ Diffusion models for recognition

方向2: 扩展到更多任务
├─ 目标检测（不只是分类）
├─ 语义分割
└─ 视频理解

方向3: 扩展到更多数据集
├─ 真实遮挡数据集（如BDD100K）
├─ 医学影像遮挡
└─ 遥感图像遮挡
```

### 12.2 长期愿景（1-2年）

```
愿景1: Brain-inspired AI架构
├─ 基于fMRI发现的新架构范式
├─ 可解释的robust recognition
└─ 类人认知的AI系统

愿景2: 理论框架
├─ 架构-性能关系的理论
├─ 遮挡鲁棒性的计算理论
└─ Human-AI对齐的量化方法

愿景3: 应用落地
├─ 自动驾驶（遮挡行人检测）
├─ 医学诊断（遮挡病灶识别）
└─ 安防监控（遮挡目标追踪）
```

---

## 13. 参考文献（初步）

```
[1] Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", ICLR 2021

[2] Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows", ICCV 2021

[3] He et al., "Deep Residual Learning for Image Recognition", CVPR 2016

[4] Touvron et al., "Training data-efficient image transformers & distillation through attention", ICML 2021

[5] He et al., "Masked Autoencoders Are Scalable Vision Learners", CVPR 2022

[6] Bao et al., "BEiT: BERT Pre-Training of Image Transformers", ICCV 2022

[7] Ge et al., "ConvNeXt: A ConvNet for the 2020s", CVPR 2022

[8] Dai et al., "CoAtNet: Marrying Convolution and Attention for All Data Sizes", ICML 2021

[9] Radford et al., "Learning Transferable Visual Models From Natural Language Supervision", ICML 2021

[10] Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision", 2023

[11] Human vs AI相关研究（补充完整）
```

---

## 附录A：模型配置详情

### A.1 模型架构详细参数

```
ViT-B/16:
├─ Image size: 224×224
├─ Patch size: 16×16
├─ Layers: 12
├─ Heads: 12
├─ Dim: 768
└─ Params: 86M

Swin-B:
├─ Image size: 224×224
├─ Window size: 7×7
├─ Layers: [2,2,18,2]
├─ Heads: [4,8,16,32]
└─ Params: 88M

ResNet-50:
├─ Layers: [3,4,6,3]
├─ Channels: [64,128,256,512]
└─ Params: 26M

... (其他模型类似)
```

### A.2 训练超参数

```
Optimizer:
├─ Type: AdamW
├─ LR: 3e-4 (ViT), 1e-3 (ResNet)
├─ Weight decay: 0.05
└─ Betas: (0.9, 0.999)

Scheduler:
├─ Type: Cosine annealing
├─ Warmup epochs: 5
├─ Min LR: 1e-6
└─ T_max: 100

Data augmentation:
├─ RandomResizedCrop(224)
├─ RandomHorizontalFlip(p=0.5)
├─ RandomRotation(15°)
└─ ColorJitter(brightness=0.2, contrast=0.2)
```

---

## 附录B：评估指标计算

### B.1 主要指标

```python
# 准确率
accuracy = correct / total

# F1分数
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)

# 遮挡鲁棒性
robustness = acc_90 - acc_10
# 越小越好（表示在遮挡下性能下降少）

# 人类相似度
error_overlap = |model_errors ∩ human_errors| / |model_errors ∪ human_errors|
# Jaccard相似度

# Attention熵
entropy = -sum(p * log(p))
# p是attention weights的归一化分布
```

---

## 附录C：时间表甘特图

```
Week:  1  2  3  4  5  6  7  8  9 10 11 12
─────────────────────────────────────────
阶段0  ████
模型训练    ████████████
错误分析          ████████
Attention分析          ████████
表征分析              ████████
fMRI分析                ████████
论文撰写                    ████████
模型设计                        ████████
实验验证                          ████████
论文完善                              ████
```

---

## 结论

本研究计划提供了一个全面的、系统的多模型对比研究框架，旨在深入理解人类与AI在遮挡识别上的差异。通过对比10+种架构、多角度分析（性能、错误模式、attention、表征、fMRI），我们将能够：

1. 揭示不同架构的内在机制差异
2. 为模型选择提供实证依据
3. 为未来架构设计提供指导原则
4. 桥桥认知科学与深度学习

**关键是分阶段执行、灵活调整、确保每个阶段都有可发表的成果。**

---

**文档版本**: 1.0
**创建日期**: 2026-01-01
**作者**: [你的名字]
**最后更新**: 2026-01-01
