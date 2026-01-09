# 数据扩充计划

**创建日期**: 2026-01-08
**目标**: 通过数据增强和遮挡生成,将训练集从210张扩充到10,000+张

---

## 📊 当前数据集状态分析

### 原始数据 (stimuli_original)

**总计**: 300张原始图片

**类别分布**:
- Aircraft1: 150张
- Aircraft2: 150张

**遮挡等级分布**:
- 10%遮挡: 100张 (每类50张)
- 70%遮挡: 100张 (每类50张)
- 90%遮挡: 100张 (每类50张)

**文件命名格式**:
```
Aircraft1_10%_1_original.jpg
Aircraft1_70%_15_original.jpg
Aircraft2_90%_23_original.jpg
```

### 当前项目数据集

**训练集**: 210张
**验证集**: 42张
**测试集**: 48张
**总计**: 300张

**问题**:
- 训练集太小 (~210张)
- 无法充分训练深度模型
- 容易过拟合

---

## 🎯 数据扩充目标

### 目标数据集规模

**训练集**: 10,000-15,000张 (全部为新生成的增强图片)
**验证集**: 300张 (所有原始人类实验图片)
**测试集**: 0张 (合并到验证集)

**理由**:
1. **训练集**: 使用增强数据,避免模型记忆人类实验图片
2. **验证集**: 使用原始图片,确保与人类表现直接对比
3. **测试集**: 不需要单独测试集,验证集即为最终评估集

### 数据扩充倍数

**目标**: 从300张原始图片生成 **10,000-15,000张** 训练图片

**扩充倍数**: 33-50倍

**策略**:
- 每张原始图片 → 生成33-50个变体
- 每个变体应用不同的增强 + 随机遮挡

---

## 🔧 数据扩充策略

### Phase 1: 几何变换 (Geometric Transformations)

对每张原始图片应用多种几何变换:

#### 1.1 旋转 (Rotation)

**参数**:
```python
rotation_angles = [-30, -20, -10, -5, 0, 5, 10, 20, 30]  # 9个角度
```

**理由**:
- 飞机在不同角度下的视角
- 增加视角多样性
- 保持飞机的可识别性

#### 1.2 缩放 (Scaling)

**参数**:
```python
scale_factors = [0.8, 0.9, 1.0, 1.1, 1.2]  # 5个缩放比例
```

**理由**:
- 模拟不同距离的飞机
- 增加尺度不变性

#### 1.3 平移 (Translation)

**参数**:
```python
translation_range = [-20, -10, 0, 10, 20]  # 像素偏移
```

**理由**:
- 飞机在不同位置
- 增加位置不变性

#### 1.4 水平翻转 (Horizontal Flip)

**参数**:
```python
flip = [False, True]  # 2种状态
```

**理由**:
- 镜像对称
- 增加数据多样性

### Phase 2: 颜色变换 (Color Transformations)

#### 2.1 亮度调整 (Brightness)

**参数**:
```python
brightness_factors = [0.7, 0.85, 1.0, 1.15, 1.3]  # 5个亮度级别
```

**理由**:
- 模拟不同光照条件
- 增加光照不变性

#### 2.2 对比度调整 (Contrast)

**参数**:
```python
contrast_factors = [0.8, 0.9, 1.0, 1.1, 1.2]  # 5个对比度级别
```

#### 2.3 饱和度调整 (Saturation)

**参数**:
```python
saturation_factors = [0.7, 0.85, 1.0, 1.15, 1.3]  # 5个饱和度级别
```

#### 2.4 色调调整 (Hue)

**参数**:
```python
hue_shifts = [-0.1, -0.05, 0, 0.05, 0.1]  # 5个色调偏移
```

### Phase 3: 随机遮挡生成 (Random Occlusion)

基于现有的`addmask.m`程序,为每个增强后的图片生成随机遮挡。

#### 3.1 遮挡等级

**三个等级**:
- 10%遮挡: 覆盖飞机区域的10%
- 70%遮挡: 覆盖飞机区域的70%
- 90%遮挡: 覆盖飞机区域的90%

#### 3.2 遮挡方式

**参数** (来自addmask.m):
```python
mask_size = 10  # 10×10像素的黑色方块
mask_color = [0, 0, 0]  # 黑色
placement = "random"  # 随机放置在飞机区域
allow_overlap = True  # 允许重叠
```

**关键特性**:
1. 只在飞机区域内添加遮挡 (不遮挡背景)
2. 使用10×10像素的小方块
3. 随机放置直到达到目标覆盖率
4. 允许方块重叠

#### 3.3 随机性

**每次生成不同的遮挡模式**:
- 相同的增强图片 + 不同的随机种子 → 不同的遮挡模式
- 每个增强图片可以生成多个遮挡变体

---

## 📐 数据生成管道设计

### 管道流程

```
原始图片 (300张)
    ↓
[Phase 1: 几何变换]
    ├─ 旋转 (9种)
    ├─ 缩放 (5种)
    ├─ 平移 (5种)
    └─ 翻转 (2种)
    ↓
增强图片 (每张原始图片 → 多个变体)
    ↓
[Phase 2: 颜色变换]
    ├─ 亮度 (5种)
    ├─ 对比度 (5种)
    ├─ 饱和度 (5种)
    └─ 色调 (5种)
    ↓
颜色增强图片
    ↓
[Phase 3: 随机遮挡]
    ├─ 10%遮挡 (随机种子1-N)
    ├─ 70%遮挡 (随机种子1-N)
    └─ 90%遮挡 (随机种子1-N)
    ↓
最终训练集 (10,000-15,000张)
```

### 生成策略

#### 策略A: 组合采样 (推荐)

**方法**: 随机组合不同的增强方式

```python
for each original_image in original_images:
    for i in range(33):  # 每张原始图片生成33个变体
        # 随机选择增强参数
        rotation = random.choice(rotation_angles)
        scale = random.choice(scale_factors)
        flip = random.choice([True, False])
        brightness = random.choice(brightness_factors)
        contrast = random.choice(contrast_factors)

        # 应用增强
        augmented = apply_augmentations(
            original_image,
            rotation=rotation,
            scale=scale,
            flip=flip,
            brightness=brightness,
            contrast=contrast
        )

        # 随机选择遮挡等级
        occlusion_level = random.choice([0.1, 0.7, 0.9])

        # 生成随机遮挡
        final_image = add_random_occlusion(
            augmented,
            coverage=occlusion_level,
            seed=i  # 不同的随机种子
        )

        # 保存
        save_image(final_image, f"{class}_{occlusion_level*100}%_aug{i}.jpg")
```

**优点**:
- 最大化数据多样性
- 避免系统性偏差
- 更接近真实场景的变化

**生成数量**: 300张 × 33个变体 = **9,900张**

#### 策略B: 网格采样 (备选)

**方法**: 系统地遍历所有增强组合

```python
for each original_image in original_images:
    for rotation in rotation_angles:
        for scale in scale_factors:
            for flip in [True, False]:
                for brightness in brightness_factors:
                    # ... 其他增强

                    augmented = apply_augmentations(...)

                    for occlusion_level in [0.1, 0.7, 0.9]:
                        final_image = add_random_occlusion(...)
                        save_image(...)
```

**优点**:
- 覆盖所有组合
- 系统性强

**缺点**:
- 组合数量爆炸 (9×5×2×5×5×3 = 6,750种组合/张)
- 可能生成过多相似图片

**生成数量**: 300张 × 6,750组合 = **2,025,000张** (太多!)

**建议**: 使用策略A (组合采样)

---

## 🗂️ 数据集划分策略

### 新的划分方案

```
原始数据 (300张)
├─ 全部放入验证集 (300张)
│  ├─ Aircraft1: 150张
│  │  ├─ 10%: 50张
│  │  ├─ 70%: 50张
│  │  └─ 90%: 50张
│  └─ Aircraft2: 150张
│     ├─ 10%: 50张
│     ├─ 70%: 50张
│     └─ 90%: 50张
│
└─ 增强数据 (10,000张)
   └─ 全部放入训练集
      ├─ Aircraft1: 5,000张
      │  ├─ 10%: ~1,667张
      │  ├─ 70%: ~1,667张
      │  └─ 90%: ~1,666张
      └─ Aircraft2: 5,000张
         ├─ 10%: ~1,667张
         ├─ 70%: ~1,667张
         └─ 90%: ~1,666张
```

### 关键原则

1. **训练集 = 100%增强数据**
   - 避免模型记忆人类实验图片
   - 确保泛化能力

2. **验证集 = 100%原始数据**
   - 与人类表现直接对比
   - 真实评估模型性能

3. **类别平衡**
   - Aircraft1 : Aircraft2 = 1:1
   - 10% : 70% : 90% = 1:1:1

4. **无数据泄露**
   - 训练集和验证集完全独立
   - 训练集中的增强图片来自不同的原始图片

---

## 💻 实现计划

### 需要创建的脚本

#### 1. 数据增强脚本

**文件**: `scripts/data_preparation/augment_dataset.py`

**功能**:
- 读取原始图片
- 应用几何和颜色变换
- 生成增强图片
- 保存到临时目录

**输入**:
- 原始图片目录: `e:\Dataset\ds005226\derivatives\stimuli_dataset\stimuli_original`

**输出**:
- 增强图片目录: `data/augmented_images` (临时)

**参数**:
```python
--input-dir: 原始图片目录
--output-dir: 输出目录
--num-augmentations: 每张图片生成的增强数量 (默认33)
--seed: 随机种子
```

#### 2. 遮挡生成脚本 (Python版)

**文件**: `scripts/data_preparation/add_occlusion.py`

**功能**:
- 将MATLAB的addmask.m转换为Python
- 为增强图片添加随机遮挡
- 保存最终训练图片

**输入**:
- 增强图片目录: `data/augmented_images`

**输出**:
- 训练集目录: `data/train_augmented`

**参数**:
```python
--input-dir: 增强图片目录
--output-dir: 输出目录
--mask-size: 遮挡方块大小 (默认10)
--occlusion-levels: 遮挡等级列表 (默认[0.1, 0.7, 0.9])
```

#### 3. 数据集划分脚本

**文件**: `scripts/data_preparation/split_dataset.py`

**功能**:
- 将原始图片复制到验证集
- 将增强图片复制到训练集
- 验证数据集平衡性

**输入**:
- 原始图片: `e:\Dataset\ds005226\derivatives\stimuli_dataset\stimuli_original`
- 增强图片: `data/train_augmented`

**输出**:
- 训练集: `data/train` (10,000张增强图片)
- 验证集: `data/val` (300张原始图片)

#### 4. 数据集验证脚本

**文件**: `scripts/data_preparation/validate_dataset.py`

**功能**:
- 检查数据集完整性
- 验证类别平衡
- 验证遮挡等级分布
- 生成统计报告

**输出**:
- 统计报告: `data/dataset_statistics.json`
- 可视化图表: `data/dataset_distribution.png`

### 完整的数据生成流程

```bash
# Step 1: 数据增强
python scripts/data_preparation/augment_dataset.py \
  --input-dir "e:\Dataset\ds005226\derivatives\stimuli_dataset\stimuli_original" \
  --output-dir "data/augmented_images" \
  --num-augmentations 33 \
  --seed 42

# Step 2: 添加遮挡
python scripts/data_preparation/add_occlusion.py \
  --input-dir "data/augmented_images" \
  --output-dir "data/train_augmented" \
  --mask-size 10 \
  --occlusion-levels 0.1 0.7 0.9

# Step 3: 划分数据集
python scripts/data_preparation/split_dataset.py \
  --original-dir "e:\Dataset\ds005226\derivatives\stimuli_dataset\stimuli_original" \
  --augmented-dir "data/train_augmented" \
  --train-dir "data/train" \
  --val-dir "data/val"

# Step 4: 验证数据集
python scripts/data_preparation/validate_dataset.py \
  --train-dir "data/train" \
  --val-dir "data/val" \
  --output "data/dataset_statistics.json"
```

---

## 📋 需要获取的信息

### 1. 原始图片信息 ✅

**已获取**:
- 总数: 300张
- 类别: Aircraft1 (150), Aircraft2 (150)
- 遮挡等级: 10% (100), 70% (100), 90% (100)
- 文件格式: JPG
- 命名格式: `Aircraft{1|2}_{10|70|90}%_{id}_original.jpg`

### 2. 图片尺寸和格式

**需要确认**:
```bash
# 检查图片尺寸
python -c "
from PIL import Image
img = Image.open('e:/Dataset/ds005226/derivatives/stimuli_dataset/stimuli_original/Aircraft1_10%_1_original.jpg')
print(f'尺寸: {img.size}')
print(f'模式: {img.mode}')
print(f'格式: {img.format}')
"
```

**预期**:
- 尺寸: 可能是224×224或其他
- 模式: RGB
- 格式: JPEG

### 3. 飞机区域检测方法

**来自addmask.m**:
```matlab
gray_img = rgb2gray(img);
plane_mask = gray_img > 0;  % 非黑色背景即为飞机
```

**需要确认**:
- 背景是否为纯黑色 (0, 0, 0)?
- 飞机区域是否有清晰边界?
- 是否需要更复杂的分割方法?

### 4. 遮挡方块参数

**来自addmask.m** ✅:
- 方块大小: 10×10像素
- 方块颜色: 黑色 (0, 0, 0)
- 放置方式: 随机放置在飞机区域
- 覆盖率: 10%, 70%, 90%

### 5. 计算资源需求

**需要评估**:
- 生成10,000张图片需要多长时间?
- 需要多少存储空间?
- 是否需要GPU加速?

**估算**:
```
假设:
- 每张图片处理时间: 1秒
- 总图片数: 10,000张
- 总时间: 10,000秒 ≈ 2.8小时

存储空间:
- 每张图片大小: ~20KB (JPEG压缩)
- 总大小: 10,000 × 20KB = 200MB
```

### 6. 数据增强参数调优

**需要实验确定**:
- 旋转角度范围: [-30°, 30°] 是否合适?
- 缩放比例: [0.8, 1.2] 是否合适?
- 颜色变换强度: 是否会影响飞机可识别性?

**建议**:
- 先生成100张样本
- 人工检查质量
- 调整参数后再大规模生成

### 7. 验证集使用策略

**需要确认**:
- 是否所有300张原始图片都用于验证?
- 还是保留一部分作为测试集?

**建议**:
- 全部300张用于验证 (与人类对比)
- 不需要单独的测试集

---

## ⚠️ 潜在问题与解决方案

### 问题1: 增强后的图片质量下降

**原因**:
- 过度的几何变换
- 过度的颜色变换

**解决方案**:
- 限制变换强度
- 人工检查样本质量
- 使用高质量的插值方法 (BICUBIC)

### 问题2: 遮挡模式不够随机

**原因**:
- 随机种子设置不当
- 遮挡算法的局限性

**解决方案**:
- 为每张图片使用不同的随机种子
- 增加遮挡模式的多样性 (不同的方块大小?)

### 问题3: 类别不平衡

**原因**:
- 增强过程中某些类别生成失败

**解决方案**:
- 在生成过程中实时监控类别分布
- 确保每个类别生成相同数量的图片

### 问题4: 训练时间过长

**原因**:
- 训练集太大 (10,000张)

**解决方案**:
- 使用更大的batch size
- 使用混合精度训练
- 使用更快的数据加载器 (多进程)

### 问题5: 模型过拟合到增强模式

**原因**:
- 增强方式过于单一
- 模型学习到增强的artifact

**解决方案**:
- 增加增强方式的多样性
- 使用更强的正则化
- 在训练时也应用在线数据增强

---

## 📊 预期效果

### 数据集规模对比

| 数据集 | 当前规模 | 扩充后规模 | 增长倍数 |
|--------|----------|------------|----------|
| 训练集 | 210张 | 10,000张 | 47.6× |
| 验证集 | 42张 | 300张 | 7.1× |
| 测试集 | 48张 | 0张 | - |
| **总计** | **300张** | **10,300张** | **34.3×** |

### 预期性能提升

基于文献和经验:

**当前性能** (210张训练集):
- ViT-B/16: 52%
- ResNet-50: 46%

**预期性能** (10,000张训练集):
- ViT-B/16: 70-80% (+18-28%)
- ResNet-50: 65-75% (+19-29%)
- MAE-ViT: 75-85% (新模型)

**理由**:
1. **数据量增加47倍**: 显著减少过拟合
2. **数据多样性**: 增强后的图片覆盖更多变化
3. **文献支持**: 小数据集扩充通常带来15-30%提升

---

## 🚀 实施时间表

### Week 1: 准备与验证 (2-3天)

**Day 1**:
- [ ] 检查原始图片尺寸和格式
- [ ] 验证飞机区域检测方法
- [ ] 创建数据增强脚本框架

**Day 2**:
- [ ] 实现几何变换函数
- [ ] 实现颜色变换函数
- [ ] 生成100张样本进行质量检查

**Day 3**:
- [ ] 将MATLAB的addmask.m转换为Python
- [ ] 测试遮挡生成功能
- [ ] 调整参数

### Week 2: 大规模生成 (2-3天)

**Day 1**:
- [ ] 运行数据增强脚本 (生成~10,000张增强图片)
- [ ] 监控生成过程

**Day 2**:
- [ ] 运行遮挡生成脚本 (为增强图片添加遮挡)
- [ ] 验证遮挡质量

**Day 3**:
- [ ] 划分数据集 (训练集/验证集)
- [ ] 验证数据集完整性和平衡性
- [ ] 生成统计报告

### Week 3: 训练与评估 (3-5天)

**Day 1-3**:
- [ ] 使用新数据集重新训练ViT-B/16
- [ ] 使用新数据集重新训练ResNet-50
- [ ] 训练MAE-ViT

**Day 4-5**:
- [ ] 评估模型性能
- [ ] 对比新旧数据集的效果
- [ ] 决策: 是否需要调整数据增强策略

---

## ✅ 检查清单

### 准备阶段

- [ ] 确认原始图片路径可访问
- [ ] 确认图片尺寸和格式
- [ ] 确认飞机区域检测方法有效
- [ ] 确认遮挡参数合理

### 实现阶段

- [ ] 创建 `augment_dataset.py`
- [ ] 创建 `add_occlusion.py`
- [ ] 创建 `split_dataset.py`
- [ ] 创建 `validate_dataset.py`

### 生成阶段

- [ ] 生成100张样本并人工检查
- [ ] 调整参数
- [ ] 大规模生成10,000张图片
- [ ] 验证数据集质量

### 训练阶段

- [ ] 更新数据加载器配置
- [ ] 重新训练所有模型
- [ ] 评估性能提升
- [ ] 对比新旧数据集效果

---

## 📞 下一步行动

### 立即需要确认的信息

1. **图片尺寸**: 运行以下命令确认
```bash
python -c "from PIL import Image; img = Image.open('e:/Dataset/ds005226/derivatives/stimuli_dataset/stimuli_original/Aircraft1_10%_1_original.jpg'); print(f'尺寸: {img.size}, 模式: {img.mode}')"
```

2. **飞机区域检测**: 验证背景是否为纯黑色
```python
import numpy as np
from PIL import Image

img = Image.open('e:/Dataset/ds005226/derivatives/stimuli_dataset/stimuli_original/Aircraft1_10%_1_original.jpg')
img_array = np.array(img)

# 检查背景
background_pixels = img_array[img_array.sum(axis=2) == 0]
print(f"黑色背景像素数: {len(background_pixels)}")
```

3. **存储空间**: 确认有足够空间存储10,000张图片 (~200MB)

### 开始实现

一旦确认上述信息,我将:
1. 创建4个数据准备脚本
2. 生成100张样本供你检查
3. 根据反馈调整参数
4. 大规模生成训练数据

---

**文档版本**: 1.0
**创建日期**: 2026-01-08
**状态**: 等待信息确认
**下次更新**: 信息确认后
