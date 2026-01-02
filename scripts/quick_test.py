"""Quick test script to verify environment and model setup."""
import sys
from pathlib import Path
import yaml
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.stimulus_dataset import create_dataloaders
from src.models.pretrained_loader import create_vit_b16_pretrained


def test_environment():
    """Test basic environment setup."""
    print("=" * 60)
    print("环境检查")
    print("=" * 60)

    # PyTorch info
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU显存: {gpu_mem:.1f} GB")
        print(f"推荐batch_size: {32 if gpu_mem > 12 else 16 if gpu_mem > 8 else 8}")
    else:
        print("⚠️  警告: 没有检测到GPU，训练会非常慢")

    print(f"当前设备: {'cuda' if torch.cuda.is_available() else 'cpu'}")


def test_config():
    """Test configuration file."""
    print("\n" + "=" * 60)
    print("配置文件检查")
    print("=" * 60)

    config_path = project_root / 'configs' / 'vit_b16_pretrained.yaml'

    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return None

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"✓ 配置文件加载成功: {config_path}")
    print(f"  实验名称: {config['experiment']['name']}")
    print(f"  模型类型: {config['model']['type']}")
    print(f"  使用预训练: {config['model']['pretrained']}")
    print(f"  训练轮数: {config['training']['epochs']}")
    print(f"  学习率: {config['training']['optimizer']['lr']}")

    return config


def test_dataset(config):
    """Test dataset loading."""
    print("\n" + "=" * 60)
    print("数据集检查")
    print("=" * 60)

    dataset_root = config['dataset']['root']
    print(f"数据集路径: {dataset_root}")

    # Check if dataset exists
    dataset_path = Path(dataset_root)
    if not dataset_path.exists():
        print(f"❌ 数据集路径不存在: {dataset_root}")
        print("请检查配置文件中的数据集路径")
        return False

    print(f"✓ 数据集路径存在")

    # Try to load a small subset
    print("\n正在加载小规模数据集进行测试...")
    try:
        dataloaders = create_dataloaders(
            dataset_root=dataset_root,
            train_subjects=config['dataset']['train_subjects'][:3],  # Only 3 subjects
            val_subjects=config['dataset']['val_subjects'][:2],      # Only 2 subjects
            batch_size=8,
            img_size=224,
            num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
            occlusion_levels=config['dataset'].get('occlusion_levels'),
        )

        print(f"✓ 训练集样本数: {len(dataloaders['train'].dataset)}")
        print(f"✓ 验证集样本数: {len(dataloaders['val'].dataset)}")

        # Get one batch
        images, labels, meta = next(iter(dataloaders['train']))
        print(f"\n单个batch信息:")
        print(f"  图像shape: {images.shape}")
        print(f"  标签shape: {labels.shape}")
        print(f"  标签类别: {labels.unique().tolist()}")
        if isinstance(meta, list) and len(meta) > 0:
            print(f"  样本元数据示例: {meta[0]}")
        elif isinstance(meta, dict):
            print(f"  样本元数据示例: {meta}")

        return dataloaders

    except Exception as e:
        print(f"❌ 数据集加载失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_model(config):
    """Test model creation."""
    print("\n" + "=" * 60)
    print("模型检查")
    print("=" * 60)

    try:
        print("正在创建ViT-B/16模型...")
        model = create_vit_b16_pretrained(
            num_classes=config['model']['num_classes'],
            pretrained=False,  # Don't download weights for quick test
            freeze_backbone=config['model']['freeze_backbone'],
        )

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"✓ 模型创建成功")
        print(f"  总参数量: {total_params / 1e6:.2f}M")
        print(f"  可训练参数: {trainable_params / 1e6:.2f}M")
        print(f"  冻结参数: {(total_params - trainable_params) / 1e6:.2f}M")

        return model

    except Exception as e:
        print(f"❌ 模型创建失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_forward_pass(model, dataloader):
    """Test forward pass."""
    print("\n" + "=" * 60)
    print("前向传播测试")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    try:
        # Get one batch
        images, labels, meta = next(iter(dataloader))
        images = images.to(device)
        labels = labels.to(device)

        print(f"输入shape: {images.shape}")

        # Forward pass
        with torch.no_grad():
            outputs = model(images)

        print(f"✓ 前向传播成功")
        print(f"  输出shape: {outputs.shape}")
        print(f"  输出范围: [{outputs.min().item():.3f}, {outputs.max().item():.3f}]")

        # Test loss computation
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        print(f"  初始损失: {loss.item():.4f}")

        # Test predictions
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == labels).float().mean().item()
        print(f"  随机准确率: {accuracy:.4f} (期望约0.5)")

        return True

    except Exception as e:
        print(f"❌ 前向传播失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_pretrained_download():
    """Test pretrained weights download."""
    print("\n" + "=" * 60)
    print("预训练权重下载测试")
    print("=" * 60)

    print("正在下载ViT-B/16 ImageNet预训练权重...")
    print("(首次运行会下载约350MB，后续会使用缓存)")

    try:
        model = create_vit_b16_pretrained(
            num_classes=2,
            pretrained=True,  # Download pretrained weights
        )
        print("✓ 预训练权重下载并加载成功")
        return True

    except Exception as e:
        print(f"❌ 预训练权重加载失败: {str(e)}")
        print("\n可能的原因:")
        print("  1. 网络连接问题")
        print("  2. Hugging Face下载受限")
        print("\n解决方案:")
        print("  - 检查网络连接")
        print("  - 使用代理或VPN")
        print("  - 手动下载权重文件")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("ViT-B/16 训练环境快速验证")
    print("=" * 80 + "\n")

    # Test 1: Environment
    test_environment()

    # Test 2: Config
    config = test_config()
    if config is None:
        print("\n❌ 配置文件测试失败，停止后续测试")
        return

    # Test 3: Dataset
    dataloaders = test_dataset(config)
    if dataloaders is None:
        print("\n❌ 数据集测试失败，停止后续测试")
        return

    # Test 4: Model
    model = test_model(config)
    if model is None:
        print("\n❌ 模型测试失败，停止后续测试")
        return

    # Test 5: Forward pass
    success = test_forward_pass(model, dataloaders['train'])
    if not success:
        print("\n❌ 前向传播测试失败")
        return

    # Test 6: Pretrained weights (optional, can be slow)
    print("\n是否测试预训练权重下载? (这会下载约350MB)")
    response = input("输入 'y' 继续，其他键跳过: ").strip().lower()
    if response == 'y':
        test_pretrained_download()
    else:
        print("跳过预训练权重下载测试")

    # Summary
    print("\n" + "=" * 80)
    print("✅ 所有基础测试通过！")
    print("=" * 80)
    print("\n下一步:")
    print("  1. 运行5-epoch快速训练验证:")
    print("     python train_vit.py --config configs/vit_b16_pretrained.yaml")
    print("\n  2. 在训练前，建议先临时修改配置文件:")
    print("     training.epochs: 5  # 改为5进行快速验证")
    print("\n  3. 启动TensorBoard监控训练:")
    print("     tensorboard --logdir experiments/phase1_baseline/vit_b16/logs")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
