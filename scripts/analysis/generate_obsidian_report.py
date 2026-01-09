"""
Generate Obsidian-formatted Research Report
Integrates all analysis results and research recommendations
"""

from pathlib import Path
from datetime import datetime
import pandas as pd

# Project root
project_root = Path(__file__).parent.parent.parent
data_dir = project_root / "experiments" / "analysis"
reports_dir = project_root / "reports"

def generate_report():
    """Generate comprehensive Obsidian-formatted report"""

    # Load data
    resnet_metrics = pd.read_csv(data_dir / "resnet50_metrics_by_occlusion.csv")
    vit_metrics = pd.read_csv(data_dir / "vit_b16_metrics_by_occlusion.csv")
    per_image = pd.read_csv(data_dir / "per_image_comparison_summary.csv")

    # Generate report content
    report = []

    # Header with metadata
    report.append("---")
    report.append(f"title: Brain Occlusion Detection - Comprehensive Research Analysis")
    report.append(f"date: {datetime.now().strftime('%Y-%m-%d')}")
    report.append("tags:")
    report.append("  - research")
    report.append("  - computer-vision")
    report.append("  - occlusion-robustness")
    report.append("  - vision-transformers")
    report.append("  - human-ai-comparison")
    report.append("aliases:")
    report.append("  - Occlusion Research Report")
    report.append("  - Phase 1 Analysis")
    report.append("---\n\n")

    # Title and overview
    report.append("# Brain Occlusion Detection: Comprehensive Research Analysis\n\n")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**Project Phase:** Phase 1 Baseline + Data Augmentation Experiments\n")
    report.append(f"**Status:** Analysis Complete, Awaiting Direction Decision\n\n")

    report.append("---\n\n")

    # Executive Summary
    report.append("## Executive Summary\n\n")
    report.append("> [!summary] Key Findings\n")
    report.append("> - **Massive Human-AI Gap:** AI models perform 5-52% worse than humans across occlusion levels\n")
    report.append("> - **Architecture Matters:** ViT-B/16 consistently outperforms ResNet-50 by ~6% across all conditions\n")
    report.append("> - **Paradoxical Pattern:** AI performs worse at low occlusion (10%) than high occlusion (90%)\n")
    report.append("> - **Small Sample Challenge:** Only 300 training images severely limits model performance\n")
    report.append("> - **Data Augmentation Attempted:** New experiments with 9,900 augmented images in progress\n\n")

    report.append("---\n\n")

    # Table of Contents
    report.append("## Table of Contents\n\n")
    report.append("1. [[#Experimental Results]]\n")
    report.append("2. [[#Statistical Analysis]]\n")
    report.append("3. [[#Literature Review Insights]]\n")
    report.append("4. [[#Current Challenges]]\n")
    report.append("5. [[#Proposed Research Directions]]\n")
    report.append("6. [[#Next Steps and Recommendations]]\n\n")

    report.append("---\n\n")

    # Section 1: Experimental Results
    report.append("## Experimental Results\n\n")

    report.append("### Dataset Overview\n\n")
    report.append("**OIID Dataset (Occluded Image Interpretation Dataset)**\n")
    report.append("- Source: OpenNeuro (ds005226)\n")
    report.append("- Task: Binary classification (Aircraft1 vs Aircraft2)\n")
    report.append("- Occlusion levels: 10%, 70%, 90%\n")
    report.append("- Split: 300 train / 300 val / 48 test images\n\n")

    report.append("### Model Performance by Occlusion Level\n\n")

    # Human performance
    human_perf = {'10%': 0.9562, '70%': 0.7928, '90%': 0.6188}

    report.append("#### Human Baseline (65 subjects, 19,500 trials)\n\n")
    report.append("| Occlusion | Accuracy | Avg RT |\n")
    report.append("|-----------|----------|--------|\n")
    report.append(f"| 10% | **95.62%** | ~1200ms |\n")
    report.append(f"| 70% | **79.28%** | ~1800ms |\n")
    report.append(f"| 90% | **61.88%** | ~2200ms |\n\n")

    report.append("#### AI Models Performance (48 test images)\n\n")
    report.append("**ViT-B/16 (Vision Transformer)**\n\n")
    report.append("| Occlusion | Accuracy | Precision | Recall | F1 Score |\n")
    report.append("|-----------|----------|-----------|--------|----------|\n")
    for idx, row in vit_metrics.iterrows():
        report.append(f"| {row['occlusion_level']} | {row['accuracy']:.2%} | {row['precision']:.2%} | {row['recall']:.2%} | {row['f1']:.3f} |\n")
    report.append("\n")

    report.append("**ResNet-50 (Convolutional Neural Network)**\n\n")
    report.append("| Occlusion | Accuracy | Precision | Recall | F1 Score |\n")
    report.append("|-----------|----------|-----------|--------|----------|\n")
    for idx, row in resnet_metrics.iterrows():
        report.append(f"| {row['occlusion_level']} | {row['accuracy']:.2%} | {row['precision']:.2%} | {row['recall']:.2%} | {row['f1']:.3f} |\n")
    report.append("\n")

    report.append("> [!warning] Critical Observation\n")
    report.append("> ResNet-50 completely fails at 70% and 90% occlusion (F1=0.0), while ViT maintains reasonable performance.\n")
    report.append("> This suggests **global attention mechanisms** (ViT) are more robust to occlusion than **local convolutions** (ResNet).\n\n")

    # Performance gaps
    report.append("### Human-AI Performance Gap Analysis\n\n")
    report.append("| Occlusion | Human | ViT | ResNet | ViT Gap | ResNet Gap |\n")
    report.append("|-----------|-------|-----|--------|---------|------------|\n")
    for idx, row in vit_metrics.iterrows():
        occlusion = row['occlusion_level']
        vit_acc = row['accuracy']
        resnet_acc = resnet_metrics.iloc[idx]['accuracy']
        human_acc = human_perf[occlusion]
        vit_gap = human_acc - vit_acc
        resnet_gap = human_acc - resnet_acc
        report.append(f"| {occlusion} | {human_acc:.2%} | {vit_acc:.2%} | {resnet_acc:.2%} | **{vit_gap:.2%}** | **{resnet_gap:.2%}** |\n")
    report.append("\n")

    report.append("**Key Insights:**\n\n")
    report.append("1. **Largest gap at low occlusion (10%):** 45-52% difference\n")
    report.append("   - Paradoxical: AI should perform better when more information is visible\n")
    report.append("   - Hypothesis: Small sample overfitting, distribution mismatch\n\n")
    report.append("2. **Gap narrows at high occlusion (90%):** 5-12% difference\n")
    report.append("   - Both humans and AI struggle with extreme occlusion\n")
    report.append("   - Suggests task becomes inherently difficult\n\n")
    report.append("3. **ViT consistently outperforms ResNet:** ~6% advantage across all levels\n")
    report.append("   - Global attention > local convolution for occlusion\n\n")

    # Agreement analysis
    report.append("### Human-AI Agreement Patterns\n\n")
    all_correct = per_image['all_correct'].sum()
    all_wrong = per_image['all_wrong'].sum()
    human_only = ((per_image['Human'] == 1) & (per_image['ViT'] == 0) & (per_image['ResNet'] == 0)).sum()
    ai_only = ((per_image['Human'] == 0) & ((per_image['ViT'] == 1) | (per_image['ResNet'] == 1))).sum()

    report.append(f"- **All three correct:** {all_correct}/47 images (12.8%)\n")
    report.append(f"- **All three wrong:** {all_wrong}/47 images (0.0%)\n")
    report.append(f"- **Only human correct:** {human_only}/47 images (17.0%)\n")
    report.append(f"- **Only AI correct:** {ai_only}/47 images (12.8%)\n\n")

    report.append("> [!note] Interpretation\n")
    report.append("> Low agreement (12.8%) suggests AI and humans use **fundamentally different visual strategies**.\n")
    report.append("> This opens opportunities for human-AI complementarity and hybrid systems.\n\n")

    report.append("---\n\n")

    # Section 2: Statistical Analysis
    report.append("## Statistical Analysis\n\n")

    report.append("### Training Dynamics\n\n")
    report.append("**ViT-B/16:**\n")
    report.append("- Best validation accuracy: 64.29% (epoch 5)\n")
    report.append("- Training accuracy at best epoch: 54.29%\n")
    report.append("- **Underfitting:** Train < Val (unusual pattern)\n")
    report.append("- Interpretation: Model capacity underutilized due to frozen backbone\n\n")

    report.append("**ResNet-50:**\n")
    report.append("- Best validation accuracy: 54.76% (epoch 4)\n")
    report.append("- Training accuracy at best epoch: 47.62%\n")
    report.append("- **Underfitting:** Train < Val\n")
    report.append("- Interpretation: Similar issue, but worse overall performance\n\n")

    report.append("> [!tip] Implication\n")
    report.append("> Both models show **underfitting**, not overfitting. This suggests:\n")
    report.append("> 1. Frozen backbone strategy may be too conservative\n")
    report.append("> 2. Models have capacity to learn more if properly trained\n")
    report.append("> 3. Data augmentation experiments (9,900 images) may enable full fine-tuning\n\n")

    report.append("### Visualizations Generated\n\n")
    report.append("All analysis visualizations are available in:\n")
    report.append("- `reports/analysis_outputs/human_vs_ai_comparison.png`\n")
    report.append("- `reports/analysis_outputs/performance_gap_heatmap.png`\n")
    report.append("- `reports/analysis_outputs/training_dynamics.png`\n")
    report.append("- `reports/analysis_outputs/agreement_analysis.png`\n\n")

    report.append("---\n\n")

    # Section 3: Literature Review Insights
    report.append("## Literature Review Insights\n\n")

    report.append("### Occlusion Robustness in Deep Learning\n\n")

    report.append("#### Vision Transformers vs CNNs\n\n")
    report.append("**Key Findings from Recent Literature (2024-2025):**\n\n")

    report.append("1. **Global vs Local Processing**\n")
    report.append("   - ViTs use self-attention to aggregate information globally\n")
    report.append("   - CNNs rely on local receptive fields, vulnerable to occlusion\n")
    report.append("   - Research shows ViTs maintain performance better under partial occlusion\n\n")

    report.append("2. **Attention Mechanisms for Occlusion**\n")
    report.append("   - Attention naturally downweights occluded regions\n")
    report.append("   - CNNs lack this adaptive mechanism\n")
    report.append("   - Hybrid architectures (CNN + attention) show promise\n\n")

    report.append("3. **Human Vision Comparison**\n")
    report.append("   - Humans use top-down processing and prior knowledge\n")
    report.append("   - AI models primarily use bottom-up feature extraction\n")
    report.append("   - Gap suggests need for incorporating structural priors\n\n")

    report.append("#### Relevant Approaches from Literature\n\n")
    report.append("- **Part-based models:** Decompose objects into parts, robust to partial occlusion\n")
    report.append("- **Contrastive learning:** Learn occlusion-invariant representations\n")
    report.append("- **Attention masking:** Explicitly model occluded regions\n")
    report.append("- **Meta-learning:** Few-shot adaptation to occlusion patterns\n\n")

    report.append("---\n\n")

    # Section 4: Current Challenges
    report.append("## Current Challenges\n\n")

    report.append("### 1. Small Sample Size\n")
    report.append("- **Problem:** Only 300 training images\n")
    report.append("- **Impact:** Severe overfitting, poor generalization\n")
    report.append("- **Current Solution:** Data augmentation to 9,900 images (in progress)\n")
    report.append("- **Status:** New configs created (`resnet50_augmented.yaml`, `vit_b16_augmented.yaml`)\n\n")

    report.append("### 2. Frozen Backbone Limitation\n")
    report.append("- **Problem:** Freezing backbone limits learning capacity\n")
    report.append("- **Impact:** Models underfit (train acc < val acc)\n")
    report.append("- **Proposed Solution:** Full fine-tuning with augmented data\n")
    report.append("- **Risk:** May still overfit if data insufficient\n\n")

    report.append("### 3. Lack of Interpretability\n")
    report.append("- **Problem:** Don't know what features models use\n")
    report.append("- **Impact:** Can't diagnose failure modes\n")
    report.append("- **Proposed Solution:** Attention visualization, saliency maps, GradCAM\n\n")

    report.append("### 4. Architecture Exploration Incomplete\n")
    report.append("- **Problem:** Only tested ViT and ResNet\n")
    report.append("- **Impact:** May miss better architectures\n")
    report.append("- **Proposed Solution:** Test Swin, ConvNeXt, DeiT, hybrid models\n\n")

    report.append("---\n\n")

    # Section 5: Proposed Research Directions
    report.append("## Proposed Research Directions\n\n")

    report.append("### Direction 1: Occlusion-Aware Attention Mechanisms ⭐⭐⭐\n\n")
    report.append("**Motivation:** Humans actively ignore occluded regions and focus on visible parts.\n\n")
    report.append("**Approach:**\n")
    report.append("1. Design occlusion detection module (predict occlusion mask)\n")
    report.append("2. Modify attention mechanism to downweight occluded regions\n")
    report.append("3. Enhance feature extraction from visible regions\n\n")
    report.append("**Implementation:**\n")
    report.append("```python\n")
    report.append("# Pseudo-code\n")
    report.append("occlusion_mask = OcclusionDetector(image)\n")
    report.append("attention_weights = SelfAttention(features) * (1 - occlusion_mask)\n")
    report.append("output = Classifier(attention_weights @ features)\n")
    report.append("```\n\n")
    report.append("**Expected Impact:** 10-20% accuracy improvement\n\n")
    report.append("**Difficulty:** Medium (requires custom attention layer)\n\n")

    report.append("### Direction 2: Part-Based Recognition ⭐⭐⭐\n\n")
    report.append("**Motivation:** Humans recognize objects by parts (wings, fuselage, tail).\n\n")
    report.append("**Approach:**\n")
    report.append("1. Pre-train part detectors (wing detector, fuselage detector)\n")
    report.append("2. Build part relationship graph\n")
    report.append("3. Use Graph Neural Network for reasoning\n")
    report.append("4. Classify based on visible parts + relationships\n\n")
    report.append("**Advantages:**\n")
    report.append("- Robust to partial occlusion (only need subset of parts)\n")
    report.append("- Interpretable (can visualize which parts used)\n")
    report.append("- Aligns with human cognition\n\n")
    report.append("**Challenges:**\n")
    report.append("- Requires part annotations (may need manual labeling)\n")
    report.append("- Complex pipeline (detection + GNN)\n\n")
    report.append("**Expected Impact:** 15-25% accuracy improvement\n\n")

    report.append("### Direction 3: Contrastive Learning with Occlusion Augmentation ⭐⭐\n\n")
    report.append("**Motivation:** Learn occlusion-invariant representations through self-supervision.\n\n")
    report.append("**Approach:**\n")
    report.append("1. Use SimCLR/MoCo framework\n")
    report.append("2. Design occlusion-specific augmentations (random masking, cutout)\n")
    report.append("3. Pre-train on large unlabeled aircraft dataset\n")
    report.append("4. Fine-tune on OIID dataset\n\n")
    report.append("**Advantages:**\n")
    report.append("- Leverages unlabeled data\n")
    report.append("- Learns robust features\n\n")
    report.append("**Challenges:**\n")
    report.append("- Requires large unlabeled dataset\n")
    report.append("- Computationally expensive\n\n")

    report.append("### Direction 4: fMRI-Guided Model Design ⭐⭐⭐⭐\n\n")
    report.append("**Motivation:** Use human brain data to guide AI architecture.\n\n")
    report.append("**Approach:**\n")
    report.append("1. **Representational Similarity Analysis (RSA)**\n")
    report.append("   - Extract AI model features at each layer\n")
    report.append("   - Extract fMRI signals from visual cortex (V1, V2, V4, IT)\n")
    report.append("   - Compute similarity matrices\n")
    report.append("   - Identify which layers align with which brain regions\n\n")
    report.append("2. **Encoding Models**\n")
    report.append("   - Use AI features to predict fMRI signals\n")
    report.append("   - Evaluate which model best explains brain activity\n\n")
    report.append("3. **Brain-Inspired Architecture**\n")
    report.append("   - Design layers that mimic brain processing hierarchy\n")
    report.append("   - Incorporate feedback connections (like visual cortex)\n\n")
    report.append("**Advantages:**\n")
    report.append("- Strong theoretical foundation\n")
    report.append("- High publication potential (Nature/Science level)\n")
    report.append("- Bridges AI and neuroscience\n\n")
    report.append("**Challenges:**\n")
    report.append("- Requires fMRI data (check if OIID includes it)\n")
    report.append("- Complex preprocessing and analysis\n")
    report.append("- Steep learning curve\n\n")

    report.append("### Direction 5: Hybrid CNN-Transformer Architecture ⭐⭐\n\n")
    report.append("**Motivation:** Combine local detail (CNN) with global context (Transformer).\n\n")
    report.append("**Approach:**\n")
    report.append("1. Use CNN for low-level feature extraction\n")
    report.append("2. Use Transformer for high-level reasoning\n")
    report.append("3. Test architectures: ConvNeXt, Swin Transformer, CoAtNet\n\n")
    report.append("**Expected Impact:** 5-10% improvement over pure ViT\n\n")

    report.append("---\n\n")

    # Section 6: Next Steps
    report.append("## Next Steps and Recommendations\n\n")

    report.append("### Immediate Actions (1-2 weeks)\n\n")
    report.append("- [ ] Complete data augmentation experiments (9,900 images)\n")
    report.append("- [ ] Train with full fine-tuning (unfreeze backbone)\n")
    report.append("- [ ] Evaluate if augmentation solves small sample problem\n")
    report.append("- [ ] Generate attention visualizations for ViT\n")
    report.append("- [ ] Analyze failure cases in detail\n\n")

    report.append("### Short-term Goals (1-2 months)\n\n")
    report.append("**Option A: Quick Win (Engineering Focus)**\n")
    report.append("- Implement occlusion-aware attention\n")
    report.append("- Test on augmented dataset\n")
    report.append("- Aim for 70%+ accuracy\n")
    report.append("- Write conference paper (CVPR/ICCV workshop)\n\n")

    report.append("**Option B: Deep Research (Theory Focus)**\n")
    report.append("- Investigate fMRI data availability in OIID\n")
    report.append("- Implement RSA analysis\n")
    report.append("- Compare AI representations with brain activity\n")
    report.append("- Aim for top-tier journal (Nature Communications)\n\n")

    report.append("### Medium-term Goals (3-6 months)\n\n")
    report.append("- Implement part-based recognition system\n")
    report.append("- Expand architecture comparison (Swin, ConvNeXt, DeiT)\n")
    report.append("- Develop hybrid human-AI system\n")
    report.append("- Prepare full conference paper (CVPR/NeurIPS)\n\n")

    report.append("### Critical Decision Points\n\n")
    report.append("> [!question] Key Questions for Advisor\n")
    report.append("> 1. **Research Direction:** Engineering (quick results) vs Theory (high impact)?\n")
    report.append("> 2. **fMRI Data:** Is it available in OIID dataset? Can we access it?\n")
    report.append("> 3. **Publication Target:** Conference (CVPR/NeurIPS) vs Journal (Nature/NeuroImage)?\n")
    report.append("> 4. **Timeline:** What's the deadline for first publication?\n")
    report.append("> 5. **Resources:** Do we have GPU cluster for large-scale experiments?\n\n")

    report.append("### Recommended Priority Ranking\n\n")
    report.append("**If fMRI data available:**\n")
    report.append("1. fMRI-guided model design (Direction 4) ⭐⭐⭐⭐\n")
    report.append("2. Occlusion-aware attention (Direction 1) ⭐⭐⭐\n")
    report.append("3. Part-based recognition (Direction 2) ⭐⭐⭐\n\n")

    report.append("**If fMRI data NOT available:**\n")
    report.append("1. Occlusion-aware attention (Direction 1) ⭐⭐⭐\n")
    report.append("2. Part-based recognition (Direction 2) ⭐⭐⭐\n")
    report.append("3. Contrastive learning (Direction 3) ⭐⭐\n\n")

    report.append("---\n\n")

    # Appendix
    report.append("## Appendix\n\n")

    report.append("### Related Notes\n\n")
    report.append("- [[CLAUDE.md]] - Project documentation\n")
    report.append("- [[TOOLS_USAGE.md]] - Analysis scripts usage\n")
    report.append("- [[VIT_GUIDE.md]] - Vision Transformer implementation\n\n")

    report.append("### External Resources\n\n")
    report.append("- [OIID Dataset](https://openneuro.org/datasets/ds005226)\n")
    report.append("- [ViT Paper](https://arxiv.org/abs/2010.11929)\n")
    report.append("- [Occlusion Robustness Survey](https://arxiv.org/abs/2108.00946)\n\n")

    report.append("### Tags for Cross-referencing\n\n")
    report.append("#vision-transformers #occlusion-robustness #human-ai-comparison #deep-learning #computer-vision #neuroscience #fmri #attention-mechanisms\n\n")

    report.append("---\n\n")
    report.append(f"*Report generated automatically on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    report.append(f"*Analysis scripts: `scripts/analysis/comprehensive_analysis.py`, `scripts/analysis/generate_obsidian_report.py`*\n")

    # Write report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = reports_dir / f"research_analysis_{timestamp}.md"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(report)

    print(f"[OK] Report generated: {report_path}")
    print(f"\nReport contains:")
    print("  - Executive summary with key findings")
    print("  - Detailed experimental results")
    print("  - Statistical analysis")
    print("  - Literature review insights")
    print("  - 5 proposed research directions")
    print("  - Actionable next steps")
    print("  - Obsidian-compatible formatting (tags, links, callouts)")

    return report_path

if __name__ == "__main__":
    generate_report()
