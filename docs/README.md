open mmlab 计算机视觉框架学习

本项目是学习open mmlab的样例工程，系统的展示了如何使用
1. MMSegmentation实现图像分割任务
2. MMDetection实现图像检测任务
3. MMPreTrainNEW实现图像分类任务

本项目支持图像分类、检测、分割的模型训练和推理，提供了即开即用的操作指导运行训练和推理任务

## 项目框架

```
mmengine-study/
├── docs/                  # 文档目录
│   ├── README.md          # 主README文件
│   └── mmpretrain-inference.md  # MMPretrain推理文档
├── examples/              # 示例代码目录
│   └── classification/    # 图像分类示例
│       ├── README.md      # 图像分类示例说明
│       ├── demo.py        # 简单演示脚本
│       ├── image_classification_inference.py  # 完整推理示例
│       ├── advanced_inference.py  # 高级推理示例
│       └── requirements.txt  # 依赖项列表
└── .venv/                 # 虚拟环境目录
```

## 功能模块

### 1. 图像分类 (MMPretrain)

MMPretrain是OpenMMLab的图像分类和预训练工具箱，提供了丰富的预训练模型和推理接口。

#### 快速开始

```bash
# 安装依赖
cd examples/classification
pip install -r requirements.txt

# 运行演示脚本
python demo.py
```

#### 主要功能

- **快速推理**：使用`inference_model`函数进行简单推理
- **批量推理**：使用`ImageClassificationInferencer`进行批量图像推理
- **特征提取**：使用`FeatureExtractor`从图像中提取特征
- **自定义配置**：支持自定义模型配置和预处理方式
- **可视化**：支持结果可视化和类激活图(CAM)可视化

详细使用说明请参考[图像分类示例](../examples/classification/README.md)。

### 2. 图像检测 (MMDetection)

待补充

### 3. 图像分割 (MMSegmentation)

待补充