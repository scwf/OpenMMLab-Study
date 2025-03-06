# MMPretrain 图像分类推理示例

本目录包含使用 MMPretrain 进行图像分类推理的示例代码。

## 目录结构

- `inference_simple.py`: 简单的演示脚本，下载示例图像并运行推理
- `inference_normal.py`: 完整的图像分类推理示例，包含多种推理方式
- `inference_advance.py`: 高级推理示例，包含自定义配置、预处理和可视化
- `requirements.txt`: 依赖项列表
- `check_pytorch.py`: 检查PyTorch和CUDA版本的工具
- `install_pytorch.py`: 自动检测CUDA版本并安装对应的PyTorch
- `setup.py`: 简化版安装脚本，帮助安装所有依赖
- `download_data.py`: 下载图像分类数据集的工具脚本

## 安装依赖

### 0. 使用简化安装脚本（推荐）

我们提供了一个简化的安装脚本，可以帮助您一键安装所有依赖：

```bash
python setup.py
```

这个脚本会自动安装：
- 基础依赖（torch, numpy等）
- MMPretrain及其依赖
- 可选依赖（grad-cam等）

如果您遇到安装问题，可以尝试以下手动安装方法。

### 1. 安装PyTorch

首先，您需要安装与您的CUDA版本兼容的PyTorch。我们提供了两种方式：

#### 自动安装（推荐）

运行以下命令，脚本会自动检测您的CUDA版本并安装对应的PyTorch：

```bash
python install_pytorch.py
```

#### 手动安装

如果您想手动安装，可以先检查您的CUDA版本：

```bash
python check_pytorch.py
```

然后根据输出的建议安装命令进行安装。

### 2. 安装其他依赖

安装完PyTorch后，安装其他依赖：

```bash
# 创建虚拟环境（可选）
python -m venv venv
# Windows激活虚拟环境
.\venv\Scripts\activate
# Linux/Mac激活虚拟环境
# source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 3. 安装可选依赖

如果在安装 `pytorch-grad-cam` 时遇到问题，可以尝试以下替代方案：

```bash
# 方法1: 尝试安装grad-cam
pip install grad-cam>=1.4.0

# 方法2: 从GitHub安装
pip install git+https://github.com/jacobgil/pytorch-grad-cam.git
```

## 下载数据集

我们提供了一个方便的脚本来下载常用的图像分类数据集：

```bash
# 查看帮助信息
python download_data.py --help

# 列出所有可用的数据集
python download_data.py --list

# 下载CIFAR10测试集（默认行为）
python download_data.py

# 下载CIFAR100训练集
python download_data.py --dataset cifar100 --split train

# 下载ImageNet验证集的10%样本
python download_data.py --dataset imagenet --split val --sample-ratio 0.1

# 指定数据保存位置
python download_data.py --data-root ./custom_data_path
```

支持的数据集包括：
- ImageNet: 1000类图像分类标准数据集
- CIFAR10: 10类小型彩色图像数据集
- CIFAR100: 100类小型彩色图像数据集

## 运行示例

### 1. 简单演示

```bash
# 默认运行（仅使用示例图像）
python inference_simple.py

# 使用CIFAR10数据集
python inference_simple.py --dataset cifar10

# 使用Tiny-ImageNet数据集
python inference_simple.py --dataset tiny-imagenet

# 运行所有数据集的推理
python inference_simple.py --dataset all

# 指定图像数量
python inference_simple.py --dataset cifar10 --num-images 10

# 使用Tiny-ImageNet的训练集而不是验证集
python inference_simple.py --dataset tiny-imagenet --split train
```

这个脚本会下载示例图像并使用预训练的ResNet-50模型进行图像分类推理。
对于CIFAR10数据集，会使用ResNet-18模型；对于Tiny-ImageNet数据集，也会使用ResNet-18模型。

### 2. 完整推理示例

```bash
python inference_normal.py
```

这个脚本展示了多种推理方式，包括：
- 使用`inference_model`快速推理
- 使用`ImageClassificationInferencer`进行批量推理
- 使用`get_model`获取模型并手动推理
- 使用`FeatureExtractor`提取特征

### 3. 高级推理示例

```bash
python inference_advance.py
```

这个脚本展示了高级推理功能，包括：
- 使用自定义配置文件
- 使用不同的预处理方式
- 使用不同的后处理方式
- 可视化类激活图 (CAM)

## 使用自己的图像

要使用自己的图像进行推理，只需修改脚本中的图像路径：

```python
# 在inference_simple.py中
IMAGE_URLS = {
    "自定义图像": "path/to/your/image.jpg",
}

# 在其他脚本中
image_path = "path/to/your/image.jpg"
```

## 使用不同的模型

MMPretrain提供了多种预训练模型，可以通过以下方式查看可用模型：

```python
from mmpretrain import ImageClassificationInferencer
models = ImageClassificationInferencer.list_models()
print(models)
```
要使用不同的模型，只需修改脚本中的模型名称：

```python
# 例如使用ResNeXt-101模型
model_name = 'resnext101_32x4d_b32x8_imagenet'
```

## 常见问题

### 1. CUDA相关问题

如果遇到CUDA相关问题，请先运行以下命令检查您的PyTorch和CUDA配置：

```bash
python check_pytorch.py
```

常见问题包括：
- PyTorch版本与CUDA版本不匹配
- NVIDIA驱动程序过旧
- 安装了CPU版本的PyTorch

### 2. 内存不足错误

当运行大型模型时，可能会遇到GPU内存不足的问题。解决方案：
- 减小批处理大小(batch_size)
- 使用较小的模型
- 使用CPU进行推理（设置`device='cpu'`）

### 3. 依赖安装问题

如果遇到依赖安装问题：
- 尝试使用 `setup.py` 脚本进行安装
- 检查您的Python版本是否兼容（建议使用Python 3.8-3.10）
- 对于特定包的安装问题，可以查看相应包的GitHub页面获取最新安装指南

## 参考资料

- [MMPretrain 文档](https://mmpretrain.readthedocs.io/)
- [MMPretrain GitHub](https://github.com/open-mmlab/mmpretrain)
- [PyTorch 安装指南](https://pytorch.org/get-started/locally/) 

