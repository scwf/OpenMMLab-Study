# MMPretrain 图像分类推理实现指南

本文档详细介绍了如何使用MMPretrain实现图像分类推理功能，包括环境配置、代码实现和使用说明。

## 目录

- [环境准备](#环境准备)
  - [检查CUDA版本](#检查cuda版本)
  - [安装PyTorch](#安装pytorch)
  - [安装MMPretrain](#安装mmpretrain)
- [代码实现](#代码实现)
  - [基础推理示例](#基础推理示例)
  - [演示脚本](#演示脚本)
  - [高级推理示例](#高级推理示例)
  - [工具脚本](#工具脚本)
- [使用说明](#使用说明)
  - [运行演示](#运行演示)
  - [使用自己的图像](#使用自己的图像)
  - [使用不同的模型](#使用不同的模型)
- [常见问题](#常见问题)
- [参考资料](#参考资料)

## 环境准备

### 检查CUDA版本

在安装PyTorch之前，首先需要确定您的系统上安装的CUDA版本。

#### 方法1：使用nvidia-smi命令

在Windows系统中，打开PowerShell或命令提示符，输入：

```powershell
nvidia-smi
```

您将看到类似以下的输出：

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.05             Driver Version: 535.104.05                |
| CUDA Version: 12.2                                                          |
+-----------------------------------------------------------------------------+
```

记下显示的CUDA版本（在这个例子中是12.2）。

#### 方法2：使用check_pytorch.py工具

我们提供了一个工具脚本来检查CUDA版本和PyTorch配置：

```bash
python examples/classification/check_pytorch.py
```

### 安装PyTorch

根据您的CUDA版本，安装对应的PyTorch版本。我们提供了两种方式：

#### 自动安装（推荐）

运行以下命令，脚本会自动检测您的CUDA版本并安装对应的PyTorch：

```bash
python examples/classification/install_pytorch.py
```

#### 手动安装

根据您的CUDA版本，使用以下命令安装PyTorch：

- 对于CUDA 12.1/12.2:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```

- 对于CUDA 11.8:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

- 对于CUDA 11.7:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
  ```

- 对于CUDA 11.6:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116
  ```

- 对于CPU版本（无CUDA）:
  ```bash
  pip install torch torchvision torchaudio
  ```

### 安装MMPretrain

安装MMPretrain及其依赖：

```bash
pip install -r examples/classification/requirements.txt
```

## 代码实现

我们实现了多个脚本来展示MMPretrain的图像分类推理功能：

### 基础推理示例

`image_classification_inference.py` 展示了多种推理方式：

1. **使用inference_model快速推理**：
   ```python
   from mmpretrain import inference_model
   result = inference_model('resnet50_8xb32_in1k', 'path/to/image.jpg')
   print(f"预测类别: {result['pred_class']}")
   ```

2. **使用ImageClassificationInferencer进行批量推理**：
   ```python
   from mmpretrain import ImageClassificationInferencer
   inferencer = ImageClassificationInferencer('resnet50_8xb32_in1k')
   results = inferencer(['image1.jpg', 'image2.jpg'])
   ```

3. **使用get_model获取模型并手动推理**：
   ```python
   from mmpretrain import get_model
   model = get_model('resnet50_8xb32_in1k', pretrained=True)
   # 手动处理输入和推理
   ```

4. **使用FeatureExtractor提取特征**：
   ```python
   from mmpretrain import get_model, FeatureExtractor
   model = get_model('resnet50_8xb32_in1k', backbone=dict(out_indices=(0, 1, 2, 3)))
   extractor = FeatureExtractor(model)
   features = extractor('image.jpg')[0]
   ```

### 演示脚本

`demo.py` 是一个简单的演示脚本，可以下载示例图像并运行推理：

```python
# 下载示例图像
image_paths = download_images()

# 使用预训练模型进行推理
model_name = 'resnet50_8xb32_in1k'
for name, img_path in image_paths.items():
    result = inference_model(model_name, img_path)
    print(f"预测类别: {result['pred_class']}")
```

### 高级推理示例

`advanced_inference.py` 展示了高级推理功能：

1. **使用自定义配置**：
   ```python
   model_cfg = dict(
       type='ImageClassifier',
       backbone=dict(
           type='ResNet',
           depth=18,
           num_stages=4,
           out_indices=(3,),
           style='pytorch'),
       neck=dict(type='GlobalAveragePooling'),
       head=dict(
           type='LinearClsHead',
           num_classes=1000,
           in_channels=512,
           loss=dict(type='CrossEntropyLoss'),
           topk=(1, 5),
       )
   )
   
   inferencer = ImageClassificationInferencer(
       model=model_cfg,
       pretrained='https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth'
   )
   ```

2. **自定义预处理**：
   ```python
   def preprocess_image(img_path, size=224):
       img = Image.open(img_path).convert('RGB')
       img = img.resize((size, size))
       img = np.array(img, dtype=np.float32)
       mean = np.array([123.675, 116.28, 103.53])
       std = np.array([58.395, 57.12, 57.375])
       img = (img - mean) / std
       img = np.transpose(img, (2, 0, 1))
       img = np.expand_dims(img, axis=0)
       return img
   ```

3. **可视化类激活图**：
   ```python
   from pytorch_grad_cam import GradCAM
   from pytorch_grad_cam.utils.image import show_cam_on_image
   
   # 创建GradCAM对象
   cam = GradCAM(model=model, target_layers=target_layers)
   
   # 生成CAM
   grayscale_cam = cam(input_tensor=input_tensor)
   cam_image = show_cam_on_image(img_normalized, grayscale_cam[0, :])
   ```

4. **批量推理并显示top-k结果**：
   ```python
   result = inferencer(image_path, topk=5)[0]
   for i, (cls_name, score) in enumerate(zip(result['pred_class'], result['pred_score'])):
       print(f"{i+1}. {cls_name}: {score:.4f}")
   ```

### 工具脚本

1. **check_pytorch.py**：检查PyTorch和CUDA版本的工具
   ```python
   import torch
   print(f"PyTorch版本: {torch.__version__}")
   print(f"CUDA是否可用: {torch.cuda.is_available()}")
   if torch.cuda.is_available():
       print(f"CUDA版本: {torch.version.cuda}")
   ```

2. **install_pytorch.py**：自动检测CUDA版本并安装对应的PyTorch
   ```python
   # 检测CUDA版本
   cuda_version = get_cuda_version()
   
   # 获取安装命令
   install_command = get_pytorch_install_command(cuda_version)
   
   # 执行安装
   subprocess.run(install_command, shell=True)
   ```

## 使用说明

### 运行演示

1. **简单演示**：
   ```bash
   python examples/classification/demo.py
   ```

2. **完整推理示例**：
   ```bash
   python examples/classification/image_classification_inference.py
   ```

3. **高级推理示例**：
   ```bash
   python examples/classification/advanced_inference.py
   ```

### 使用自己的图像

要使用自己的图像进行推理，只需修改脚本中的图像路径：

```python
# 在demo.py中
IMAGE_URLS = {
    "自定义图像": "path/to/your/image.jpg",
}

# 在其他脚本中
image_path = "path/to/your/image.jpg"
```

### 使用不同的模型

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
python examples/classification/check_pytorch.py
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

### 3. 模型下载问题

如果模型下载速度慢或失败，可以：
- 使用代理
- 手动下载模型并指定本地路径
- 使用镜像站点

## 参考资料

- [MMPretrain 文档](https://mmpretrain.readthedocs.io/)
- [MMPretrain GitHub](https://github.com/open-mmlab/mmpretrain)
- [PyTorch 安装指南](https://pytorch.org/get-started/locally/)
- [CUDA 下载页面](https://developer.nvidia.com/cuda-downloads) 