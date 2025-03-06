# 根据CUDA版本安装PyTorch指南

本指南将帮助您根据本机的CUDA版本正确安装PyTorch，确保GPU加速功能正常工作。

## 目录

- [检查CUDA版本](#检查cuda版本)
- [确定兼容的PyTorch版本](#确定兼容的pytorch版本)
- [安装PyTorch](#安装pytorch)
- [验证安装](#验证安装)
- [常见问题解决](#常见问题解决)
- [附录：CUDA与PyTorch版本对应关系](#附录cuda与pytorch版本对应关系)

## 检查CUDA版本

在安装PyTorch之前，首先需要确定您的系统上安装的CUDA版本。

### 方法1：使用nvidia-smi命令

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

记下显示的CUDA最高版本（在这个例子中是12.2）
注意这个版本是是你本地驱动可支持的最高版本，而非实际版本，实际版本通过方法2和3来获取。

### 方法2：检查CUDA工具包版本

如果您已经安装了CUDA工具包，可以通过以下命令检查版本：

```powershell
nvcc --version
```

输出示例：

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Tue_Jun__6_00:20:08_Pacific_Daylight_Time_2023
Cuda compilation tools, release 12.2, V12.2.91
Build cuda_12.2.r12.2/compiler.32965470_0
```

### 方法3：检查环境变量

您也可以检查CUDA的安装路径：

```powershell
echo $env:CUDA_PATH
```

路径中通常包含版本号，例如：`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2`

## 确定兼容的PyTorch版本

PyTorch的不同版本支持不同的CUDA版本。以下是一些常见的对应关系：

| PyTorch版本 | 支持的CUDA版本 |
|------------|--------------|
| 2.2.x      | CUDA 11.8, 12.1 |
| 2.1.x      | CUDA 11.8, 12.1 |
| 2.0.x      | CUDA 11.7, 11.8 |
| 1.13.x     | CUDA 11.6, 11.7 |
| 1.12.x     | CUDA 11.3, 11.6 |
| 1.11.x     | CUDA 11.3 |
| 1.10.x     | CUDA 11.3 |
| 1.9.x      | CUDA 11.1 |
| 1.8.x      | CUDA 11.1 |

建议选择与您的CUDA版本兼容的最新PyTorch版本。

## 安装PyTorch

### 使用pip安装

根据您的CUDA版本，使用以下命令安装PyTorch：

#### 对于CUDA 12.1/12.2

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 对于CUDA 11.8

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 对于CUDA 11.7

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

#### 对于CUDA 11.6

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116
```

#### 对于CPU版本（无CUDA）

```powershell
pip install torch torchvision torchaudio
```

### 使用conda安装

如果您使用Anaconda或Miniconda，可以通过以下命令安装：

#### 对于CUDA 12.1

```powershell
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

#### 对于CUDA 11.8

```powershell
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### 对于CPU版本（无CUDA）

```powershell
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

## 验证安装

安装完成后，您可以通过以下Python代码验证PyTorch是否正确识别您的GPU：

```python
import torch

# 检查PyTorch版本
print(f"PyTorch版本: {torch.__version__}")

# 检查CUDA是否可用
print(f"CUDA是否可用: {torch.cuda.is_available()}")

# 如果CUDA可用，显示CUDA版本
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"可用GPU数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
```

将上述代码保存为`check_pytorch.py`，然后运行：

```powershell
python check_pytorch.py
```

如果一切正常，您应该能看到您的GPU信息和CUDA版本。

## 常见问题解决

### 问题1：安装后CUDA不可用

如果`torch.cuda.is_available()`返回`False`，可能的原因包括：

1. PyTorch版本与CUDA版本不匹配
2. NVIDIA驱动程序过旧
3. CUDA工具包安装不完整

解决方案：
- 更新NVIDIA驱动程序
- 重新安装与您的CUDA版本匹配的PyTorch版本
- 确保环境变量正确设置

### 问题2：导入PyTorch时出现DLL加载错误

这通常是因为找不到CUDA相关的DLL文件。

解决方案：
- 确保CUDA工具包正确安装
- 将CUDA的bin目录添加到系统PATH环境变量中
- 尝试重新安装PyTorch

### 问题3：内存不足错误

当运行大型模型时，可能会遇到GPU内存不足的问题。

解决方案：
- 减小批处理大小(batch size)
- 使用混合精度训练
- 实现梯度累积
- 使用模型并行或数据并行

## 附录：CUDA与PyTorch版本对应关系

下表详细列出了PyTorch各版本支持的CUDA版本：

| PyTorch版本 | 支持的CUDA版本 | 支持的Python版本 |
|------------|--------------|----------------|
| 2.2.0      | 11.8, 12.1   | >=3.8, <=3.11  |
| 2.1.0      | 11.8, 12.1   | >=3.8, <=3.11  |
| 2.0.0      | 11.7, 11.8   | >=3.8, <=3.11  |
| 1.13.1     | 11.6, 11.7   | >=3.7, <=3.10  |
| 1.12.1     | 11.3, 11.6   | >=3.7, <=3.10  |
| 1.11.0     | 11.3         | >=3.7, <=3.10  |
| 1.10.0     | 11.3         | >=3.6, <=3.9   |
| 1.9.0      | 11.1         | >=3.6, <=3.9   |
| 1.8.0      | 11.1         | >=3.6, <=3.9   |
| 1.7.0      | 10.2, 11.0   | >=3.6, <=3.9   |
| 1.6.0      | 10.2         | >=3.6, <=3.8   |
| 1.5.0      | 10.2         | >=3.5, <=3.8   |
| 1.4.0      | 10.1         | >=3.5, <=3.8   |

请注意，PyTorch的最新版本通常会支持最新的CUDA版本，但可能需要一段时间才能提供官方支持。

## 参考资源

- [PyTorch官方安装指南](https://pytorch.org/get-started/locally/)
- [NVIDIA CUDA下载页面](https://developer.nvidia.com/cuda-downloads)
- [PyTorch版本发布说明](https://github.com/pytorch/pytorch/releases)