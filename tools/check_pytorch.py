#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PyTorch和CUDA版本检查工具

此脚本用于检查PyTorch安装情况和CUDA可用性
"""

import torch
import platform
import sys

def check_pytorch_cuda():
    """检查PyTorch和CUDA版本信息"""
    
    print("=" * 50)
    print("系统信息:")
    print(f"操作系统: {platform.system()} {platform.version()}")
    print(f"Python版本: {platform.python_version()}")
    print("=" * 50)
    
    print("PyTorch信息:")
    print(f"PyTorch版本: {torch.__version__}")
    
    # 检查CUDA是否可用
    cuda_available = torch.cuda.is_available()
    print(f"CUDA是否可用: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"cuDNN版本: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else '不可用'}")
        print(f"可用GPU数量: {torch.cuda.device_count()}")
        
        # 显示所有GPU信息
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            
            # 获取GPU内存信息
            try:
                total_memory = torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024
                print(f"  总内存: {total_memory:.2f} GB")
            except:
                print("  无法获取内存信息")
    else:
        print("CUDA不可用，PyTorch将使用CPU运行")
        print("可能的原因:")
        print("1. 未安装NVIDIA GPU")
        print("2. NVIDIA驱动未正确安装")
        print("3. CUDA版本与PyTorch不兼容")
        print("4. 安装了CPU版本的PyTorch")
    
    print("=" * 50)
    print("PyTorch安装建议:")
    
    if cuda_available:
        print(f"您的PyTorch已正确配置CUDA {torch.version.cuda}")
        print("可以使用以下命令安装与当前CUDA版本匹配的PyTorch:")
        cuda_version = torch.version.cuda.split('.')
        major, minor = cuda_version[0], cuda_version[1]
        
        if int(major) == 12:
            print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        elif int(major) == 11 and int(minor) >= 8:
            print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        elif int(major) == 11 and int(minor) == 7:
            print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117")
        elif int(major) == 11 and int(minor) == 6:
            print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116")
        else:
            print(f"对于CUDA {torch.version.cuda}，请参考PyTorch官方网站获取安装命令")
    else:
        print("要安装支持CUDA的PyTorch，请先确认您的NVIDIA驱动和CUDA版本")
        print("然后访问 https://pytorch.org/get-started/locally/ 获取适合您系统的安装命令")
    
    print("=" * 50)

if __name__ == "__main__":
    check_pytorch_cuda() 