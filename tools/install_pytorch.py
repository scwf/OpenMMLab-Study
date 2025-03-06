#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PyTorch自动安装脚本

此脚本会检测系统的CUDA版本，并安装对应的PyTorch版本
"""

import os
import sys
import subprocess
import platform
import re
from pathlib import Path

def get_cuda_version():
    """获取系统CUDA版本"""
    cuda_version = None
    
    # 方法1: 使用nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            # 使用正则表达式查找CUDA版本
            match = re.search(r'CUDA Version: (\d+\.\d+)', result.stdout)
            if match:
                cuda_version = match.group(1)
                print(f"通过nvidia-smi检测到CUDA版本: {cuda_version}")
                return cuda_version
    except Exception as e:
        print(f"无法通过nvidia-smi获取CUDA版本: {e}")
    
    # 方法2: 使用nvcc
    try:
        result = subprocess.run(['nvcc', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            # 使用正则表达式查找CUDA版本
            match = re.search(r'release (\d+\.\d+)', result.stdout)
            if match:
                cuda_version = match.group(1)
                print(f"通过nvcc检测到CUDA版本: {cuda_version}")
                return cuda_version
    except Exception as e:
        print(f"无法通过nvcc获取CUDA版本: {e}")
    
    # 方法3: 检查环境变量
    if platform.system() == 'Windows':
        try:
            result = subprocess.run(['echo', '%CUDA_PATH%'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
            if 'CUDA' in result.stdout:
                # 使用正则表达式查找CUDA版本
                match = re.search(r'v(\d+\.\d+)', result.stdout)
                if match:
                    cuda_version = match.group(1)
                    print(f"通过环境变量检测到CUDA版本: {cuda_version}")
                    return cuda_version
        except Exception as e:
            print(f"无法通过环境变量获取CUDA版本: {e}")
    
    return cuda_version

def get_pytorch_install_command(cuda_version):
    """根据CUDA版本获取PyTorch安装命令"""
    if cuda_version is None:
        print("未检测到CUDA，将安装CPU版本的PyTorch")
        return "pip install torch torchvision torchaudio"
    
    # 解析CUDA版本
    try:
        major, minor = cuda_version.split('.')
        major, minor = int(major), int(minor)
    except:
        print(f"无法解析CUDA版本: {cuda_version}，将安装CPU版本的PyTorch")
        return "pip install torch torchvision torchaudio"
    
    # 根据CUDA版本选择PyTorch安装命令
    if major == 12:
        # CUDA 12.x 使用 cu121
        return "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    elif major == 11:
        if minor >= 8:
            # CUDA 11.8+ 使用 cu118
            return "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        elif minor == 7:
            # CUDA 11.7 使用 cu117
            return "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117"
        elif minor == 6:
            # CUDA 11.6 使用 cu116
            return "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116"
        elif minor >= 3:
            # CUDA 11.3-11.5 使用 cu113
            return "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113"
        else:
            # CUDA 11.0-11.2 使用 cu111
            return "pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html"
    elif major == 10:
        # CUDA 10.x 使用 cu102
        return "pip install torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html"
    else:
        print(f"不支持的CUDA版本: {cuda_version}，将安装CPU版本的PyTorch")
        return "pip install torch torchvision torchaudio"

def install_pytorch():
    """安装PyTorch"""
    print("=" * 50)
    print("PyTorch自动安装脚本")
    print("=" * 50)
    
    # 检测CUDA版本
    cuda_version = get_cuda_version()
    
    if cuda_version:
        print(f"检测到CUDA版本: {cuda_version}")
    else:
        print("未检测到CUDA，将安装CPU版本的PyTorch")
    
    # 获取安装命令
    install_command = get_pytorch_install_command(cuda_version)
    
    # 询问用户是否继续
    print("\n将执行以下命令安装PyTorch:")
    print(f"  {install_command}")
    
    choice = input("\n是否继续安装? (y/n): ").strip().lower()
    
    if choice == 'y':
        print("\n开始安装PyTorch...")
        try:
            subprocess.run(install_command, shell=True, check=True)
            print("\nPyTorch安装完成!")
            print("\n您可以运行 'python check_pytorch.py' 验证安装")
        except Exception as e:
            print(f"\n安装失败: {e}")
            print("请手动安装PyTorch，访问 https://pytorch.org/get-started/locally/")
    else:
        print("\n安装已取消")
        print("您可以访问 https://pytorch.org/get-started/locally/ 获取安装命令")

if __name__ == "__main__":
    install_pytorch() 