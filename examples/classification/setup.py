#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MMPretrain图像分类示例安装脚本

此脚本帮助用户安装所需的依赖项
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def install_basic_dependencies():
    """安装基础依赖"""
    print("=" * 50)
    print("正在安装基础依赖...")
    print("=" * 50)
    
    # 基础依赖列表
    basic_deps = [
        "torch>=1.8.0",
        "torchvision>=0.9.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "pillow>=8.0.0",
        "opencv-python>=4.5.0",
        "openmim>=0.3.7"
    ]
    
    # 安装基础依赖
    for dep in basic_deps:
        print(f"安装 {dep}...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True)
        except subprocess.CalledProcessError as e:
            print(f"安装 {dep} 失败: {e}")
            print(f"请尝试手动安装: pip install {dep}")
    
    print("\n基础依赖安装完成！\n")

def install_mmpretrain():
    """安装MMPretrain及其依赖"""
    print("=" * 50)
    print("正在安装MMPretrain及其依赖...")
    print("=" * 50)
    
    # MMPretrain依赖列表
    mm_deps = [
        "mmengine>=0.7.0",
        "mmcv>=2.0.0",
        "mmpretrain>=1.0.0"
    ]
    
    # 安装MMPretrain依赖
    for dep in mm_deps:
        print(f"安装 {dep}...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True)
        except subprocess.CalledProcessError as e:
            print(f"安装 {dep} 失败: {e}")
            print(f"请尝试手动安装: pip install {dep}")
    
    print("\nMMPretrain及其依赖安装完成！\n")

def install_optional_dependencies():
    """安装可选依赖"""
    print("=" * 50)
    print("正在安装可选依赖...")
    print("=" * 50)
    
    # 尝试安装grad-cam
    print("尝试安装grad-cam (用于类激活图可视化)...")
    
    # 方法1: 尝试安装pytorch-grad-cam
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "pytorch-grad-cam>=1.3.7"], check=True)
        print("pytorch-grad-cam安装成功！")
        return
    except subprocess.CalledProcessError:
        print("pytorch-grad-cam安装失败，尝试替代方案...")
    
    # 方法2: 尝试安装grad-cam
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "grad-cam>=1.4.0"], check=True)
        print("grad-cam安装成功！")
        return
    except subprocess.CalledProcessError:
        print("grad-cam安装失败，尝试从GitHub安装...")
    
    # 方法3: 从GitHub安装
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "git+https://github.com/jacobgil/pytorch-grad-cam.git"], check=True)
        print("从GitHub安装grad-cam成功！")
    except subprocess.CalledProcessError as e:
        print(f"所有grad-cam安装方法都失败: {e}")
        print("您仍然可以使用其他功能，但类激活图可视化将不可用")
    
    print("\n可选依赖安装完成！\n")

def main():
    """主函数"""
    print("=" * 50)
    print("MMPretrain图像分类示例安装脚本")
    print("=" * 50)
    
    # 检查Python版本
    python_version = platform.python_version()
    print(f"Python版本: {python_version}")
    
    # 询问用户是否继续
    choice = input("是否继续安装依赖? (y/n): ").strip().lower()
    
    if choice != 'y':
        print("安装已取消")
        return
    
    # 安装依赖
    install_basic_dependencies()
    install_mmpretrain()
    install_optional_dependencies()
    
    print("=" * 50)
    print("安装完成！")
    print("您现在可以运行示例脚本:")
    print("1. python demo.py - 简单演示")
    print("2. python image_classification_inference.py - 完整推理示例")
    print("3. python advanced_inference.py - 高级推理示例")
    print("=" * 50)

if __name__ == "__main__":
    main() 