#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MMPretrain图像分类数据集下载脚本

本脚本用于下载和准备图像分类数据集，支持多种常用数据集
"""

import os
import argparse
import shutil
import zipfile
from pathlib import Path
from mmengine.registry import init_default_scope
from mmpretrain.datasets import build_dataset

def download_imagenet(data_root='data/imagenet', split='val', sample_ratio=0.1):
    """下载ImageNet数据集的一部分样本用于测试
    
    Args:
        data_root (str): 数据集保存路径
        split (str): 数据集分割，'train'或'val'
        sample_ratio (float): 采样比例，默认0.1表示下载10%的数据
    
    Returns:
        dataset: 构建好的数据集对象
    """
    print(f"正在准备ImageNet数据集 ({split})...")
    print(f"数据将保存到: {data_root}")
    
    # 确保目录存在
    os.makedirs(data_root, exist_ok=True)
    
    # ImageNet需要特定的目录结构
    # 检查是否已经有正确的目录结构
    split_dir = os.path.join(data_root, split)
    os.makedirs(split_dir, exist_ok=True)
    
    # 检查是否有类别子目录
    class_dirs = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
    
    if not class_dirs:
        print("警告: ImageNet数据集需要特定的目录结构")
        print("每个类别需要一个子目录，例如:")
        print(f"{split_dir}/n01440764/xxx.JPEG")
        print(f"{split_dir}/n01443537/xxx.JPEG")
        print("...")
        print("\n由于MMPretrain无法自动下载完整的ImageNet数据集，您需要:")
        print("1. 从官方网站下载ImageNet数据集: https://image-net.org/")
        print("2. 解压缩并按照上述结构组织文件")
        print("3. 或者使用其他数据集如CIFAR10或Tiny-ImageNet进行测试")
        print("   python download_data.py --dataset tiny-imagenet")
        
        # 创建一个示例类别目录，以便代码可以继续运行
        print("\n创建示例类别目录以便测试...")
        sample_class_dir = os.path.join(split_dir, "n01440764")
        os.makedirs(sample_class_dir, exist_ok=True)
        
        # 创建一个示例图像文件
        try:
            from PIL import Image
            import numpy as np
            
            # 创建一个简单的彩色图像
            img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            img.save(os.path.join(sample_class_dir, "sample.JPEG"))
            print(f"创建了示例图像: {os.path.join(sample_class_dir, 'sample.JPEG')}")
        except Exception as e:
            print(f"创建示例图像时出错: {e}")
            # 创建一个空文件
            with open(os.path.join(sample_class_dir, "sample.JPEG"), "wb") as f:
                f.write(b"")
    
    # 构建ImageNet验证集配置
    dataset_cfg = dict(
        type='ImageNet',
        data_root=data_root,
        split=split,
        pipeline=[dict(type='LoadImageFromFile')],
    )
    
    # 注意：MMPretrain的ImageNet数据集不支持sample_ratio参数
    # 我们将在下载后手动处理采样
    
    # 初始化默认作用域
    init_default_scope('mmpretrain')
    
    try:
        # 构建数据集（这会触发下载）
        dataset = build_dataset(dataset_cfg)
        total_samples = len(dataset)
        print(f"成功准备ImageNet数据集，共 {total_samples} 个样本")
        
        # 如果需要采样，打印提示信息
        if sample_ratio < 1.0:
            sample_size = int(total_samples * sample_ratio)
            print(f"根据采样比例 {sample_ratio*100:.1f}%，建议使用前 {sample_size} 个样本进行测试")
            print("您可以在代码中使用以下方式进行采样：")
            print(f"dataset = dataset[:{sample_size}]")
        
        return dataset
    except Exception as e:
        print(f"准备ImageNet数据集时出错: {e}")
        print("\n提示: ImageNet是一个大型数据集，MMPretrain无法自动下载完整数据集")
        print("建议使用CIFAR10或Tiny-ImageNet等较小的数据集进行测试:")
        print("python download_data.py --dataset cifar10")
        print("python download_data.py --dataset tiny-imagenet")
        return None

def download_cifar(dataset_name='cifar10', data_root='data/cifar', split='test'):
    """下载CIFAR数据集
    
    Args:
        dataset_name (str): 'cifar10'或'cifar100'
        data_root (str): 数据集保存路径
        split (str): 数据集分割，'train'或'test'
    
    Returns:
        dataset: 构建好的数据集对象
    """
    print(f"正在准备{dataset_name.upper()}数据集 ({split})...")
    print(f"数据将保存到: {data_root}")
    
    # 确保目录存在
    os.makedirs(data_root, exist_ok=True)
    
    # 构建CIFAR数据集配置
    dataset_cfg = dict(
        type=dataset_name.upper(),
        data_root=data_root,
        split=split,
        pipeline=[dict(type='LoadImageFromFile')],
    )
    
    # 初始化默认作用域
    init_default_scope('mmpretrain')
    
    try:
        # 构建数据集（这会触发下载）
        dataset = build_dataset(dataset_cfg)
        print(f"成功准备{dataset_name.upper()}数据集，共 {len(dataset)} 个样本")
        return dataset
    except Exception as e:
        print(f"下载{dataset_name.upper()}数据集时出错: {e}")
        return None

def download_tiny_imagenet(data_root='data/tiny-imagenet', split='val'):
    """下载Tiny-ImageNet数据集
    
    Tiny-ImageNet是ImageNet的一个小型版本，包含200个类别，每个类别500张训练图像和50张验证图像。
    图像尺寸为64×64像素。总大小约为150MB。
    
    Args:
        data_root (str): 数据集保存路径
        split (str): 数据集分割，'train'或'val'
        
    Returns:
        bool: 是否成功下载和准备数据集
    """
    import requests
    from tqdm import tqdm
    
    print(f"正在准备Tiny-ImageNet数据集...")
    print(f"数据将保存到: {data_root}")
    
    # 确保目录存在
    os.makedirs(data_root, exist_ok=True)
    
    # Tiny-ImageNet下载链接
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = os.path.join(data_root, "tiny-imagenet-200.zip")
    extract_dir = os.path.join(data_root, "tiny-imagenet-200")
    
    # 检查是否已经下载和解压
    if os.path.exists(extract_dir) and os.path.exists(os.path.join(extract_dir, "train")) and os.path.exists(os.path.join(extract_dir, "val")):
        print(f"Tiny-ImageNet数据集已存在: {extract_dir}")
        print(f"包含训练集和验证集目录")
        return True
    
    # 下载数据集
    if not os.path.exists(zip_path):
        print(f"正在下载Tiny-ImageNet数据集 (约150MB)...")
        try:
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 KB
            
            with open(zip_path, 'wb') as f, tqdm(
                    desc="下载进度",
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                for data in response.iter_content(block_size):
                    size = f.write(data)
                    bar.update(size)
            
            print(f"下载完成: {zip_path}")
        except Exception as e:
            print(f"下载Tiny-ImageNet数据集时出错: {e}")
            return False
    else:
        print(f"已存在Tiny-ImageNet压缩包: {zip_path}")
    
    # 解压数据集
    if not os.path.exists(extract_dir):
        print("正在解压Tiny-ImageNet数据集...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_root)
            print(f"解压完成: {extract_dir}")
        except Exception as e:
            print(f"解压Tiny-ImageNet数据集时出错: {e}")
            return False
    else:
        print(f"已存在解压后的Tiny-ImageNet目录: {extract_dir}")
    
    # 检查数据集结构
    train_dir = os.path.join(extract_dir, "train")
    val_dir = os.path.join(extract_dir, "val")
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print(f"错误: Tiny-ImageNet数据集结构不完整")
        return False
    
    # 检查验证集结构，可能需要重新组织
    val_images_dir = os.path.join(val_dir, "images")
    val_annotations_file = os.path.join(val_dir, "val_annotations.txt")
    
    # 如果验证集图像都在images目录下，需要按类别重新组织
    if os.path.exists(val_images_dir) and os.path.exists(val_annotations_file) and len([d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d)) and d != "images"]) < 10:
        print("正在重新组织验证集图像...")
        
        # 读取验证集标注文件
        val_annotations = {}
        with open(val_annotations_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    image_name, class_id = parts[0], parts[1]
                    val_annotations[image_name] = class_id
        
        # 按类别创建目录并移动图像
        for image_name, class_id in val_annotations.items():
            # 创建类别目录
            class_dir = os.path.join(val_dir, class_id)
            os.makedirs(class_dir, exist_ok=True)
            
            # 移动图像
            src_path = os.path.join(val_images_dir, image_name)
            dst_path = os.path.join(class_dir, image_name)
            
            if os.path.exists(src_path) and not os.path.exists(dst_path):
                shutil.copy(src_path, dst_path)
        
        print("验证集图像重新组织完成")
    
    print("Tiny-ImageNet数据集准备完成")
    print(f"- 训练集: {train_dir}")
    print(f"- 验证集: {val_dir}")
    print("包含200个类别，每个类别500张训练图像和50张验证图像")
    print("图像尺寸: 64×64像素")
    
    # 打印使用示例
    print("\n使用示例:")
    print("1. 在推理脚本中指定Tiny-ImageNet数据集路径")
    print("2. 使用适合小图像的模型，如ResNet-18或MobileNet")
    
    return None

def list_available_datasets():
    """列出可用的数据集"""
    print("可用的数据集:")
    print("1. ImageNet - 1000类图像分类标准数据集 (需要手动下载)")
    print("2. CIFAR10 - 10类小型彩色图像数据集 (32x32像素，约170MB)")
    print("3. CIFAR100 - 100类小型彩色图像数据集 (32x32像素，约170MB)")
    print("4. Tiny-ImageNet - 200类小型图像数据集 (64x64像素，约150MB)")

def main():
    """主函数"""
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='MMPretrain图像分类数据集下载工具')
    parser.add_argument('--dataset', type=str, default='cifar10', 
                        choices=['imagenet', 'cifar10', 'cifar100', 'tiny-imagenet'],
                        help='要下载的数据集名称 (默认: cifar10)')
    parser.add_argument('--data-root', type=str, default='./data',
                        help='数据集保存的根目录 (默认: ./data)')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'test', 'val'],
                        help='数据集分割 (默认: test)')
    parser.add_argument('--sample-ratio', type=float, default=0.1,
                        help='采样比例，仅对ImageNet有效 (默认: 0.1)')
    parser.add_argument('--list', action='store_true',
                        help='列出所有可用的数据集')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 如果指定了--list参数，则列出可用数据集并退出
    if args.list:
        list_available_datasets()
        return
    
    # 根据指定的数据集名称下载相应的数据集
    if args.dataset == 'imagenet':
        data_path = os.path.join(args.data_root, 'imagenet')
        download_imagenet(data_path, args.split, args.sample_ratio)
    elif args.dataset.startswith('cifar'):
        data_path = os.path.join(args.data_root, args.dataset)
        download_cifar(args.dataset, data_path, args.split)
    elif args.dataset == 'tiny-imagenet':
        data_path = os.path.join(args.data_root, 'tiny-imagenet')
        download_tiny_imagenet(data_path, args.split)
    else:
        print(f"不支持的数据集: {args.dataset}")
        print("使用 --list 参数查看所有可用的数据集")

if __name__ == '__main__':
    main()