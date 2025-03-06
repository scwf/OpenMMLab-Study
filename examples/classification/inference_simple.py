#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MMPretrain图像分类演示脚本

本脚本会下载示例图像并使用预训练模型进行图像分类推理
同时也支持对CIFAR10和Tiny-ImageNet数据集进行推理
"""

import os
import sys
import urllib.request
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties
import numpy as np
import random
from mmpretrain import inference_model, ImageClassificationInferencer

# 设置中文字体支持
def set_chinese_font():
    """设置中文字体"""
    # 尝试设置中文字体
    try:
        # 尝试使用系统中可能存在的中文字体
        if os.name == 'nt':  # Windows系统
            font_list = ['Microsoft YaHei', 'SimHei', 'SimSun', 'NSimSun', 'FangSong', 'KaiTi']
        else:  # Linux/Mac系统
            font_list = ['WenQuanYi Micro Hei', 'Hiragino Sans GB', 'Heiti SC', 'STHeiti', 'Source Han Sans CN']
        
        # 尝试每一个字体，直到找到可用的
        font_found = False
        for font_name in font_list:
            try:
                font_prop = FontProperties(fname=matplotlib.font_manager.findfont(font_name))
                plt.rcParams['font.family'] = font_prop.get_name()
                font_found = True
                print(f"使用中文字体: {font_name}")
                break
            except:
                continue
        
        if not font_found:
            # 如果没有找到中文字体，使用英文显示
            print("警告: 未找到支持中文的字体，将使用英文显示")
    except Exception as e:
        print(f"设置中文字体时出错: {e}")
        print("将使用英文显示")

# 调用设置中文字体函数
set_chinese_font()

# 创建数据目录
DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)

# 示例图像URL
IMAGE_URLS = {
    "dog": "https://github.com/open-mmlab/mmpretrain/raw/main/demo/dog.jpg",
    "snake": "https://github.com/open-mmlab/mmpretrain/raw/main/demo/demo.JPEG",
    "bird": "https://github.com/open-mmlab/mmpretrain/raw/main/demo/bird.JPEG",
}

# CIFAR10类别名称
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def load_tiny_imagenet_classes(tiny_imagenet_dir=None):
    """加载Tiny-ImageNet的类别映射
    
    Args:
        tiny_imagenet_dir: Tiny-ImageNet数据集目录
        
    Returns:
        class_map: WordNet ID到类别名称的映射
    """
    # 设置默认数据集目录
    if tiny_imagenet_dir is None:
        tiny_imagenet_dir = Path("./data/tiny-imagenet/tiny-imagenet-200")
    else:
        tiny_imagenet_dir = Path(tiny_imagenet_dir)
    
    # 初始化空的类别映射
    class_map = {}
    
    # 尝试从words.txt文件加载类别映射
    words_file = tiny_imagenet_dir / "words.txt"
    
    if words_file.exists():
        try:
            print(f"从 {words_file} 加载类别映射...")
            with open(words_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        wnid, names = parts
                        # 使用第一个名称作为主要类别名称
                        class_name = names.split(',')[0].strip()
                        class_map[wnid] = class_name
            print(f"成功加载了 {len(class_map)} 个类别映射")
        except Exception as e:
            print(f"加载Tiny-ImageNet类别映射时出错: {e}")
            print("将使用WordNet ID作为类别名称")
    else:
        print(f"警告: 未找到类别映射文件 {words_file}")
        print("将使用WordNet ID作为类别名称")
        
        # 尝试从目录结构中获取类别ID
        try:
            train_dir = tiny_imagenet_dir / "train"
            if train_dir.exists():
                class_dirs = [d.name for d in train_dir.iterdir() if d.is_dir()]
                for class_id in class_dirs:
                    class_map[class_id] = class_id  # 使用ID作为名称
                print(f"从目录结构中获取了 {len(class_map)} 个类别ID")
        except Exception as e:
            print(f"从目录结构获取类别ID时出错: {e}")
    
    return class_map

def download_images():
    """下载示例图像"""
    image_paths = {}
    
    print("正在下载示例图像...")
    for name, url in IMAGE_URLS.items():
        filename = os.path.basename(url)
        save_path = DATA_DIR / filename
        
        if not save_path.exists():
            print(f"下载 {name} 图像: {url}")
            urllib.request.urlretrieve(url, save_path)
        else:
            print(f"{name} 图像已存在: {save_path}")
            
        image_paths[name] = str(save_path)
    
    return image_paths


def show_result(img_path, result):
    """显示推理结果"""
    plt.figure(figsize=(8, 6))
    img = plt.imread(str(img_path))  # 确保路径是字符串
    plt.imshow(img)
    
    # 使用英文显示结果，避免中文字体问题
    plt.title(f"Prediction: {result['pred_class']}\nConfidence: {result['pred_score']:.4f}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def get_cifar10_images(num_images=5):
    """获取CIFAR10数据集中的图像
    
    Args:
        num_images: 要获取的图像数量
        
    Returns:
        image_paths: 图像路径字典，格式为 {类别_索引: 路径}
        image_data: 图像数据列表，用于直接推理
    """
    cifar10_dir = Path("./data/cifar10")
    
    if not cifar10_dir.exists():
        print(f"错误: CIFAR10数据集目录不存在: {cifar10_dir}")
        print("请先运行 'python download_data.py --dataset cifar10' 下载数据集")
        return {}, []
    
    # 检查CIFAR10数据集文件
    batch_dir = cifar10_dir / "cifar-10-batches-py"
    if not batch_dir.exists():
        print(f"错误: CIFAR10批次目录不存在: {batch_dir}")
        
        # 检查是否有压缩包
        tar_file = cifar10_dir / "cifar-10-python.tar.gz"
        if tar_file.exists():
            print(f"找到CIFAR10压缩包: {tar_file}")
            print("请先解压缩该文件，然后再运行此脚本")
        return {}, []
    
    # 从CIFAR10批次文件中加载数据
    try:
        import pickle
        import numpy as np
        from PIL import Image
        import io
        
        # 加载测试批次
        test_batch_path = batch_dir / "test_batch"
        if not test_batch_path.exists():
            print(f"错误: 测试批次文件不存在: {test_batch_path}")
            return {}, []
        
        # 加载元数据以获取类别名称
        meta_path = batch_dir / "batches.meta"
        if meta_path.exists():
            with open(meta_path, 'rb') as f:
                meta_data = pickle.load(f, encoding='bytes')
                label_names = [name.decode('utf-8') for name in meta_data[b'label_names']]
        else:
            # 如果元数据文件不存在，使用预定义的类别名称
            label_names = CIFAR10_CLASSES
        
        # 加载测试数据
        with open(test_batch_path, 'rb') as f:
            test_data = pickle.load(f, encoding='bytes')
            test_images = test_data[b'data']
            test_labels = test_data[b'labels']
        
        # 重塑图像数据为 (32, 32, 3) 格式
        test_images = test_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        
        # 随机选择指定数量的图像
        total_images = len(test_images)
        selected_indices = random.sample(range(total_images), min(num_images, total_images))
        
        # 创建临时目录保存图像
        temp_dir = DATA_DIR / "cifar10_temp"
        temp_dir.mkdir(exist_ok=True)
        
        # 保存选定的图像并创建路径字典
        image_paths = {}
        image_data = []
        
        for i, idx in enumerate(selected_indices):
            img_data = test_images[idx]
            label = test_labels[idx]
            class_name = label_names[label]
            
            # 创建PIL图像并保存
            img = Image.fromarray(img_data)
            img_path = temp_dir / f"{class_name}_{i}.png"
            img.save(img_path)
            
            # 添加到路径字典
            image_paths[f"{class_name}_{i}"] = str(img_path)
            image_data.append(img_data)
            
        print(f"成功从CIFAR10数据集中提取了 {len(image_paths)} 张图像")
        return image_paths, image_data
        
    except Exception as e:
        print(f"处理CIFAR10数据集时出错: {e}")
        import traceback
        traceback.print_exc()
        return {}, []


def get_tiny_imagenet_images(num_images=5, split='val'):
    """获取Tiny-ImageNet数据集中的图像
    
    Args:
        num_images: 要获取的图像数量
        split: 数据集分割，'train'或'val'
        
    Returns:
        image_paths: 图像路径字典，格式为 {类别_索引: 路径}
    """
    tiny_imagenet_dir = Path("./data/tiny-imagenet/tiny-imagenet-200")
    
    if not tiny_imagenet_dir.exists():
        print(f"错误: Tiny-ImageNet数据集目录不存在: {tiny_imagenet_dir}")
        print("请先运行 'python download_data.py --dataset tiny-imagenet' 下载数据集")
        return {}
    
    # 检查数据集分割目录
    split_dir = tiny_imagenet_dir / split
    if not split_dir.exists():
        print(f"错误: Tiny-ImageNet {split}集目录不存在: {split_dir}")
        return {}
    
    # 收集所有类别目录
    class_dirs = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
    
    if not class_dirs:
        print(f"错误: 在Tiny-ImageNet {split}集中没有找到类别目录")
        return {}
    
    # 随机选择一些类别
    selected_classes = random.sample(class_dirs, min(num_images, len(class_dirs)))
    
    # 为每个选定的类别选择一张图像
    image_paths = {}
    
    for i, class_name in enumerate(selected_classes):
        class_dir = os.path.join(split_dir, class_name)
        
        # 获取该类别的所有图像
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
        
        if images:
            # 随机选择一张图像
            selected_image = random.choice(images)
            image_path = os.path.join(class_dir, selected_image)
            
            # 添加到路径字典
            image_paths[f"{class_name}_{i}"] = image_path
    
    print(f"成功从Tiny-ImageNet数据集中提取了 {len(image_paths)} 张图像")
    return image_paths


def run_inference_demo():
    """运行推理演示"""
    # 下载示例图像
    image_paths = download_images()
    
    # 选择模型
    model_name = 'resnet50_8xb32_in1k'  # 使用ResNet-50
    
    print(f"\n使用模型 {model_name} 进行推理")
    print("=" * 50)
    
    # 方法1: 使用inference_model进行单张图像推理
    print("\n方法1: 使用inference_model进行单张图像推理")
    for name, img_path in image_paths.items():
        print(f"\n处理 {name} 图像...")
        result = inference_model(model_name, img_path)
        print(f"预测类别: {result['pred_class']}")
        print(f"预测置信度: {result['pred_score']:.4f}")
    
    # 方法2: 使用ImageClassificationInferencer进行批量推理
    print("\n\n方法2: 使用ImageClassificationInferencer进行批量推理")
    inferencer = ImageClassificationInferencer(model_name)
    
    # 批量推理所有图像
    image_list = list(image_paths.values())
    results = inferencer(image_list)
    
    for i, (name, result) in enumerate(zip(image_paths.keys(), results)):
        print(f"\n{name} 图像结果:")
        print(f"预测类别: {result['pred_class']}")
        print(f"预测置信度: {result['pred_score']:.4f}")
    
    # 显示第一张图像的结果
    first_image = list(image_paths.values())[0]
    first_result = results[0]
    show_result(first_image, first_result)


def run_cifar10_inference(num_images=5):
    """在CIFAR10数据集上运行推理
    
    Args:
        num_images: 要推理的图像数量
    """
    print("\n" + "=" * 50)
    print("CIFAR10数据集推理示例")
    print("=" * 50)
    
    # 获取CIFAR10图像
    cifar_images, _ = get_cifar10_images(num_images)
    
    if not cifar_images:
        print("无法获取CIFAR10图像，跳过推理")
        return
    
    print(f"获取了 {len(cifar_images)} 张CIFAR10图像")
    
    # 选择适合CIFAR10的模型
    # 注意: 使用在CIFAR10上预训练的模型会有更好的效果
    model_name = 'resnet18_8xb16_cifar10'  # 使用在CIFAR10上训练的ResNet-18
    
    print(f"\n使用模型 {model_name} 进行CIFAR10推理")
    
    # 创建推理器
    inferencer = ImageClassificationInferencer(model_name)
    
    # 批量推理所有CIFAR10图像
    image_list = list(cifar_images.values())
    results = inferencer(image_list)
    
    # 显示结果
    for i, (name, result) in enumerate(zip(cifar_images.keys(), results)):
        print(f"\n图像 {name} 结果:")
        print(f"预测类别: {result['pred_class']}")
        print(f"预测置信度: {result['pred_score']:.4f}")
        
        # 获取真实类别（从图像名称中提取）
        true_class = name.split('_')[0]
        print(f"真实类别: {true_class}")
        
        # 检查预测是否正确
        is_correct = true_class in result['pred_class'].lower()
        print(f"预测{'正确' if is_correct else '错误'}")
    
    # 显示第一张CIFAR10图像的结果
    if cifar_images:
        first_image = list(cifar_images.values())[0]
        first_result = results[0]
        show_result(first_image, first_result)


def run_tiny_imagenet_inference(num_images=5, split='val'):
    """在Tiny-ImageNet数据集上运行推理
    
    Args:
        num_images: 要推理的图像数量
        split: 数据集分割，'train'或'val'
    """
    print("\n" + "=" * 50)
    print(f"Tiny-ImageNet数据集推理示例 ({split}集)")
    print("=" * 50)
    
    # 获取Tiny-ImageNet图像
    tiny_imagenet_images = get_tiny_imagenet_images(num_images, split)
    
    if not tiny_imagenet_images:
        print("无法获取Tiny-ImageNet图像，跳过推理")
        return
    
    print(f"获取了 {len(tiny_imagenet_images)} 张Tiny-ImageNet图像")
    
    # 加载Tiny-ImageNet类别映射
    tiny_imagenet_dir = Path("./data/tiny-imagenet/tiny-imagenet-200")
    class_map = load_tiny_imagenet_classes(tiny_imagenet_dir)
    
    # 选择适合小图像的模型
    # 对于Tiny-ImageNet，可以使用在ImageNet上预训练的模型
    model_name = 'resnet18_8xb32_in1k'  # 使用在ImageNet上训练的ResNet-18
    
    print(f"\n使用模型 {model_name} 进行Tiny-ImageNet推理")
    
    # 创建推理器
    inferencer = ImageClassificationInferencer(model_name)
    
    # 批量推理所有Tiny-ImageNet图像
    image_list = list(tiny_imagenet_images.values())
    results = inferencer(image_list)
    
    # 显示结果
    correct_count = 0
    for i, (name, result) in enumerate(zip(tiny_imagenet_images.keys(), results)):
        print(f"\n图像 {name} 结果:")
        print(f"预测类别: {result['pred_class']}")
        print(f"预测置信度: {result['pred_score']:.4f}")
        
        # 获取真实类别（从图像名称中提取）
        true_class_id = name.split('_')[0]
        true_class_name = class_map.get(true_class_id, true_class_id)
        print(f"真实类别ID: {true_class_id}")
        print(f"真实类别名称: {true_class_name}")
        
        # 检查预测是否正确（简单字符串匹配）
        # 由于ImageNet和Tiny-ImageNet的类别名称可能有差异，这里使用简单的包含关系判断
        pred_class_lower = result['pred_class'].lower()
        true_class_lower = true_class_name.lower()
        
        # 检查预测类别是否包含真实类别名称的关键词，或者真实类别名称是否包含预测类别的关键词
        is_correct = (
            any(word in pred_class_lower for word in true_class_lower.split('_')) or
            any(word in true_class_lower for word in pred_class_lower.split('_'))
        )
        
        if is_correct:
            correct_count += 1
            print(f"预测正确 ✓")
        else:
            print(f"预测错误 ✗")
    
    # 计算准确率
    if tiny_imagenet_images:
        accuracy = correct_count / len(tiny_imagenet_images) * 100
        print(f"\n准确率: {accuracy:.2f}% ({correct_count}/{len(tiny_imagenet_images)})")
    
    # 显示第一张Tiny-ImageNet图像的结果
    if tiny_imagenet_images:
        first_image = list(tiny_imagenet_images.values())[0]
        first_result = results[0]
        show_result(first_image, first_result)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='MMPretrain图像分类推理演示')
    parser.add_argument('--dataset', type=str, default='demo',
                        choices=['demo', 'cifar10', 'tiny-imagenet', 'all'],
                        help='要使用的数据集 (默认: demo)')
    parser.add_argument('--num-images', type=int, default=5,
                        help='每个数据集要推理的图像数量 (默认: 5)')
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val', 'test'],
                        help='数据集分割 (默认: val，仅对Tiny-ImageNet有效)')
    return parser.parse_args()


if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    
    # 根据指定的数据集运行相应的推理
    if args.dataset == 'demo' or args.dataset == 'all':
        run_inference_demo()
    
    if args.dataset == 'cifar10' or args.dataset == 'all':
        run_cifar10_inference(num_images=args.num_images)
    
    if args.dataset == 'tiny-imagenet' or args.dataset == 'all':
        run_tiny_imagenet_inference(num_images=args.num_images, split=args.split) 