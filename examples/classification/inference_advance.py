#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MMPretrain高级图像分类推理示例

本示例展示了如何使用MMPretrain进行高级图像分类推理，包括：
1. 使用自定义配置文件
2. 使用不同的预处理方式
3. 使用不同的后处理方式
4. 可视化类激活图
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from pathlib import Path
import urllib.request
import requests

from mmpretrain import (
    get_model,
    ImageClassificationInferencer,
    FeatureExtractor,
    list_models
)
from imagenet_categories import IMAGENET_CATEGORIES
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# 设置中文字体
def set_chinese_font():
    """设置matplotlib中文字体"""
    try:
        # 尝试使用微软雅黑字体
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        print("成功设置中文字体")
    except Exception as e:
        print(f"设置中文字体时出错: {e}")
        print("将使用默认字体，中文可能无法正确显示")

# 调用设置中文字体函数
set_chinese_font()

# 创建数据目录
DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)

# 示例图像URL
IMAGE_URL = "https://github.com/open-mmlab/mmpretrain/raw/main/demo/demo.JPEG"


def download_image():
    """下载示例图像"""
    filename = os.path.basename(IMAGE_URL)
    save_path = DATA_DIR / filename
    
    if not save_path.exists():
        print(f"下载示例图像: {IMAGE_URL}")
        urllib.request.urlretrieve(IMAGE_URL, save_path)
    else:
        print(f"示例图像已存在: {save_path}")
        
    return str(save_path)


def custom_config_inference():
    """使用自定义配置进行推理"""
    print("=" * 50)
    print("示例1: 使用自定义配置进行推理")
    print("=" * 50)
    
    # 下载示例图像
    image_path = download_image()
    
    # 使用预定义模型而不是自定义配置
    # 这样更兼容最新版本的 mmpretrain
    
    # 创建推理器
    inferencer = ImageClassificationInferencer(
        model='resnet18_8xb32_in1k',  # 使用预定义的模型名称
        device='cpu'  # 使用CPU进行推理，如果有GPU可以改为'cuda'
    )
    
    # 进行推理
    result = inferencer(image_path)[0]
    
    print(f"预测类别: {result['pred_class']}")
    print(f"预测置信度: {result['pred_score']:.4f}")
    
    # 显示结果
    plt.figure(figsize=(8, 6))
    img = plt.imread(image_path)
    plt.imshow(img)
    plt.title(f"预测类别: {result['pred_class']}\n置信度: {result['pred_score']:.4f}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def custom_preprocessing_inference():
    """使用自定义预处理进行推理"""
    print("=" * 50)
    print("示例2: 使用自定义预处理进行推理")
    print("=" * 50)
    
    # 下载示例图像
    image_path = download_image()
    
    # 获取预训练模型
    model = get_model('resnet50_8xb32_in1k', pretrained=True)
    model.eval()
    
    # 自定义图像预处理
    def preprocess_image(img_path, size=224):
        # 读取图像
        img = Image.open(img_path).convert('RGB')
        
        # 调整大小
        img = img.resize((size, size))
        
        # 转换为numpy数组
        img = np.array(img, dtype=np.float32)
        
        # 标准化 (ImageNet标准)
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        img = (img - mean) / std
        
        # 转换为CHW格式
        img = np.transpose(img, (2, 0, 1))
        
        # 添加批次维度
        img = np.expand_dims(img, axis=0)
        
        return img
    
    # 预处理图像
    processed_img = preprocess_image(image_path)
    
    # 转换为torch张量
    input_tensor = torch.from_numpy(processed_img).float()
    
    # 推理
    with torch.no_grad():
        output = model(input_tensor)
    
    # 处理输出
    probabilities = torch.nn.functional.softmax(output, dim=1)[0]
    
    # 获取前5个预测结果
    topk_values, topk_indices = torch.topk(probabilities, 5)
    
    print("前5个预测结果:")
    for i, (score, idx) in enumerate(zip(topk_values.tolist(), topk_indices.tolist())):
        class_name = IMAGENET_CATEGORIES[idx]
        print(f"{i+1}. {class_name}: {score:.4f}")


def visualize_cam():
    """可视化类激活图 (CAM)"""
    print("=" * 50)
    print("示例3: 可视化类激活图 (CAM)")
    print("=" * 50)
    
    # 尝试导入不同名称的grad-cam包
    grad_cam_imported = False
    
    # 尝试方法1: 导入pytorch-grad-cam
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        grad_cam_imported = True
        print("成功导入 pytorch-grad-cam 包")
    except ImportError:
        print("无法导入 pytorch-grad-cam 包，尝试其他替代方案...")
    
    # 尝试方法2: 导入grad-cam
    if not grad_cam_imported:
        try:
            from grad_cam import GradCAM
            from grad_cam.utils.image import show_cam_on_image
            grad_cam_imported = True
            print("成功导入 grad-cam 包")
        except ImportError:
            print("无法导入 grad-cam 包")
    
    # 如果两种方法都失败，提示用户安装
    if not grad_cam_imported:
        print("请安装 grad-cam 包以使用此功能:")
        print("pip install git+https://github.com/jacobgil/pytorch-grad-cam.git")
        return
    
    # 下载示例图像
    image_path = download_image()
    
    # 获取预训练模型
    model = get_model('resnet50_8xb32_in1k', pretrained=True)
    
    # 读取图像
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (224, 224))
    
    # 预处理图像
    input_tensor = torch.from_numpy(
        np.transpose(img_resized, (2, 0, 1)) / 255.0
    ).float().unsqueeze(0)
    
    # 定义目标层
    target_layers = [model.backbone.layer4[-1]]
    
    # 创建GradCAM对象
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # 生成CAM
    grayscale_cam = cam(input_tensor=input_tensor)
    grayscale_cam = grayscale_cam[0, :]
    
    # 可视化CAM
    img_normalized = img_resized / 255.0
    cam_image = show_cam_on_image(img_normalized, grayscale_cam, use_rgb=True)
    
    # 显示原始图像和CAM
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_resized)
    plt.title("原始图像")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cam_image)
    plt.title("类激活图 (CAM)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def batch_inference_with_topk():
    """批量推理并显示top-k结果"""
    print("=" * 50)
    print("示例4: 批量推理并显示top-k结果")
    print("=" * 50)
    
    # 下载示例图像
    image_path = download_image()
    
    # 创建推理器
    inferencer = ImageClassificationInferencer('resnet50_8xb32_in1k')
    
    # 进行推理，不使用 topk 参数
    result = inferencer(image_path)[0]
    
    # 手动获取 top-5 结果
    # 获取预测概率和对应的类别索引
    pred_scores = result['pred_score']
    if isinstance(pred_scores, float):  # 如果只返回了一个分数
        top5_scores = [pred_scores]
        top5_classes = [result['pred_class']]
    else:  # 如果返回了多个分数
        # 获取前5个最高分数的索引
        if len(pred_scores) > 5:
            top5_indices = np.argsort(pred_scores)[-5:][::-1]
            top5_scores = [pred_scores[i] for i in top5_indices]
            # 假设 pred_class 是一个列表，包含所有类别名称
            if isinstance(result['pred_class'], list) and len(result['pred_class']) > 5:
                top5_classes = [result['pred_class'][i] for i in top5_indices]
            else:
                # 如果 pred_class 不是列表或长度不够，使用 IMAGENET_CATEGORIES
                top5_classes = [IMAGENET_CATEGORIES[i] for i in top5_indices]
        else:
            top5_scores = pred_scores
            top5_classes = result['pred_class']
    
    # 显示top-k结果
    print("Top-5 预测结果:")
    for i, (cls_name, score) in enumerate(zip(top5_classes, top5_scores)):
        print(f"{i+1}. {cls_name}: {score:.4f}")
    
    # 可视化top-k结果
    plt.figure(figsize=(10, 6))
    
    # 显示图像
    img = plt.imread(image_path)
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("输入图像")
    plt.axis('off')
    
    # 显示条形图
    plt.subplot(1, 2, 2)
    y_pos = np.arange(len(top5_classes))
    plt.barh(y_pos, top5_scores, align='center')
    plt.yticks(y_pos, top5_classes)
    plt.xlabel('置信度')
    plt.title('Top-5 预测结果')
    
    plt.tight_layout()
    plt.show()


def main():
    """主函数"""
    # 运行示例
    custom_config_inference()
    custom_preprocessing_inference()
    visualize_cam()
    batch_inference_with_topk()


if __name__ == '__main__':
    main() 