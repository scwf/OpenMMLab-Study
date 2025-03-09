#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
图像分类推理示例

本示例展示了如何使用MMPretrain进行图像分类推理，包括：
1. 使用inference_model快速推理
2. 使用ImageClassificationInferencer进行批量推理
3. 使用get_model获取模型并手动推理
4. 使用FeatureExtractor提取特征
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from mmpretrain import (
    inference_model,
    get_model,
    list_models,
    ImageClassificationInferencer,
    FeatureExtractor
)
from imagenet_categories import IMAGENET_CATEGORIES


def show_result(img_path, result):
    """显示推理结果"""
    img = Image.open(img_path)
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.title(f"预测类别: {result['pred_class']}\n置信度: {result['pred_score']:.4f}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def example_inference_model():
    """使用inference_model进行快速推理"""
    print("=" * 50)
    print("示例1: 使用inference_model进行快速推理")
    print("=" * 50)
    
    # 使用预训练的ResNet-50模型
    model_name = 'resnet50_8xb32_in1k'
    # 替换为您自己的图像路径
    image_path = './data/dog.jpg'
    
    # 如果没有图形界面，请设置show=False
    result = inference_model(model_name, image_path, show=False)
    
    print(f"预测类别: {result['pred_class']}")
    print(f"预测标签索引: {result['pred_label']}")
    print(f"预测置信度: {result['pred_score']:.4f}")
    
    # 显示结果
    show_result(image_path, result)


def example_classification_inferencer():
    """使用ImageClassificationInferencer进行批量推理"""
    print("=" * 50)
    print("示例2: 使用ImageClassificationInferencer进行批量推理")
    print("=" * 50)
    
    # 创建推理器实例
    inferencer = ImageClassificationInferencer('resnet50_8xb32_in1k')
    
    # 单张图像推理
    image_path = './data/dog.jpg'
    result = inferencer(image_path)[0]  # 注意inferencer返回的是结果列表
    
    print(f"单张图像推理结果:")
    print(f"预测类别: {result['pred_class']}")
    print(f"预测置信度: {result['pred_score']:.4f}")
    
    # 批量推理
    image_list = ['./data/bird.JPEG', './data/dog.jpg', './data/demo.JPEG']
    results = inferencer(image_list, batch_size=2)
    
    print(f"\n批量推理结果:")
    for i, res in enumerate(results):
        print(f"图像 {i+1}: {res['pred_class']} (置信度: {res['pred_score']:.4f})")


def example_get_model():
    """使用get_model获取模型并手动推理"""
    print("=" * 50)
    print("示例3: 使用get_model获取模型并手动推理")
    print("=" * 50)
    
    # 获取预训练模型
    model = get_model('resnet50_8xb32_in1k', pretrained=True)
    model.eval()
    
    # 准备输入数据
    image_path = './data/dog.jpg'
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    
    # 图像预处理
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])
    img = np.array(img, dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))  # (H, W, C) -> (C, H, W)
    img = np.expand_dims(img, axis=0)  # (C, H, W) -> (1, C, H, W)
    
    # 转换为torch张量
    input_tensor = torch.from_numpy(img).float()
    
    # 推理
    with torch.no_grad():
        output = model(input_tensor)
    
    # 处理输出
    probabilities = torch.nn.functional.softmax(output, dim=1)[0]
    pred_score, pred_label = torch.max(probabilities, dim=0)
    
    # 获取ImageNet类别名称
    pred_class = IMAGENET_CATEGORIES[pred_label.item()]
    
    print(f"预测类别: {pred_class}")
    print(f"预测标签索引: {pred_label.item()}")
    print(f"预测置信度: {pred_score.item():.4f}")


def example_feature_extractor():
    """使用FeatureExtractor提取特征"""
    print("=" * 50)
    print("示例4: 使用FeatureExtractor提取特征")
    print("=" * 50)
    
    # 创建一个模型，输出多个阶段的特征
    model = get_model('resnet50_8xb32_in1k', backbone=dict(out_indices=(0, 1, 2, 3)))
    
    # 创建特征提取器
    extractor = FeatureExtractor(model)
    
    # 提取特征
    image_path = './data/dog.jpg'
    features = extractor(image_path)[0]
    
    # 打印每个阶段的特征形状
    for i, feat in enumerate(features):
        print(f"阶段 {i+1} 特征形状: {feat.shape}")
    
    # 可视化特征（以第一个特征为例）
    if len(features) > 0:
        # 取第一个特征的均值
        feature_map = features[0].mean(dim=0).numpy()
        
        # 检查特征图的形状
        if len(feature_map.shape) == 0:  # 如果是标量
            print(f"特征是标量值: {feature_map}, 无法可视化")
        elif len(feature_map.shape) == 1:  # 如果是一维向量
            plt.figure(figsize=(10, 4))
            plt.plot(feature_map)
            plt.title("特征向量可视化")
            plt.xlabel("特征维度")
            plt.ylabel("特征值")
            plt.grid(True)
            plt.show()
        else:  # 如果是二维或更高维
            plt.figure(figsize=(6, 6))
            plt.imshow(feature_map, cmap='viridis')
            plt.title("特征可视化")
            plt.colorbar()
            plt.show()


def list_available_models():
    """列出可用的图像分类模型"""
    print("=" * 50)
    print("可用的图像分类模型:")
    print("=" * 50)
    
    # 列出所有图像分类模型
    classification_models = ImageClassificationInferencer.list_models()
    
    # 只打印部分模型作为示例
    for model in classification_models[:10]:
        print(model)
    
    print(f"总共有 {len(classification_models)} 个可用模型")


def main():
    """主函数"""
    # 列出可用模型
    list_available_models()
    
    # 运行示例
    # 注意：请替换示例中的图像路径为您自己的图像
    example_inference_model()
    example_classification_inferencer()
    example_get_model()
    example_feature_extractor()


if __name__ == '__main__':
    main() 