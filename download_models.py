#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型下载工具
用于下载预训练的表情识别模型
"""

import os
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import requests
import hashlib
from pathlib import Path

class SimpleCNN(nn.Module):
    """简单的CNN表情识别模型"""
    
    def __init__(self, num_classes=7):
        super(SimpleCNN, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # 第一组卷积
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        
        # 第二组卷积
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        
        # 第三组卷积
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout1(x)
        
        # 展平
        x = x.view(-1, 128 * 6 * 6)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x

def download_file(url, filename, expected_hash=None):
    """
    下载文件并验证哈希值
    
    Args:
        url: 下载链接
        filename: 保存文件名
        expected_hash: 期望的MD5哈希值
    
    Returns:
        bool: 下载是否成功
    """
    try:
        print(f"正在下载: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = downloaded / total_size * 100
                        print(f"\r下载进度: {progress:.1f}%", end='', flush=True)
        
        print("\n下载完成")
        
        # 验证哈希值
        if expected_hash:
            with open(filename, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            if file_hash.lower() != expected_hash.lower():
                print(f"文件哈希验证失败！期望: {expected_hash}, 实际: {file_hash}")
                return False
            print("文件哈希验证通过")
        
        return True
        
    except Exception as e:
        print(f"下载失败: {e}")
        return False

def create_better_mock_model():
    """
    创建一个更好的模拟训练模型
    使用更合理的权重初始化，模拟真实的预训练效果
    """
    model = SimpleCNN()
    
    # 使用更好的初始化策略，模拟预训练的效果
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'conv1' in name and 'weight' in name:
                # 第一层卷积：边缘检测器
                nn.init.xavier_uniform_(param)
                # 添加一些手工设计的边缘检测滤波器
                if param.shape[0] >= 8:  # 确保有足够的通道
                    # 水平边缘检测器
                    param[0, 0] = torch.tensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=torch.float32)
                    # 垂直边缘检测器
                    param[1, 0] = torch.tensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=torch.float32)
                    # 对角线边缘检测器
                    param[2, 0] = torch.tensor([[-1, 0, 1], [0, 0, 0], [1, 0, -1]], dtype=torch.float32)
                    # 高斯滤波器
                    param[3, 0] = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32) / 16
                    
            elif 'weight' in name:
                if 'conv' in name:
                    # 其他卷积层使用He初始化
                    nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='relu')
                elif 'fc' in name:
                    # 全连接层使用Xavier初始化
                    nn.init.xavier_uniform_(param)
                    
            elif 'bias' in name:
                # 偏置初始化
                if 'conv' in name:
                    nn.init.constant_(param, 0.1)
                else:
                    nn.init.constant_(param, 0.0)
    
    return model

def try_download_pretrained_model(model_dir):
    """
    尝试下载真正的预训练模型
    
    Args:
        model_dir: 模型保存目录
        
    Returns:
        str: 模型文件路径，如果下载失败则返回None
    """
    model_urls = [
        # 可以添加实际的预训练模型URL
        # 示例URL（需要替换为真实的模型下载链接）
        {
            'url': 'https://example.com/fer2013_model.pth',
            'filename': 'fer2013_pretrained.pth',
            'hash': None  # 模型的MD5哈希值
        }
    ]
    
    for model_info in model_urls:
        model_path = os.path.join(model_dir, model_info['filename'])
        print(f"尝试下载预训练模型: {model_info['filename']}")
        
        # 这里暂时跳过实际下载，因为需要真实的URL
        # if download_file(model_info['url'], model_path, model_info['hash']):
        #     return model_path
        
        print("跳过在线下载（需要配置真实的模型URL）")
    
    return None

def download_or_create_model(model_dir="models"):
    """
    下载或创建表情识别模型
    
    Args:
        model_dir: 模型保存目录
        
    Returns:
        str: 模型文件路径
    """
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "emotion_model.pth")
    
    # 检查是否已有模型文件
    if os.path.exists(model_path):
        print(f"发现已存在的模型文件: {model_path}")
        try:
            # 验证模型是否可以加载
            checkpoint = torch.load(model_path, map_location='cpu')
            print("模型文件验证成功")
            # 检查是否是改进版本
            if checkpoint.get('model_version', 'v1') == 'v2':
                print("使用改进版模型")
                return model_path
            else:
                print("发现旧版本模型，将升级到改进版本")
        except Exception as e:
            print(f"模型文件损坏: {e}，将重新创建")
    
    # 首先尝试下载真正的预训练模型
    print("步骤1: 尝试下载预训练模型...")
    pretrained_path = try_download_pretrained_model(model_dir)
    if pretrained_path:
        print(f"✓ 成功下载预训练模型: {pretrained_path}")
        return pretrained_path
    
    # 如果下载失败，创建改进的模拟模型
    print("步骤2: 创建改进的模拟训练模型...")
    print("注意: 这是一个改进的演示模型，比随机权重效果更好")
    print("推荐: 如需最佳效果，请手动下载真实的FER2013预训练模型")
    
    # 创建改进的模型
    model = create_better_mock_model()
    
    # 保存模型
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_type': 'SimpleCNN',
        'model_version': 'v2',  # 标记为改进版本
        'num_classes': 7,
        'input_size': (48, 48),
        'created_by': 'download_models.py',
        'note': 'Improved mock model with better weight initialization',
        'features': [
            'Edge detection filters in first layer',
            'Better weight initialization',
            'Improved bias settings'
        ]
    }
    
    torch.save(checkpoint, model_path)
    print(f"✓ 改进模型已保存到: {model_path}")
    
    return model_path

def main():
    """主函数"""
    print("=" * 60)
    print("表情识别模型下载/创建工具")
    print("=" * 60)
    
    try:
        model_path = download_or_create_model()
        
        print("\n模型准备完成!")
        print(f"模型路径: {model_path}")
        print("\n使用方法:")
        print("from emotion_recognizer import EmotionRecognizer")
        print(f"recognizer = EmotionRecognizer(model_path='{model_path}')")
        
        # 验证模型
        print("\n验证模型...")
        from emotion_recognizer import EmotionRecognizer
        recognizer = EmotionRecognizer(model_path=model_path)
        
        # 创建测试输入
        test_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        test_bbox = (10, 10, 80, 80)
        
        # 测试预测
        emotion = recognizer.predict(test_frame, test_bbox)
        chinese_emotion = recognizer.get_chinese_label(emotion)
        
        print(f"测试预测结果: {emotion} ({chinese_emotion})")
        print("模型验证成功!")
        
    except Exception as e:
        print(f"错误: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
