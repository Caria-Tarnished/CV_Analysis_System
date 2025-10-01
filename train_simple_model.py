#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简易FER2013模型训练脚本
当无法下载预训练模型时，可以使用此脚本快速训练一个基础模型
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from emotion_recognizer import SimpleCNN
import pandas as pd
from tqdm import tqdm

class FER2013Dataset(Dataset):
    """FER2013数据集加载器"""
    
    def __init__(self, csv_file, train=True, transform=None):
        """
        Args:
            csv_file: FER2013 CSV文件路径
            train: 是否为训练集
            transform: 数据增强
        """
        self.data = pd.read_csv(csv_file)
        self.train = train
        self.transform = transform
        
        # 筛选训练集或测试集
        if train:
            self.data = self.data[self.data['Usage'] == 'Training']
        else:
            self.data = self.data[self.data['Usage'] == 'PublicTest']
        
        print(f"加载 {'训练' if train else '测试'}集: {len(self.data)} 个样本")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # 解析像素数据
        pixels = np.array([int(p) for p in row['pixels'].split()], dtype=np.uint8)
        image = pixels.reshape(48, 48)
        
        # 标签
        label = int(row['emotion'])
        
        # 数据增强
        if self.transform and self.train:
            image = self.transform(image)
        
        # 归一化
        image = image.astype(np.float32) / 255.0
        image = (image - 0.485) / 0.229
        
        # 转换为张量
        image = torch.from_numpy(image).unsqueeze(0)  # [1, 48, 48]
        
        return image, label

def simple_augment(image):
    """简单的数据增强"""
    # 随机水平翻转
    if np.random.random() > 0.5:
        image = cv2.flip(image, 1)
    
    # 随机亮度调整
    if np.random.random() > 0.5:
        factor = np.random.uniform(0.8, 1.2)
        image = np.clip(image * factor, 0, 255).astype(np.uint8)
    
    return image

def train_model(csv_file, epochs=10, batch_size=64, learning_rate=0.001):
    """
    训练模型
    
    Args:
        csv_file: FER2013数据集CSV文件路径
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
    """
    print("=" * 70)
    print("FER2013 表情识别模型训练")
    print("=" * 70)
    
    # 检查CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 创建数据集
    print("\n加载数据集...")
    train_dataset = FER2013Dataset(csv_file, train=True, transform=simple_augment)
    test_dataset = FER2013Dataset(csv_file, train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # 创建模型
    print("\n创建模型...")
    model = SimpleCNN(num_classes=7)
    model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    # 训练循环
    best_acc = 0.0
    print(f"\n开始训练 (共 {epochs} 轮)...")
    print("-" * 70)
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [训练]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*train_correct/train_total:.2f}%'})
        
        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # 测试阶段
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [测试]")
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*test_correct/test_total:.2f}%'})
        
        test_acc = 100. * test_correct / test_total
        avg_test_loss = test_loss / len(test_loader)
        
        # 打印结果
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  训练 - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  测试 - Loss: {avg_test_loss:.4f}, Acc: {test_acc:.2f}%")
        
        # 学习率调整
        scheduler.step(test_acc)
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            print(f"新的最佳准确率! 保存模型...")
            
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'model_type': 'SimpleCNN',
                'model_version': 'v1_trained',
                'num_classes': 7,
                'input_size': (48, 48),
                'epoch': epoch + 1,
                'test_accuracy': test_acc,
                'train_accuracy': train_acc,
                'created_by': 'train_simple_model.py'
            }
            
            os.makedirs('models', exist_ok=True)
            torch.save(checkpoint, 'models/emotion_model.pth')
        
        print("-" * 70)
    
    print(f"\n训练完成!")
    print(f"最佳测试准确率: {best_acc:.2f}%")
    print(f"模型已保存到: models/emotion_model.pth")
    
    return model

def quick_train(csv_file=None):
    """快速训练（较少轮数，用于快速测试）"""
    if csv_file is None:
        print("错误: 请提供FER2013数据集CSV文件路径")
        print("\n如何获取数据集:")
        print("1. 访问 Kaggle: https://www.kaggle.com/datasets/msambare/fer2013")
        print("2. 下载 fer2013.csv 文件")
        print("3. 运行: python train_simple_model.py --csv_file path/to/fer2013.csv")
        return None
    
    if not os.path.exists(csv_file):
        print(f"错误: 找不到文件 {csv_file}")
        return None
    
    print("快速训练模式 (5轮)")
    print("这将创建一个基础模型，准确率约 50-60%")
    print("如需更好效果，建议训练更多轮数 (20-30轮)")
    print()
    
    return train_model(csv_file, epochs=5, batch_size=64, learning_rate=0.001)

def full_train(csv_file=None):
    """完整训练（更多轮数，更好效果）"""
    if csv_file is None:
        print("错误: 请提供FER2013数据集CSV文件路径")
        return None
    
    if not os.path.exists(csv_file):
        print(f"错误: 找不到文件 {csv_file}")
        return None
    
    print("完整训练模式 (20轮)")
    print("这将需要较长时间，但能获得更好的模型效果")
    print()
    
    return train_model(csv_file, epochs=20, batch_size=64, learning_rate=0.001)

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='FER2013表情识别模型训练')
    parser.add_argument('--csv_file', type=str, default=None, help='FER2013数据集CSV文件路径')
    parser.add_argument('--mode', type=str, default='quick', choices=['quick', 'full'], 
                       help='训练模式: quick(快速5轮) 或 full(完整20轮)')
    parser.add_argument('--epochs', type=int, default=None, help='自定义训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    
    args = parser.parse_args()
    
    if args.csv_file is None:
        print("\n" + "=" * 70)
        print("FER2013 模型训练工具")
        print("=" * 70)
        print("\n使用说明:")
        print("1. 从Kaggle下载FER2013数据集")
        print("   地址: https://www.kaggle.com/datasets/msambare/fer2013")
        print("\n2. 运行训练命令:")
        print("   快速训练(5轮): python train_simple_model.py --csv_file fer2013.csv --mode quick")
        print("   完整训练(20轮): python train_simple_model.py --csv_file fer2013.csv --mode full")
        print("\n3. 等待训练完成，模型将保存到 models/emotion_model.pth")
        print("\n" + "=" * 70)
        return
    
    if not os.path.exists(args.csv_file):
        print(f"错误: 找不到文件 {args.csv_file}")
        return
    
    # 执行训练
    if args.epochs:
        train_model(args.csv_file, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
    elif args.mode == 'quick':
        quick_train(args.csv_file)
    else:
        full_train(args.csv_file)

if __name__ == "__main__":
    main()

