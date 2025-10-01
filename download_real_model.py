#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预训练模型下载助手
帮助用户下载真实的FER2013预训练模型
"""

import os
import sys
import torch
from download_models import SimpleCNN, download_or_create_model

def print_banner():
    """打印横幅"""
    print("=" * 70)
    print(" " * 20 + "表情识别预训练模型下载助手")
    print("=" * 70)
    print()

def show_model_sources():
    """显示可用的模型下载源"""
    print("📦 可用的预训练模型下载源：\n")
    
    sources = [
        {
            "name": "GitHub - FER2013 PyTorch",
            "url": "https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch",
            "description": "流行的FER2013 PyTorch实现",
            "steps": [
                "1. 访问上述GitHub仓库",
                "2. 下载预训练模型文件 (通常是.pth或.pt文件)",
                "3. 将文件重命名为 'emotion_model.pth'",
                "4. 放到 'models' 目录下"
            ]
        },
        {
            "name": "Hugging Face - Emotion Recognition",
            "url": "https://huggingface.co/models?search=fer2013",
            "description": "Hugging Face模型库中的FER2013模型",
            "steps": [
                "1. 访问Hugging Face并搜索 'fer2013'",
                "2. 选择合适的PyTorch模型",
                "3. 下载模型权重文件",
                "4. 按照模型说明进行格式转换（如需要）",
                "5. 保存为 'models/emotion_model.pth'"
            ]
        },
        {
            "name": "Google Drive - 共享模型",
            "url": "搜索: FER2013 pretrained model site:drive.google.com",
            "description": "研究人员分享的预训练模型",
            "steps": [
                "1. 在Google搜索中使用上述搜索词",
                "2. 找到可信的共享链接",
                "3. 下载模型文件",
                "4. 验证文件完整性",
                "5. 保存为 'models/emotion_model.pth'"
            ]
        },
        {
            "name": "百度网盘 - 国内资源",
            "url": "搜索: FER2013 预训练模型 百度网盘",
            "description": "国内分享的模型资源（速度较快）",
            "steps": [
                "1. 在百度搜索相关关键词",
                "2. 找到可靠的分享链接",
                "3. 下载模型文件",
                "4. 保存为 'models/emotion_model.pth'"
            ]
        }
    ]
    
    for i, source in enumerate(sources, 1):
        print(f"[{i}] {source['name']}")
        print(f"    🔗 {source['url']}")
        print(f"    📝 {source['description']}")
        print(f"    \n    下载步骤：")
        for step in source['steps']:
            print(f"    {step}")
        print()

def check_current_model():
    """检查当前模型状态"""
    model_path = "models/emotion_model.pth"
    
    print("\n" + "=" * 70)
    print("📁 当前模型状态检查")
    print("=" * 70 + "\n")
    
    if os.path.exists(model_path):
        print(f"✓ 发现模型文件: {model_path}")
        
        try:
            # 加载并检查模型
            checkpoint = torch.load(model_path, map_location='cpu')
            
            if isinstance(checkpoint, dict):
                print(f"✓ 模型类型: {checkpoint.get('model_type', '未知')}")
                print(f"✓ 模型版本: {checkpoint.get('model_version', 'v1')}")
                
                if 'created_by' in checkpoint:
                    print(f"✓ 创建工具: {checkpoint['created_by']}")
                    if checkpoint['created_by'] == 'download_models.py':
                        print("  ⚠️  这是一个模拟模型，建议下载真实的预训练模型以获得更好效果")
                    
                if 'note' in checkpoint:
                    print(f"✓ 备注: {checkpoint['note']}")
                    
                # 检查模型大小
                file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
                print(f"✓ 文件大小: {file_size:.2f} MB")
                
                if file_size < 1:
                    print("  ⚠️  文件较小，可能不是完整的预训练模型")
                else:
                    print("  ✓ 文件大小合理")
                    
            else:
                print("✓ 模型格式: 直接保存的模型对象")
                file_size = os.path.getsize(model_path) / (1024 * 1024)
                print(f"✓ 文件大小: {file_size:.2f} MB")
                
        except Exception as e:
            print(f"✗ 模型文件可能损坏: {e}")
            
    else:
        print(f"✗ 未找到模型文件: {model_path}")
        print("  需要下载或创建模型")

def create_improved_model():
    """创建改进的模型"""
    print("\n" + "=" * 70)
    print("🔧 创建改进版模拟模型")
    print("=" * 70 + "\n")
    
    print("正在创建改进的模拟训练模型...")
    print("注意: 这不是真实的预训练模型，但比随机权重效果好")
    print()
    
    model_path = download_or_create_model()
    print(f"\n✓ 模型已保存到: {model_path}")
    
    return model_path

def show_integration_instructions():
    """显示如何集成下载的模型"""
    print("\n" + "=" * 70)
    print("📖 模型集成说明")
    print("=" * 70 + "\n")
    
    print("将下载的模型文件放到正确位置后，系统会自动使用它：")
    print()
    print("1. 确保模型文件命名为: emotion_model.pth")
    print("2. 放到目录: CV_Analysis_System/models/")
    print("3. 完整路径应为: CV_Analysis_System/models/emotion_model.pth")
    print()
    print("如果模型格式不兼容，可能需要转换：")
    print()
    print("```python")
    print("import torch")
    print("from emotion_recognizer import SimpleCNN")
    print()
    print("# 加载你的模型")
    print("model = SimpleCNN()")
    print("# 加载权重（根据实际格式调整）")
    print("model.load_state_dict(torch.load('your_model.pth'))")
    print()
    print("# 保存为标准格式")
    print("checkpoint = {")
    print("    'model_state_dict': model.state_dict(),")
    print("    'model_type': 'SimpleCNN',")
    print("    'num_classes': 7")
    print("}")
    print("torch.save(checkpoint, 'models/emotion_model.pth')")
    print("```")

def main():
    """主函数"""
    print_banner()
    
    while True:
        print("\n请选择操作：")
        print("[1] 查看预训练模型下载源")
        print("[2] 检查当前模型状态")
        print("[3] 创建改进版模拟模型（临时方案）")
        print("[4] 查看模型集成说明")
        print("[0] 退出")
        print()
        
        choice = input("请输入选项 (0-4): ").strip()
        
        if choice == '1':
            show_model_sources()
            
        elif choice == '2':
            check_current_model()
            
        elif choice == '3':
            create_improved_model()
            
        elif choice == '4':
            show_integration_instructions()
            
        elif choice == '0':
            print("\n再见！")
            break
    
    else:
            print("无效选项，请重新选择")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
