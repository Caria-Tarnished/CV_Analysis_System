#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H5模型转PyTorch模型工具
将Keras/TensorFlow的.h5模型转换为PyTorch的.pth格式
"""

import os
import sys
import argparse
import numpy as np
import torch
from emotion_recognizer import SimpleCNN

def convert_h5_to_pytorch(h5_path, output_path):
    """
    转换H5模型到PyTorch格式
    
    注意：这个转换假设.h5模型和SimpleCNN有相似的架构
    如果架构差异很大，可能需要手动调整
    
    Args:
        h5_path: .h5模型路径
        output_path: 输出的.pth模型路径
    """
    print("=" * 70)
    print("H5模型到PyTorch模型转换工具")
    print("=" * 70)
    print()
    
    # 检查文件
    if not os.path.exists(h5_path):
        print(f"错误: 找不到文件 {h5_path}")
        return False
    
    try:
        # 导入TensorFlow
        print("步骤1: 加载TensorFlow模型...")
        import tensorflow as tf
        
        # 加载H5模型
        keras_model = tf.keras.models.load_model(h5_path)
        print(f"? 成功加载H5模型: {h5_path}")
        
        # 显示模型信息
        print("\n模型架构:")
        keras_model.summary()
        
        # 创建PyTorch模型
        print("\n步骤2: 创建PyTorch模型...")
        pytorch_model = SimpleCNN(num_classes=7)
        print("? PyTorch模型已创建")
        
        # 权重转换
        print("\n步骤3: 转换权重...")
        print("??  注意: 自动权重转换可能不完美")
        print("   如果模型架构差异很大，建议重新训练")
        
        # 这里是一个简化的转换过程
        # 实际的转换可能需要根据具体的模型架构调整
        
        try:
            # 尝试提取权重
            keras_weights = keras_model.get_weights()
            print(f"? 提取了 {len(keras_weights)} 个权重张量")
            
            # 由于架构可能不同，我们不进行自动转换
            # 而是创建一个标记，提示用户这是从H5转换的
            print("\n??  警告: 自动权重映射较为复杂")
            print("   建议采用以下方案之一：")
            print("   1. 直接使用.h5模型（系统已支持）")
            print("   2. 使用FER2013数据集重新训练PyTorch模型")
            print("   3. 手动进行权重映射（需要深度学习知识）")
            
        except Exception as e:
            print(f"??  权重转换遇到问题: {e}")
        
        # 保存一个标记文件
        checkpoint = {
            'model_state_dict': pytorch_model.state_dict(),
            'model_type': 'SimpleCNN',
            'model_version': 'converted_from_h5',
            'num_classes': 7,
            'input_size': (48, 48),
            'source_model': h5_path,
            'note': 'This model was converted from H5 but weights are NOT mapped. Please use original H5 or retrain.',
            'conversion_status': 'architecture_only'
        }
        
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        torch.save(checkpoint, output_path)
        
        print(f"\n? 模型架构已保存到: {output_path}")
        print("\n" + "=" * 70)
        print("重要提示:")
        print("=" * 70)
        print("由于H5和PyTorch模型架构可能不同，自动权重转换不可靠。")
        print("\n推荐方案:")
        print("1. 直接使用原始.h5模型（系统支持，只需安装tensorflow）")
        print("2. 使用训练脚本重新训练PyTorch模型")
        print("\n如需直接使用.h5模型:")
        print("  - 安装: pip install tensorflow")
        print(f"  - 在代码中将模型路径设为: {h5_path}")
        print("=" * 70)
        
        return True
        
    except ImportError:
        print("错误: 未安装TensorFlow")
        print("请安装: pip install tensorflow")
        return False
        
    except Exception as e:
        print(f"转换失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_h5_model_directly(h5_path):
    """
    测试H5模型是否可以直接在系统中使用
    
    Args:
        h5_path: .h5模型路径
    """
    print("=" * 70)
    print("测试H5模型兼容性")
    print("=" * 70)
    print()
    
    try:
        from emotion_recognizer import EmotionRecognizer
        
        print("加载模型...")
        recognizer = EmotionRecognizer(model_path=h5_path)
        
        print("? 模型加载成功!")
        print("\n模型信息:")
        print(f"  - 模型类型: {recognizer.model_type}")
        print(f"  - 输入尺寸: {recognizer.input_size}")
        print(f"  - 运行设备: {recognizer.device}")
        
        # 创建测试输入
        print("\n进行测试推理...")
        test_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        test_bbox = (10, 10, 80, 80)
        
        emotion, confidence = recognizer.predict_with_confidence(test_frame, test_bbox)
        chinese_emotion = recognizer.get_chinese_label(emotion)
        
        print(f"? 测试推理成功!")
        print(f"  - 预测表情: {emotion} ({chinese_emotion})")
        print(f"  - 置信度: {confidence:.2%}")
        
        print("\n" + "=" * 70)
        print("结论: 此H5模型可以直接在系统中使用!")
        print("=" * 70)
        print("\n使用方法:")
        print("1. 确保已安装TensorFlow: pip install tensorflow")
        print(f"2. 在 main_gui.py 中设置模型路径为: {h5_path}")
        print("3. 启动系统: python run_system.py")
        
        return True
        
    except Exception as e:
        print(f"? 测试失败: {e}")
        print("\n可能的原因:")
        print("1. 未安装TensorFlow: pip install tensorflow")
        print("2. 模型架构不兼容")
        print("3. 模型文件损坏")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='H5模型转换和测试工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 测试H5模型是否可用
  python convert_h5_to_pth.py --test --h5_model models/emotion_model.h5
  
  # 转换H5到PyTorch（不推荐，建议直接使用H5）
  python convert_h5_to_pth.py --h5_model models/emotion_model.h5 --output models/emotion_model.pth
        """
    )
    
    parser.add_argument('--h5_model', type=str, required=True,
                       help='输入的.h5模型文件路径')
    parser.add_argument('--output', type=str, default=None,
                       help='输出的.pth模型文件路径（可选）')
    parser.add_argument('--test', action='store_true',
                       help='测试H5模型是否可以直接使用')
    
    args = parser.parse_args()
    
    if args.test:
        # 测试模式
        success = test_h5_model_directly(args.h5_model)
        return 0 if success else 1
    else:
        # 转换模式
        if args.output is None:
            args.output = args.h5_model.replace('.h5', '.pth')
        
        print("??  注意: 由于架构差异，自动转换通常不可靠")
        print("建议: 使用 --test 选项测试H5模型是否可直接使用\n")
        
        response = input("是否继续转换? (y/n): ").strip().lower()
        if response != 'y':
            print("已取消")
            return 0
        
        success = convert_h5_to_pytorch(args.h5_model, args.output)
        return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())

