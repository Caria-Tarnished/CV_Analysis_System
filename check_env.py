#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境检查脚本
验证所有必需的Python库是否正确安装
"""

import sys
from typing import Dict, Any

def check_package_version(package_name: str, import_name: str = None) -> Dict[str, Any]:
    """
    检查包是否安装并获取版本信息
    
    Args:
        package_name: 包名
        import_name: 导入名（如果与包名不同）
    
    Returns:
        dict: 包含状态和版本信息的字典
    """
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'Unknown')
        return {
            'name': package_name,
            'status': 'OK',
            'version': version,
            'error': None
        }
    except ImportError as e:
        return {
            'name': package_name,
            'status': 'FAILED',
            'version': None,
            'error': str(e)
        }

def main():
    """主函数：检查所有必需的包"""
    print("=" * 60)
    print("计算机视觉情感分析系统 - 环境检查")
    print("=" * 60)
    print()
    
    # 需要检查的包列表
    packages_to_check = [
        ('OpenCV', 'cv2'),
        ('NumPy', 'numpy'),
        ('Matplotlib', 'matplotlib'),
        ('SciPy', 'scipy'),
        ('PyQt5', 'PyQt5'),
        ('PyTorch', 'torch'),
        ('TorchVision', 'torchvision'),
        ('dlib', 'dlib'),
        ('Pillow', 'PIL'),
        ('Scikit-learn', 'sklearn')
    ]
    
    results = []
    all_passed = True
    
    for package_name, import_name in packages_to_check:
        result = check_package_version(package_name, import_name)
        results.append(result)
        
        # 打印结果
        status = "?" if result['status'] == 'OK' else "?"
        version_info = f"v{result['version']}" if result['version'] else "未安装"
        
        print(f"{status} {package_name:<15} {version_info}")
        
        if result['status'] == 'FAILED':
            all_passed = False
            print(f"  错误: {result['error']}")
    
    print()
    print("-" * 60)
    
    # 检查PyTorch CUDA支持
    try:
        import torch
        if torch.cuda.is_available():
            print(f"? PyTorch CUDA支持: 可用 (设备数: {torch.cuda.device_count()})")
        else:
            print("! PyTorch CUDA支持: 不可用 (将使用CPU)")
    except:
        pass
    
    # 检查Python版本
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"? Python版本: {python_version}")
    
    print("-" * 60)
    
    if all_passed:
        print("? 所有依赖包检查通过！环境配置正确。")
        return 0
    else:
        print("? 部分依赖包缺失或安装失败。")
        print("请安装缺失的包：pip install <package_name>")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
