#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算机视觉情感与心率分析系统 - 启动脚本
提供多种启动模式选择
"""

import sys
import os
import argparse
from pathlib import Path  # noqa: F401 (kept for potential future use)

def check_dependencies():
    """检查系统依赖"""
    print("检查系统依赖...")
    
    required_modules = [
        'cv2', 'numpy', 'scipy', 'matplotlib', 
        'PyQt5', 'torch', 'torchvision'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"? {module}")
        except ImportError:
            print(f"? {module} - 未安装")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\n缺少以下依赖模块: {', '.join(missing_modules)}")
        print("请运行以下命令安装:")
        print("pip install opencv-python numpy scipy matplotlib PyQt5 torch torchvision")
        return False
    
    print("? 所有依赖检查通过\n")
    return True

def check_camera():
    """检查摄像头可用性"""
    print("检查摄像头...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("? 摄像头工作正常")
                cap.release()
                return True
            else:
                print("? 摄像头无法读取图像")
                cap.release()
                return False
        else:
            print("? 无法打开摄像头")
            return False
            
    except Exception as e:
        print(f"? 摄像头检查失败: {e}")
        return False

def run_environment_check():
    """运行环境检查"""
    print("=" * 60)
    print("环境检查")
    print("=" * 60)
    
    try:
        from check_env import main as check_main
        return check_main() == 0
    except ImportError:
        print("? 找不到check_env.py，请确保文件存在")
        return False

def run_gui_mode():
    """运行GUI模式"""
    print("启动GUI模式...")
    
    try:
        from main_gui import main as gui_main
        gui_main()
    except ImportError as e:
        print(f"? 无法导入GUI模块: {e}")
        print("请确保PyQt5已正确安装")
        return False
    except Exception as e:
        print(f"? GUI运行错误: {e}")
        return False
    
    return True

def run_test_mode(test_type):
    """运行测试模式"""
    print(f"启动测试模式: {test_type}")
    
    if test_type == 'video':
        try:
            from test_video import main as test_main
            return test_main() == 0
        except ImportError:
            print("? 找不到test_video.py")
            return False
            
    elif test_type == 'heart_rate':
        try:
            from test_heart_rate import main as test_main
            return test_main() == 0
        except ImportError:
            print("? 找不到test_heart_rate.py")
            return False
            
    else:
        print(f"? 不支持的测试类型: {test_type}")
        return False

def run_demo_mode():
    """运行演示模式（综合功能展示）"""
    print("启动演示模式...")
    print("演示模式将依次展示各项功能")
    
    try:
        import cv2
        import time
        from video_handler import VideoStreamHandler
        from face_detector import FaceDetector
        from emotion_recognizer import EmotionRecognizer
        from heart_rate_estimator import HeartRateEstimator
        from statistics_tracker import StatisticsTracker
        from text_renderer import draw_text_cn
        
        print("\n1. 初始化各组件...")
        video_handler = VideoStreamHandler(source=0)
        face_detector = FaceDetector(method='haar')
        
        # 检查并下载表情识别模型
        model_path = "models/emotion_model.pth"
        if not os.path.exists(model_path):
            print("首次运行，正在创建表情识别模型...")
            try:
                from download_models import download_or_create_model
                model_path = download_or_create_model()
            except Exception as e:
                print(f"模型创建失败: {e}，使用默认模型")
                model_path = None
        
        emotion_recognizer = EmotionRecognizer(model_path=model_path, confidence_threshold=0.2)
        hr_estimator = HeartRateEstimator()
        stats_tracker = StatisticsTracker()
        
        print("2. 启动视频流...")
        if not video_handler.start():
            print("? 无法启动摄像头")
            return False
        
        print("3. 开始综合检测...")
        print("按 'q' 退出演示")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            frame = video_handler.read()
            if frame is None:
                continue
                
            frame_count += 1
            display_frame = frame.copy()
            
            # 人脸检测
            faces = face_detector.detect(frame)
            stats_tracker.update_frame_stats(len(faces))
            
            # 处理检测到的人脸
            for i, face_bbox in enumerate(faces):
                x, y, w, h = face_bbox
                
                # 绘制人脸框
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(display_frame, f'Face {i+1}', (x, y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 表情识别
                emotion = emotion_recognizer.predict(frame, face_bbox)
                chinese_emotion = emotion_recognizer.get_chinese_label(emotion)
                stats_tracker.update_emotion(emotion)
                
                # 使用中文渲染，避免乱码
                display_frame = draw_text_cn(
                    display_frame, chinese_emotion, (x, y + h + 25),
                    color=(255, 0, 0), font_size=22, stroke_width=1, stroke_fill=(0,0,0)
                )
                
                # 心率估计（仅处理第一个人脸）
                if i == 0:
                    hr_estimator.process_frame(frame, face_bbox)
                    
                    if frame_count % 30 == 0:  # 每30帧估计一次
                        bpm = hr_estimator.estimate_bpm()
                        if bpm:
                            stats_tracker.update_heart_rate(bpm)
                            cv2.putText(display_frame, f'HR: {bpm:.1f} BPM', 
                                      (x, y + h + 50),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # 显示统计信息
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            
            info_y = 30
            # 使用中文渲染，避免 OpenCV 中文乱码
            display_frame = draw_text_cn(
                display_frame, f"FPS: {fps:.1f}", (10, info_y),
                color=(255, 255, 255), font_size=20, stroke_width=1, stroke_fill=(0,0,0)
            )
            info_y += 25
            
            display_frame = draw_text_cn(
                display_frame, f"人脸: {len(faces)}", (10, info_y),
                color=(255, 255, 255), font_size=20, stroke_width=1, stroke_fill=(0,0,0)
            )
            info_y += 25
            
            emotion_counts = stats_tracker.get_emotion_counts()
            if emotion_counts:
                most_common = max(emotion_counts.items(), key=lambda x: x[1])
                chinese_most = emotion_recognizer.get_chinese_label(most_common[0])
                display_frame = draw_text_cn(
                    display_frame, f"最高频表情: {chinese_most}", (10, info_y),
                    color=(255, 255, 255), font_size=20, stroke_width=1, stroke_fill=(0,0,0)
                )
            
            # 显示图像
            cv2.imshow('CV Analysis System Demo', display_frame)
            
            # 检查退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 显示最终统计
        print("\n" + "=" * 50)
        print("演示结果摘要")
        print("=" * 50)
        stats_tracker.print_summary()
        
        # 清理资源
        video_handler.stop()
        cv2.destroyAllWindows()
        
        return True
        
    except Exception as e:
        print(f"? 演示模式运行错误: {e}")
        return False

def print_banner():
    """打印系统信息横幅"""
    # 使用纯ASCII边框，避免cmd/PowerShell控制台乱码
    banner = (
        "\n" +
        "+--------------------------------------------------------------+\n"
        "|                                                              |\n"
        "|  计算机视觉情感与心率分析系统                               |\n"
        "|  Computer Vision Emotion & Heart Rate Analysis System        |\n"
        "|                                                              |\n"
        "|  版本: 1.0.0                                                 |\n"
        "|  作者: Caria_Tarnished                                        |\n"
        "|                                                              |\n"
        "+--------------------------------------------------------------+\n"
    )
    print(banner)

def main():
    """主函数"""
    print_banner()
    
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='CV情感心率分析系统')
    parser.add_argument('--mode', choices=['gui', 'test', 'demo', 'check'], 
                       default='gui', help='运行模式')
    parser.add_argument('--test-type', choices=['video', 'heart_rate'], 
                       default='video', help='测试类型（仅在test模式下有效）')
    parser.add_argument('--skip-check', action='store_true', 
                       help='跳过环境检查')
    
    args = parser.parse_args()
    
    # 环境检查
    if not args.skip_check:
        if not check_dependencies():
            return 1
            
        if not check_camera():
            print("?? 摄像头检查失败，某些功能可能无法正常使用")
            input("按回车键继续...")
    
    # 根据模式运行
    success = False
    
    if args.mode == 'check':
        success = run_environment_check()
        
    elif args.mode == 'gui':
        success = run_gui_mode()
        
    elif args.mode == 'test':
        success = run_test_mode(args.test_type)
        
    elif args.mode == 'demo':
        success = run_demo_mode()
    
    if success:
        print("\n> 程序正常结束")
        return 0
    else:
        print("\n! 程序异常结束")
        return 1

def interactive_mode():
    """交互式启动模式"""
    print_banner()
    
    while True:
        print("\n请选择运行模式:")
        print("1. GUI模式 - 完整的图形界面程序")
        print("2. 演示模式 - 综合功能演示")
        print("3. 测试模式 - 单独测试各组件")
        print("4. 环境检查 - 检查系统依赖")
        print("5. 退出")
        
        try:
            choice = input("\n请输入选择 (1-5): ").strip()
            
            if choice == '1':
                run_gui_mode()
                
            elif choice == '2':
                run_demo_mode()
                
            elif choice == '3':
                print("\n选择测试类型:")
                print("1. 视频流和人脸检测测试")
                print("2. 心率估计测试")
                
                test_choice = input("请输入选择 (1-2): ").strip()
                if test_choice == '1':
                    run_test_mode('video')
                elif test_choice == '2':
                    run_test_mode('heart_rate')
                else:
                    print("无效选择")
                    
            elif choice == '4':
                run_environment_check()
                
            elif choice == '5':
                print("退出程序")
                break
                
            else:
                print("无效选择，请重新输入")
                
        except KeyboardInterrupt:
            print("\n\n收到中断信号，退出程序")
            break
        except Exception as e:
            print(f"运行错误: {e}")

if __name__ == '__main__':
    # 如果没有命令行参数，运行交互模式
    if len(sys.argv) == 1:
        interactive_mode()
    else:
        sys.exit(main())
