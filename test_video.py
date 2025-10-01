#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频流测试脚本
测试VideoStreamHandler和FaceDetector的集成功能
"""

import cv2
import time
import sys
from video_handler import VideoStreamHandler
from text_renderer import draw_text_cn

def test_video_stream():
    """测试基础视频流功能"""
    print("=" * 50)
    print("测试视频流处理器")
    print("=" * 50)
    
    # 创建视频流处理器
    video_handler = VideoStreamHandler(source=0)  # 使用摄像头
    
    # 启动视频流
    if not video_handler.start():
        print("? 无法启动视频流")
        return False
        
    print("? 视频流启动成功")
    print("按 'q' 键退出测试...")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # 读取帧
            frame = video_handler.read()
            
            if frame is not None:
                frame_count += 1
                
                # 获取视频信息
                info = video_handler.get_frame_info()
                
                # 在帧上添加信息文本（中文通过 Pillow 绘制）
                frame = draw_text_cn(frame, f"FPS: {info['actual_fps']:.1f}", (10, 30), color=(0, 255, 0), font_size=22, stroke_width=1, stroke_fill=(0,0,0))
                frame = draw_text_cn(frame, f"分辨率: {info['width']}x{info['height']}", (10, 60), color=(0, 255, 0), font_size=22, stroke_width=1, stroke_fill=(0,0,0))
                frame = draw_text_cn(frame, f"缓冲区: {info['buffer_size']}", (10, 90), color=(0, 255, 0), font_size=22, stroke_width=1, stroke_fill=(0,0,0))
                frame = draw_text_cn(frame, f"帧数: {frame_count}", (10, 120), color=(0, 255, 0), font_size=22, stroke_width=1, stroke_fill=(0,0,0))
                
                # 添加中心十字线
                h, w = frame.shape[:2]
                cv2.line(frame, (w//2-20, h//2), (w//2+20, h//2), (255, 0, 0), 2)
                cv2.line(frame, (w//2, h//2-20), (w//2, h//2+20), (255, 0, 0), 2)
                
                # 显示图像
                # 使用ASCII窗口标题以避免Windows上标题栏乱码
                cv2.imshow('Video Stream Test', frame)
                
            else:
                print("?? 无法读取帧")
                
            # 检查退出键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # 空格键暂停
                print("视频已暂停，按任意键继续...")
                cv2.waitKey(0)
                
    except KeyboardInterrupt:
        print("\n收到中断信号")
        
    finally:
        # 计算统计信息
        elapsed_time = time.time() - start_time
        avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        print(f"\n测试统计:")
        print(f"总帧数: {frame_count}")
        print(f"运行时间: {elapsed_time:.2f}秒")
        print(f"平均FPS: {avg_fps:.2f}")
        
        # 清理资源
        video_handler.stop()
        cv2.destroyAllWindows()
        
    return True

def test_video_with_face_detection():
    """测试视频流与人脸检测的集成"""
    print("=" * 50)
    print("测试视频流 + 人脸检测")
    print("=" * 50)
    
    try:
        from face_detector import FaceDetector
    except ImportError:
        print("?? FaceDetector未实现，跳过人脸检测测试")
        return test_video_stream()
    
    # 创建视频流处理器和人脸检测器
    video_handler = VideoStreamHandler(source=0)
    face_detector = FaceDetector(method='haar')  # 使用Haar级联，较快
    
    # 启动视频流
    if not video_handler.start():
        print("? 无法启动视频流")
        return False
        
    print("? 视频流和人脸检测器初始化成功")
    print("按 'q' 键退出，按 'f' 切换人脸检测...")
    
    face_detection_enabled = True
    frame_count = 0
    face_count = 0
    
    try:
        while True:
            # 读取帧
            frame = video_handler.read()
            
            if frame is not None:
                frame_count += 1
                display_frame = frame.copy()
                
                # 人脸检测
                if face_detection_enabled:
                    faces = face_detector.detect(frame)
                    face_count = len(faces)
                    
                    # 绘制人脸框
                    for (x, y, w, h) in faces:
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        cv2.putText(display_frame, 'Face', (x, y - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # 添加信息文本
                info = video_handler.get_frame_info()
                font = cv2.FONT_HERSHEY_SIMPLEX
                y_offset = 30
                
                # 使用中文渲染，避免乱码
                texts = [
                    (f"FPS: {info['actual_fps']:.1f}", (0, 255, 0)),
                    (f"人脸检测: {'开启' if face_detection_enabled else '关闭'}", (0, 255, 0)),
                    (f"检测到人脸: {face_count}", (0, 255, 0)),
                    (f"帧数: {frame_count}", (0, 255, 0)),
                    ("按 'f' 切换检测, 'q' 退出", (255, 255, 255)),
                ]

                for i, (text, color) in enumerate(texts):
                    display_frame = draw_text_cn(
                        display_frame, text, (10, y_offset + i * 30),
                        color=color, font_size=20, stroke_width=1, stroke_fill=(0,0,0)
                    )
                
                # 显示图像
                cv2.imshow('Video + Face Detection Test', display_frame)
                
            # 检查按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                face_detection_enabled = not face_detection_enabled
                print(f"人脸检测已{'开启' if face_detection_enabled else '关闭'}")
                
    except KeyboardInterrupt:
        print("\n收到中断信号")
        
    finally:
        # 清理资源
        video_handler.stop()
        cv2.destroyAllWindows()
        
    return True

def main():
    """主函数"""
    print("视频处理模块测试")
    print("请确保摄像头已连接并可用")
    print()
    
    # 检查摄像头是否可用
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("? 无法打开摄像头，请检查摄像头连接")
        print("提示: 请确保没有其他程序正在使用摄像头")
        return 1
    cap.release()
    
    while True:
        print("\n选择测试模式:")
        print("1. 基础视频流测试")
        print("2. 视频流 + 人脸检测测试")
        print("3. 退出")
        
        try:
            choice = input("请输入选择 (1-3): ").strip()
            
            if choice == '1':
                test_video_stream()
            elif choice == '2':
                test_video_with_face_detection()
            elif choice == '3':
                print("退出测试")
                break
            else:
                print("无效选择，请重新输入")
                
        except KeyboardInterrupt:
            print("\n\n退出测试")
            break
        except Exception as e:
            print(f"测试过程中发生错误: {e}")
            
    return 0

if __name__ == "__main__":
    sys.exit(main())
