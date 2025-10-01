#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
心率估计测试脚本
结合视频流、人脸检测和心率估计的综合测试
"""

import cv2
import time
import sys
import numpy as np
from video_handler import VideoStreamHandler
from text_renderer import draw_text_cn
from face_detector import FaceDetector
from heart_rate_estimator import HeartRateEstimator

def test_heart_rate_estimation():
    """测试实时心率估计"""
    print("=" * 60)
    print("实时心率估计测试")
    print("=" * 60)
    print()
    print("使用说明:")
    print("- 请保持面部在摄像头前，尽量减少移动")
    print("- 确保光线充足且稳定")
    print("- 心率估计需要一段时间来积累数据")
    print("- 按 'q' 退出测试")
    print("- 按 'r' 重置心率估计器")
    print("- 按 's' 保存信号图")
    print()
    
    # 初始化组件
    video_handler = VideoStreamHandler(source=0)
    face_detector = FaceDetector(method='haar')
    hr_estimator = HeartRateEstimator(buffer_size=180, fps=30)  # 6秒缓冲区
    
    # 启动视频流
    if not video_handler.start():
        print("? 无法启动视频流")
        return False
        
    print("? 系统初始化完成，开始心率检测...")
    print()
    
    # 统计变量
    frame_count = 0
    last_bpm_time = time.time()
    bpm_history = []
    quality_history = []
    
    # ROI可视化设置
    show_roi = True
    roi_method = 'forehead'
    hr_estimator.set_roi_method(roi_method)
    
    try:
        while True:
            # 读取帧
            frame = video_handler.read()
            
            if frame is None:
                time.sleep(0.01)
                continue
                
            frame_count += 1
            display_frame = frame.copy()
            
            # 检测人脸
            faces = face_detector.detect(frame)
            
            current_bpm = None
            signal_quality = 0.0
            
            if len(faces) > 0:
                # 使用第一个检测到的人脸
                face_bbox = faces[0]
                x, y, w, h = face_bbox
                
                # 绘制人脸框
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(display_frame, 'Face', (x, y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 绘制ROI区域
                if show_roi:
                    roi_x, roi_y, roi_w, roi_h = hr_estimator._get_roi_coordinates(face_bbox)
                    cv2.rectangle(display_frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), 
                                (255, 0, 0), 2)
                    cv2.putText(display_frame, f'ROI ({roi_method})', (roi_x, roi_y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # 处理帧进行心率估计
                success = hr_estimator.process_frame(frame, face_bbox)
                
                if success:
                    # 获取信号质量
                    signal_quality = hr_estimator.get_signal_quality()
                    
                    # 每15帧尝试估计一次心率
                    if frame_count % 15 == 0:
                        current_bpm = hr_estimator.estimate_bpm()
                        
                        if current_bpm is not None:
                            bpm_history.append(current_bpm)
                            quality_history.append(signal_quality)
                            last_bpm_time = time.time()
                            
                            # 保持历史记录在合理长度
                            if len(bpm_history) > 20:
                                bpm_history.pop(0)
                                quality_history.pop(0)
                                
                            print(f"心率: {current_bpm:.1f} BPM, 质量: {signal_quality:.2f}")
            
            # 显示信息
            info_y = 30
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # 基本信息（中文使用 Pillow 绘制，避免乱码）
            display_frame = draw_text_cn(
                display_frame,
                f"帧数: {frame_count}",
                (10, info_y),
                color=(255, 255, 255),
                font_size=20,
                stroke_width=1,
                stroke_fill=(0, 0, 0),
            )
            info_y += 25
            
            display_frame = draw_text_cn(
                display_frame,
                f"检测到人脸: {len(faces)}",
                (10, info_y),
                color=(255, 255, 255),
                font_size=20,
                stroke_width=1,
                stroke_fill=(0, 0, 0),
            )
            info_y += 25
            
            # 心率信息
            if current_bpm is not None:
                color = (0, 255, 0) if signal_quality > 0.3 else (0, 255, 255)
                display_frame = draw_text_cn(
                    display_frame,
                    f"心率: {current_bpm:.1f} BPM",
                    (10, info_y),
                    color=color,
                    font_size=22,
                    stroke_width=1,
                    stroke_fill=(0, 0, 0),
                )
            else:
                if len(faces) > 0:
                    display_frame = draw_text_cn(
                        display_frame,
                        "心率: 计算中...",
                        (10, info_y),
                        color=(0, 255, 255),
                        font_size=22,
                        stroke_width=1,
                        stroke_fill=(0, 0, 0),
                    )
                else:
                    display_frame = draw_text_cn(
                        display_frame,
                        "心率: 需要检测人脸",
                        (10, info_y),
                        color=(0, 0, 255),
                        font_size=22,
                        stroke_width=1,
                        stroke_fill=(0, 0, 0),
                    )
            info_y += 25
            
            # 信号质量
            quality_color = (0, 255, 0) if signal_quality > 0.5 else (0, 255, 255) if signal_quality > 0.2 else (0, 0, 255)
            display_frame = draw_text_cn(
                display_frame,
                f"信号质量: {signal_quality:.2f}",
                (10, info_y),
                color=quality_color,
                font_size=20,
                stroke_width=1,
                stroke_fill=(0, 0, 0),
            )
            info_y += 25
            
            # 平均心率
            if len(bpm_history) > 0:
                avg_bpm = np.mean(bpm_history[-10:])  # 最近10次的平均值
                display_frame = draw_text_cn(
                    display_frame,
                    f"平均心率: {avg_bpm:.1f} BPM",
                    (10, info_y),
                    color=(255, 255, 255),
                    font_size=20,
                    stroke_width=1,
                    stroke_fill=(0, 0, 0),
                )
                info_y += 25
            
            # 缓冲区状态
            buffer_fill = len(hr_estimator.signal_buffer) / hr_estimator.buffer_size
            display_frame = draw_text_cn(
                display_frame,
                f"缓冲区: {buffer_fill*100:.0f}%",
                (10, info_y),
                color=(255, 255, 255),
                font_size=20,
                stroke_width=1,
                stroke_fill=(0, 0, 0),
            )
            info_y += 25
            
            # 操作提示
            display_frame = draw_text_cn(
                display_frame,
                "q-退出  r-重置  s-保存信号图",
                (10, display_frame.shape[0] - 20),
                color=(255, 255, 255),
                font_size=18,
                stroke_width=1,
                stroke_fill=(0, 0, 0),
            )
            
            # 绘制心率趋势图（右上角）
            if len(bpm_history) > 1:
                graph_x, graph_y = display_frame.shape[1] - 200, 50
                graph_w, graph_h = 180, 100
                
                # 绘制图表背景
                cv2.rectangle(display_frame, (graph_x, graph_y), 
                            (graph_x + graph_w, graph_y + graph_h), (50, 50, 50), -1)
                cv2.rectangle(display_frame, (graph_x, graph_y), 
                            (graph_x + graph_w, graph_y + graph_h), (255, 255, 255), 1)
                
                # 绘制心率曲线
                if len(bpm_history) > 1:
                    points = []
                    min_bpm = min(bpm_history)
                    max_bpm = max(bpm_history)
                    bpm_range = max(max_bpm - min_bpm, 10)  # 至少10 BPM的范围
                    
                    for i, bpm in enumerate(bpm_history[-15:]):  # 最近15个点
                        px = graph_x + int(i * graph_w / 14)
                        py = graph_y + graph_h - int((bpm - min_bpm) / bpm_range * graph_h)
                        points.append((px, py))
                    
                    # 绘制折线
                    for i in range(1, len(points)):
                        cv2.line(display_frame, points[i-1], points[i], (0, 255, 0), 2)
                
                # 图表标题
                cv2.putText(display_frame, "Heart Rate", (graph_x, graph_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # 显示图像
            # 使用ASCII窗口标题避免标题栏乱码
            cv2.imshow('Heart Rate Estimation Test', display_frame)
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("重置心率估计器...")
                hr_estimator.reset()
                bpm_history.clear()
                quality_history.clear()
            elif key == ord('s'):
                print("保存信号图...")
                hr_estimator.plot_signal(f"heart_rate_signal_{int(time.time())}.png")
            elif key == ord('1'):
                roi_method = 'forehead'
                hr_estimator.set_roi_method(roi_method)
                print(f"ROI方法设置为: {roi_method}")
            elif key == ord('2'):
                roi_method = 'cheeks'
                hr_estimator.set_roi_method(roi_method)
                print(f"ROI方法设置为: {roi_method}")
            elif key == ord('3'):
                roi_method = 'full_face'
                hr_estimator.set_roi_method(roi_method)
                print(f"ROI方法设置为: {roi_method}")
                
    except KeyboardInterrupt:
        print("\n收到中断信号")
        
    finally:
        # 打印测试结果摘要
        print("\n" + "=" * 50)
        print("测试结果摘要")
        print("=" * 50)
        
        if len(bpm_history) > 0:
            avg_bpm = np.mean(bpm_history)
            std_bpm = np.std(bpm_history)
            min_bpm = np.min(bpm_history)
            max_bpm = np.max(bpm_history)
            
            print(f"测量次数: {len(bpm_history)}")
            print(f"平均心率: {avg_bpm:.1f} ± {std_bpm:.1f} BPM")
            print(f"心率范围: {min_bpm:.1f} - {max_bpm:.1f} BPM")
            
            avg_quality = np.mean(quality_history) if quality_history else 0
            print(f"平均信号质量: {avg_quality:.2f}")
        else:
            print("未获得有效的心率测量数据")
            
        print(f"总处理帧数: {frame_count}")
        
        # 清理资源
        video_handler.stop()
        cv2.destroyAllWindows()
        
    return True

def _get_roi_coordinates(hr_estimator, face_bbox):
    """辅助函数：获取ROI坐标用于可视化"""
    x, y, w, h = face_bbox
    
    if hr_estimator.roi_method == 'forehead':
        roi_x = x + w // 6
        roi_y = y
        roi_w = w * 2 // 3
        roi_h = h // 3
    elif hr_estimator.roi_method == 'cheeks':
        roi_x = x + w // 6
        roi_y = y + h // 3
        roi_w = w * 2 // 3
        roi_h = h // 3
    else:  # full_face
        roi_x, roi_y, roi_w, roi_h = x, y, w, h
        
    return roi_x, roi_y, roi_w, roi_h

# 将辅助函数添加到HeartRateEstimator类
HeartRateEstimator._get_roi_coordinates = _get_roi_coordinates

def main():
    """主函数"""
    print("心率估计测试程序")
    print("请确保摄像头已连接并可用")
    print()
    
    # 检查摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("? 无法打开摄像头，请检查摄像头连接")
        return 1
    cap.release()
    
    while True:
        print("选择测试模式:")
        print("1. 实时心率估计测试")
        print("2. 退出")
        
        try:
            choice = input("请输入选择 (1-2): ").strip()
            
            if choice == '1':
                test_heart_rate_estimation()
            elif choice == '2':
                print("退出程序")
                break
            else:
                print("无效选择，请重新输入")
                
        except KeyboardInterrupt:
            print("\n\n退出程序")
            break
        except Exception as e:
            print(f"程序运行出错: {e}")
            
    return 0

if __name__ == "__main__":
    sys.exit(main())
