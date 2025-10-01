#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频流处理模块
提供多线程的视频流读取和处理功能
"""

import cv2
import threading
import time
from collections import deque
from typing import Optional, Union
import numpy as np

class VideoStreamHandler:
    """
    视频流处理器
    支持多线程读取视频流，避免主线程阻塞
    """
    
    def __init__(self, source: Union[int, str] = 0, buffer_size: int = 2):
        """
        初始化视频流处理器
        
        Args:
            source: 视频源，可以是摄像头索引(int)或视频文件路径(str)
            buffer_size: 帧缓冲区大小
        """
        self.source = source
        self.buffer_size = buffer_size
        
        # 视频捕获对象
        self.cap = None
        
        # 线程相关
        self.thread = None
        self.is_running = False
        self.thread_lock = threading.Lock()
        
        # 帧缓冲区（使用deque实现环形缓冲区）
        self.frame_buffer = deque(maxlen=buffer_size)
        self.buffer_lock = threading.Lock()
        
        # 统计信息
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        
        # 视频信息
        self.frame_width = 0
        self.frame_height = 0
        self.source_fps = 0
        
    def start(self) -> bool:
        """
        启动视频流读取
        
        Returns:
            bool: 是否成功启动
        """
        if self.is_running:
            print("视频流已经在运行中")
            return True
            
        # 初始化视频捕获
        self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            print(f"无法打开视频源: {self.source}")
            return False
            
        # 获取视频信息
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.source_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"视频源信息: {self.frame_width}x{self.frame_height}, FPS: {self.source_fps}")
        
        # 设置缓冲区大小（减少延迟）
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # 启动读取线程
        self.is_running = True
        self.thread = threading.Thread(target=self._read_frames, daemon=True)
        self.thread.start()
        
        # 等待第一帧
        timeout = 5.0  # 5秒超时
        start_time = time.time()
        while len(self.frame_buffer) == 0 and (time.time() - start_time) < timeout:
            time.sleep(0.01)
            
        if len(self.frame_buffer) == 0:
            print("无法获取第一帧，启动失败")
            self.stop()
            return False
            
        print("视频流启动成功")
        return True
        
    def _read_frames(self):
        """
        内部方法：在单独线程中持续读取帧
        """
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("无法读取帧，可能到达视频末尾")
                    break
                    
                # 更新帧缓冲区
                with self.buffer_lock:
                    self.frame_buffer.append(frame.copy())
                    
                # 更新统计信息
                self.frame_count += 1
                current_time = time.time()
                if current_time - self.last_time >= 1.0:  # 每秒更新一次FPS
                    self.fps = self.frame_count / (current_time - self.last_time)
                    self.frame_count = 0
                    self.last_time = current_time
                    
                # 控制读取速度，避免CPU占用过高
                time.sleep(0.001)
                
            except Exception as e:
                print(f"读取帧时发生错误: {e}")
                break
                
        # 线程结束清理
        with self.thread_lock:
            self.is_running = False
            
    def read(self) -> Optional[np.ndarray]:
        """
        读取最新的一帧
        
        Returns:
            np.ndarray: 图像帧，如果无可用帧则返回None
        """
        if not self.is_running:
            return None
            
        with self.buffer_lock:
            if len(self.frame_buffer) > 0:
                return self.frame_buffer[-1].copy()  # 返回最新帧的副本
            else:
                return None
                
    def stop(self):
        """
        停止视频流读取并释放资源
        """
        print("正在停止视频流...")
        
        # 设置停止标志
        with self.thread_lock:
            self.is_running = False
            
        # 等待线程结束
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            
        # 释放视频捕获资源
        if self.cap:
            self.cap.release()
            self.cap = None
            
        # 清空缓冲区
        with self.buffer_lock:
            self.frame_buffer.clear()
            
        print("视频流已停止")
        
    def get_fps(self) -> float:
        """
        获取当前实际FPS
        
        Returns:
            float: 当前FPS
        """
        return self.fps
        
    def get_frame_info(self) -> dict:
        """
        获取帧信息
        
        Returns:
            dict: 包含宽度、高度、FPS等信息
        """
        return {
            'width': self.frame_width,
            'height': self.frame_height,
            'source_fps': self.source_fps,
            'actual_fps': self.fps,
            'buffer_size': len(self.frame_buffer),
            'is_running': self.is_running
        }
        
    def is_opened(self) -> bool:
        """
        检查视频流是否正常运行
        
        Returns:
            bool: 是否正常运行
        """
        return self.is_running and self.cap is not None and self.cap.isOpened()
        
    def __del__(self):
        """
        析构函数：确保资源被正确释放
        """
        self.stop()

# 测试代码
if __name__ == "__main__":
    print("测试VideoStreamHandler类...")
    
    # 创建视频流处理器
    video_handler = VideoStreamHandler(0)  # 使用默认摄像头
    
    # 启动视频流
    if video_handler.start():
        print("按 'q' 键退出...")
        
        try:
            while True:
                # 读取帧
                frame = video_handler.read()
                
                if frame is not None:
                    # 在帧上添加信息
                    info = video_handler.get_frame_info()
                    cv2.putText(frame, f"FPS: {info['actual_fps']:.1f}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Buffer: {info['buffer_size']}", 
                              (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # 显示图像
                    cv2.imshow('Video Stream Test', frame)
                    
                # 检查退出键
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\n收到中断信号")
            
        finally:
            # 清理资源
            video_handler.stop()
            cv2.destroyAllWindows()
            
    else:
        print("无法启动视频流")
