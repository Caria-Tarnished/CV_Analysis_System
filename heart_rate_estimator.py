#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
心率估计模块
基于远程光电容积描记（rPPG）的视频心率估计
"""

import cv2  # noqa: F401 (opencv is optional here; kept for potential future use)
import numpy as np
import time
from collections import deque
from typing import Optional, Tuple, List
from scipy import signal
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

class HeartRateEstimator:
    """
    基于rPPG的心率估计器
    通过分析面部皮肤颜色变化来估计心率
    """
    
    def __init__(self, buffer_size: int = 150, fps: float = 30.0, window_size: int = 5):
        """
        初始化心率估计器
        
        Args:
            buffer_size: 信号缓冲区大小（帧数）
            fps: 视频帧率
            window_size: 滑动窗口大小（用于平滑结果）
        """
        self.buffer_size = buffer_size
        self.fps = fps
        self.window_size = window_size
        
        # 信号缓冲区
        self.signal_buffer = deque(maxlen=buffer_size)
        self.timestamps = deque(maxlen=buffer_size)
        
        # 心率结果缓冲区（用于平滑）
        self.bpm_buffer = deque(maxlen=window_size)
        
        # ROI相关参数
        self.roi_method = 'forehead'  # 'forehead', 'cheeks', 'full_face'
        
        # 滤波器参数
        self.min_bpm = 45   # 最小心率 (45 BPM = 0.75 Hz)
        self.max_bpm = 150  # 最大心率 (150 BPM = 2.5 Hz)
        self.min_freq = self.min_bpm / 60.0
        self.max_freq = self.max_bpm / 60.0
        
        # 信号质量评估
        self.quality_threshold = 0.1
        
        print("[HR] 心率估计器初始化完成")
        print(f"  缓冲区大小: {buffer_size} 帧")
        print(f"  帧率: {fps} FPS")
        print(f"  心率范围: {self.min_bpm}-{self.max_bpm} BPM")
        
    def _extract_roi(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        从人脸区域提取感兴趣区域（ROI）
        
        Args:
            frame: 输入帧
            face_bbox: 人脸边界框 (x, y, w, h)
            
        Returns:
            np.ndarray: ROI区域，如果提取失败则返回None
        """
        try:
            x, y, w, h = face_bbox
            
            # 确保边界框在图像范围内
            h_img, w_img = frame.shape[:2]
            x = max(0, min(x, w_img - 1))
            y = max(0, min(y, h_img - 1))
            w = max(1, min(w, w_img - x))
            h = max(1, min(h, h_img - y))
            
            if self.roi_method == 'forehead':
                # 前额区域：人脸上方1/3，中间2/3宽度
                roi_x = x + w // 6
                roi_y = y
                roi_w = w * 2 // 3
                roi_h = h // 3
                
            elif self.roi_method == 'cheeks':
                # 脸颊区域：人脸中间1/3高度，两侧区域
                roi_x = x + w // 6
                roi_y = y + h // 3
                roi_w = w * 2 // 3
                roi_h = h // 3
                
            elif self.roi_method == 'full_face':
                # 全脸区域：整个人脸
                roi_x, roi_y, roi_w, roi_h = x, y, w, h
                
            else:
                # 默认使用前额
                roi_x = x + w // 6
                roi_y = y
                roi_w = w * 2 // 3
                roi_h = h // 3
                
            # 确保ROI在合理范围内
            roi_x = max(0, min(roi_x, w_img - 1))
            roi_y = max(0, min(roi_y, h_img - 1))
            roi_w = max(1, min(roi_w, w_img - roi_x))
            roi_h = max(1, min(roi_h, h_img - roi_y))
            
            # 提取ROI
            roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            
            return roi if roi.size > 0 else None
            
        except Exception as e:
            print(f"ROI提取错误: {e}")
            return None
            
    def _extract_green_signal(self, roi: np.ndarray) -> float:
        """
        从ROI中提取绿色通道的平均值
        
        Args:
            roi: 感兴趣区域
            
        Returns:
            float: 绿色通道平均值
        """
        try:
            if len(roi.shape) == 3:
                # BGR格式，绿色通道是索引1
                green_channel = roi[:, :, 1]
            else:
                # 灰度图，直接使用
                green_channel = roi
                
            # 计算平均值
            mean_value = np.mean(green_channel)
            
            return float(mean_value)
            
        except Exception as e:
            print(f"绿色信号提取错误: {e}")
            return 0.0
            
    def process_frame(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> bool:
        """
        处理单帧图像，提取心率信号
        
        Args:
            frame: 输入帧
            face_bbox: 人脸边界框 (x, y, w, h)
            
        Returns:
            bool: 是否成功处理
        """
        try:
            # 提取ROI
            roi = self._extract_roi(frame, face_bbox)
            if roi is None:
                return False
                
            # 提取绿色信号
            green_signal = self._extract_green_signal(roi)
            
            # 添加到缓冲区
            current_time = time.time()
            self.signal_buffer.append(green_signal)
            self.timestamps.append(current_time)
            
            return True
            
        except Exception as e:
            print(f"帧处理错误: {e}")
            return False
            
    def _preprocess_signal(self, signal_data: np.ndarray) -> np.ndarray:
        """
        预处理信号：去趋势、归一化
        
        Args:
            signal_data: 原始信号数据
            
        Returns:
            np.ndarray: 预处理后的信号
        """
        try:
            # 去趋势（移除线性趋势）
            detrended = signal.detrend(signal_data, type='linear')
            
            # 归一化（零均值，单位方差）
            if np.std(detrended) > 0:
                normalized = (detrended - np.mean(detrended)) / np.std(detrended)
            else:
                normalized = detrended
                
            return normalized
            
        except Exception as e:
            print(f"信号预处理错误: {e}")
            return signal_data
            
    def _bandpass_filter(self, signal_data: np.ndarray) -> np.ndarray:
        """
        带通滤波器：保留心率频率范围
        
        Args:
            signal_data: 输入信号
            
        Returns:
            np.ndarray: 滤波后的信号
        """
        try:
            # 设计Butterworth带通滤波器
            nyquist = self.fps / 2.0
            low_freq = self.min_freq / nyquist
            high_freq = self.max_freq / nyquist
            
            # 确保频率在有效范围内
            low_freq = max(0.01, min(low_freq, 0.99))
            high_freq = max(low_freq + 0.01, min(high_freq, 0.99))
            
            # 设计滤波器
            b, a = signal.butter(4, [low_freq, high_freq], btype='band')
            
            # 应用滤波器（使用filtfilt避免相位失真）
            filtered_signal = signal.filtfilt(b, a, signal_data)
            
            return filtered_signal
            
        except Exception as e:
            print(f"带通滤波错误: {e}")
            return signal_data
            
    def _estimate_bpm_fft(self, filtered_signal: np.ndarray) -> Tuple[float, float]:
        """
        使用FFT估计心率
        
        Args:
            filtered_signal: 滤波后的信号
            
        Returns:
            Tuple[float, float]: (BPM值, 信号质量分数)
        """
        try:
            # 计算FFT
            fft_data = fft(filtered_signal)
            freqs = fftfreq(len(filtered_signal), 1.0 / self.fps)
            
            # 只保留正频率部分
            positive_freqs = freqs[:len(freqs)//2]
            positive_fft = np.abs(fft_data[:len(fft_data)//2])
            
            # 在心率频率范围内查找峰值
            freq_mask = (positive_freqs >= self.min_freq) & (positive_freqs <= self.max_freq)
            valid_freqs = positive_freqs[freq_mask]
            valid_fft = positive_fft[freq_mask]
            
            if len(valid_fft) == 0:
                return 0.0, 0.0
                
            # 找到功率最大的频率
            peak_idx = np.argmax(valid_fft)
            peak_freq = valid_freqs[peak_idx]
            peak_power = valid_fft[peak_idx]
            
            # 转换为BPM
            bpm = peak_freq * 60.0
            
            # 计算信号质量分数（峰值与周围功率的比值）
            if len(valid_fft) > 1:
                avg_power = np.mean(valid_fft)
                quality_score = peak_power / (avg_power + 1e-6)
            else:
                quality_score = 0.0
                
            return bpm, quality_score
            
        except Exception as e:
            print(f"FFT心率估计错误: {e}")
            return 0.0, 0.0
            
    def estimate_bpm(self) -> Optional[float]:
        """
        估计当前心率
        
        Returns:
            Optional[float]: 估计的BPM值，如果无法估计则返回None
        """
        # 检查缓冲区是否有足够的数据
        if len(self.signal_buffer) < self.buffer_size * 0.8:
            return None
            
        try:
            # 转换为numpy数组
            signal_data = np.array(list(self.signal_buffer))
            
            # 检查信号变化是否足够大
            if np.std(signal_data) < self.quality_threshold:
                return None
                
            # 预处理信号
            preprocessed = self._preprocess_signal(signal_data)
            
            # 带通滤波
            filtered = self._bandpass_filter(preprocessed)
            
            # FFT估计心率
            bpm, quality = self._estimate_bpm_fft(filtered)
            
            # 检查结果的合理性
            if self.min_bpm <= bpm <= self.max_bpm and quality > 1.0:
                # 添加到BPM缓冲区用于平滑
                self.bpm_buffer.append(bpm)
                
                # 返回平滑后的结果
                smoothed_bpm = np.median(list(self.bpm_buffer))
                return float(smoothed_bpm)
            else:
                return None
                
        except Exception as e:
            print(f"心率估计错误: {e}")
            return None
            
    def get_signal_quality(self) -> float:
        """
        获取当前信号质量评分
        
        Returns:
            float: 信号质量分数 (0-1)
        """
        if len(self.signal_buffer) < 10:
            return 0.0
            
        try:
            # deque 不支持切片，先转换为列表再切片
            recent_values = list(self.signal_buffer)
            signal_data = np.array(recent_values[-30:])  # 最近30帧
            
            # 计算信号变化程度
            std_score = min(np.std(signal_data) / 10.0, 1.0)
            
            # 计算信号稳定性（变化率的一致性）
            if len(signal_data) > 5:
                diff = np.diff(signal_data)
                stability_score = 1.0 - min(np.std(diff) / np.mean(np.abs(diff) + 1e-6), 1.0)
            else:
                stability_score = 0.0
                
            # 综合评分
            quality = (std_score + stability_score) / 2.0
            
            return max(0.0, min(quality, 1.0))
            
        except Exception as e:
            print(f"信号质量评估错误: {e}")
            return 0.0
            
    def get_bpm_history(self) -> List[float]:
        """
        获取心率历史记录
        
        Returns:
            List[float]: BPM历史记录
        """
        return list(self.bpm_buffer)
        
    def set_roi_method(self, method: str):
        """
        设置ROI提取方法
        
        Args:
            method: ROI方法 ('forehead', 'cheeks', 'full_face')
        """
        if method in ['forehead', 'cheeks', 'full_face']:
            self.roi_method = method
            print(f"[HR] ROI方法已设置为: {method}")
        else:
            print(f"[HR] 不支持的ROI方法: {method}")
            
    def reset(self):
        """重置所有缓冲区"""
        self.signal_buffer.clear()
        self.timestamps.clear()
        self.bpm_buffer.clear()
        print("[HR] 心率估计器已重置")
        
    def plot_signal(self, save_path: Optional[str] = None):
        """
        绘制当前信号
        
        Args:
            save_path: 保存路径（可选）
        """
        if len(self.signal_buffer) < 10:
            print("信号数据不足，无法绘制")
            return
            
        try:
            signal_data = np.array(list(self.signal_buffer))
            time_data = np.arange(len(signal_data)) / self.fps
            
            plt.figure(figsize=(12, 8))
            
            # 原始信号
            plt.subplot(3, 1, 1)
            plt.plot(time_data, signal_data)
            plt.title('原始信号')
            plt.ylabel('绿色通道值')
            plt.grid(True)
            
            # 预处理后的信号
            preprocessed = self._preprocess_signal(signal_data)
            plt.subplot(3, 1, 2)
            plt.plot(time_data, preprocessed)
            plt.title('预处理后信号')
            plt.ylabel('归一化值')
            plt.grid(True)
            
            # 滤波后的信号
            filtered = self._bandpass_filter(preprocessed)
            plt.subplot(3, 1, 3)
            plt.plot(time_data, filtered)
            plt.title('带通滤波后信号')
            plt.xlabel('时间 (秒)')
            plt.ylabel('滤波值')
            plt.grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                print(f"? 信号图已保存到: {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            print(f"绘制信号错误: {e}")

# 测试代码
if __name__ == "__main__":
    print("测试心率估计器...")
    
    # 创建心率估计器
    hr_estimator = HeartRateEstimator(buffer_size=150, fps=30)
    
    # 模拟测试数据（心率约75 BPM）
    print("生成模拟信号（75 BPM）...")
    
    # 生成5秒的模拟数据
    duration = 5.0
    fps = 30
    num_frames = int(duration * fps)
    
    # 心率信号 (1.25 Hz = 75 BPM) + 噪声
    t = np.linspace(0, duration, num_frames)
    heart_signal = np.sin(2 * np.pi * 1.25 * t)  # 75 BPM
    noise = np.random.normal(0, 0.3, num_frames)
    baseline = 100 + 10 * np.sin(2 * np.pi * 0.1 * t)  # 慢变化的基线
    
    simulated_signal = baseline + 5 * heart_signal + noise
    
    # 模拟人脸框
    fake_face_bbox = (100, 100, 200, 200)
    
    # 逐帧处理模拟数据
    for i, signal_value in enumerate(simulated_signal):
        # 创建模拟帧（绿色通道设置为信号值）
        fake_frame = np.ones((400, 400, 3), dtype=np.uint8) * 128
        fake_frame[100:300, 100:300, 1] = int(signal_value)  # 绿色通道
        
        # 处理帧
        hr_estimator.process_frame(fake_frame, fake_face_bbox)
        
        # 每30帧尝试估计一次心率
        if i > 60 and i % 30 == 0:
            bpm = hr_estimator.estimate_bpm()
            quality = hr_estimator.get_signal_quality()
            
            if bpm is not None:
                print(f"帧 {i}: 估计心率 = {bpm:.1f} BPM, 信号质量 = {quality:.2f}")
            else:
                print(f"帧 {i}: 无法估计心率, 信号质量 = {quality:.2f}")
                
    # 最终结果
    final_bpm = hr_estimator.estimate_bpm()
    if final_bpm:
        print(f"\n最终估计心率: {final_bpm:.1f} BPM (目标: 75 BPM)")
        error = abs(final_bpm - 75.0)
        print(f"估计误差: {error:.1f} BPM")
    else:
        print("\n无法获得最终心率估计")
        
    # 绘制信号
    print("\n绘制信号图...")
    hr_estimator.plot_signal("heart_rate_signal_test.png")
    
    print("心率估计器测试完成")
