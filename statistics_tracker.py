#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计追踪模块
用于跟踪和统计表情识别和心率检测的结果
"""

import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple
import numpy as np
import json

class StatisticsTracker:
    """
    统计追踪器
    记录和分析表情、心率等数据
    """
    
    def __init__(self, history_size: int = 1000):
        """
        初始化统计追踪器
        
        Args:
            history_size: 历史数据保存数量
        """
        self.history_size = history_size
        
        # 表情统计
        self.emotion_counts = defaultdict(int)
        self.emotion_history = deque(maxlen=history_size)
        
        # 心率统计
        self.heart_rate_history = deque(maxlen=history_size)
        self.heart_rate_timestamps = deque(maxlen=history_size)
        
        # 时间统计
        self.start_time = time.time()
        self.last_update_time = time.time()
        
        # 总统计
        self.total_frames = 0
        self.total_faces_detected = 0
        self.total_emotions_detected = 0
        
    def update_emotion(self, emotion_label: str, timestamp: Optional[float] = None):
        """
        更新表情统计
        
        Args:
            emotion_label: 表情标签
            timestamp: 时间戳（可选）
        """
        if timestamp is None:
            timestamp = time.time()
            
        # 更新计数
        self.emotion_counts[emotion_label] += 1
        self.total_emotions_detected += 1
        
        # 添加到历史记录
        self.emotion_history.append({
            'emotion': emotion_label,
            'timestamp': timestamp
        })
        
        self.last_update_time = timestamp
        
    def update_heart_rate(self, bpm: float, timestamp: Optional[float] = None):
        """
        更新心率统计
        
        Args:
            bpm: 心率值（每分钟心跳次数）
            timestamp: 时间戳（可选）
        """
        if timestamp is None:
            timestamp = time.time()
            
        # 添加到历史记录
        self.heart_rate_history.append(bpm)
        self.heart_rate_timestamps.append(timestamp)
        
        self.last_update_time = timestamp
        
    def update_frame_stats(self, faces_detected: int):
        """
        更新帧统计信息
        
        Args:
            faces_detected: 检测到的人脸数量
        """
        self.total_frames += 1
        self.total_faces_detected += faces_detected
        
    def get_emotion_counts(self) -> Dict[str, int]:
        """
        获取表情计数
        
        Returns:
            Dict[str, int]: 表情计数字典
        """
        return dict(self.emotion_counts)
        
    def get_emotion_percentages(self) -> Dict[str, float]:
        """
        获取表情百分比
        
        Returns:
            Dict[str, float]: 表情百分比字典
        """
        total = sum(self.emotion_counts.values())
        if total == 0:
            return {}
            
        return {
            emotion: (count / total) * 100 
            for emotion, count in self.emotion_counts.items()
        }
        
    def get_recent_emotions(self, minutes: int = 5) -> List[Dict]:
        """
        获取最近几分钟的表情记录
        
        Args:
            minutes: 时间范围（分钟）
            
        Returns:
            List[Dict]: 最近的表情记录
        """
        cutoff_time = time.time() - (minutes * 60)
        
        recent_emotions = [
            record for record in self.emotion_history
            if record['timestamp'] >= cutoff_time
        ]
        
        return recent_emotions
        
    def get_heart_rate_stats(self) -> Dict:
        """
        获取心率统计信息
        
        Returns:
            Dict: 心率统计信息
        """
        if not self.heart_rate_history:
            return {
                'current': 0,
                'average': 0,
                'min': 0,
                'max': 0,
                'std': 0,
                'count': 0
            }
            
        hr_array = np.array(list(self.heart_rate_history))
        
        return {
            'current': float(hr_array[-1]) if len(hr_array) > 0 else 0,
            'average': float(np.mean(hr_array)),
            'min': float(np.min(hr_array)),
            'max': float(np.max(hr_array)),
            'std': float(np.std(hr_array)),
            'count': len(hr_array)
        }
        
    def get_recent_heart_rate(self, minutes: int = 5) -> Tuple[List[float], List[float]]:
        """
        获取最近几分钟的心率数据
        
        Args:
            minutes: 时间范围（分钟）
            
        Returns:
            Tuple[List[float], List[float]]: (心率值列表, 时间戳列表)
        """
        cutoff_time = time.time() - (minutes * 60)
        
        recent_hr = []
        recent_timestamps = []
        
        for hr, timestamp in zip(self.heart_rate_history, self.heart_rate_timestamps):
            if timestamp >= cutoff_time:
                recent_hr.append(hr)
                recent_timestamps.append(timestamp)
                
        return recent_hr, recent_timestamps
        
    def get_session_stats(self) -> Dict:
        """
        获取会话统计信息
        
        Returns:
            Dict: 会话统计信息
        """
        current_time = time.time()
        session_duration = current_time - self.start_time
        
        # 计算平均检测率
        avg_faces_per_frame = (
            self.total_faces_detected / self.total_frames 
            if self.total_frames > 0 else 0
        )
        
        # 计算情感检测率
        emotion_detection_rate = (
            self.total_emotions_detected / self.total_frames 
            if self.total_frames > 0 else 0
        )
        
        return {
            'session_duration': session_duration,
            'total_frames': self.total_frames,
            'total_faces_detected': self.total_faces_detected,
            'total_emotions_detected': self.total_emotions_detected,
            'avg_faces_per_frame': avg_faces_per_frame,
            'emotion_detection_rate': emotion_detection_rate,
            'heart_rate_readings': len(self.heart_rate_history)
        }
        
    def get_most_common_emotion(self) -> Tuple[str, int]:
        """
        获取最常见的表情
        
        Returns:
            Tuple[str, int]: (表情标签, 出现次数)
        """
        if not self.emotion_counts:
            return "neutral", 0
            
        most_common = max(self.emotion_counts.items(), key=lambda x: x[1])
        return most_common
        
    def reset_emotion_stats(self):
        """重置表情统计"""
        self.emotion_counts.clear()
        self.emotion_history.clear()
        self.total_emotions_detected = 0
        print("? 表情统计已重置")
        
    def reset_heart_rate_stats(self):
        """重置心率统计"""
        self.heart_rate_history.clear()
        self.heart_rate_timestamps.clear()
        print("? 心率统计已重置")
        
    def reset_all_stats(self):
        """重置所有统计"""
        self.emotion_counts.clear()
        self.emotion_history.clear()
        self.heart_rate_history.clear()
        self.heart_rate_timestamps.clear()
        
        self.total_frames = 0
        self.total_faces_detected = 0
        self.total_emotions_detected = 0
        
        self.start_time = time.time()
        self.last_update_time = time.time()
        
        print("? 所有统计已重置")
        
    def export_stats(self, filename: str):
        """
        导出统计数据到文件
        
        Args:
            filename: 导出文件名
        """
        try:
            stats_data = {
                'session_stats': self.get_session_stats(),
                'emotion_counts': self.get_emotion_counts(),
                'emotion_percentages': self.get_emotion_percentages(),
                'heart_rate_stats': self.get_heart_rate_stats(),
                'emotion_history': list(self.emotion_history),
                'heart_rate_history': {
                    'values': list(self.heart_rate_history),
                    'timestamps': list(self.heart_rate_timestamps)
                },
                'export_timestamp': time.time()
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(stats_data, f, indent=2, ensure_ascii=False)
                
            print(f"? 统计数据已导出到: {filename}")
            
        except Exception as e:
            print(f"? 导出统计数据失败: {e}")
            
    def print_summary(self):
        """打印统计摘要"""
        print("\n" + "=" * 50)
        print("统计摘要")
        print("=" * 50)
        
        # 会话信息
        session_stats = self.get_session_stats()
        duration_min = session_stats['session_duration'] / 60
        
        print(f"会话时长: {duration_min:.2f} 分钟")
        print(f"处理帧数: {session_stats['total_frames']}")
        print(f"检测人脸: {session_stats['total_faces_detected']}")
        print(f"平均每帧人脸数: {session_stats['avg_faces_per_frame']:.2f}")
        
        # 表情统计
        print(f"\n表情识别:")
        emotion_counts = self.get_emotion_counts()
        emotion_percentages = self.get_emotion_percentages()
        
        if emotion_counts:
            for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = emotion_percentages.get(emotion, 0)
                print(f"  {emotion}: {count} 次 ({percentage:.1f}%)")
                
            most_common_emotion, count = self.get_most_common_emotion()
            print(f"  最常见表情: {most_common_emotion} ({count} 次)")
        else:
            print("  无表情数据")
            
        # 心率统计
        print(f"\n心率统计:")
        hr_stats = self.get_heart_rate_stats()
        
        if hr_stats['count'] > 0:
            print(f"  当前心率: {hr_stats['current']:.1f} BPM")
            print(f"  平均心率: {hr_stats['average']:.1f} BPM")
            print(f"  心率范围: {hr_stats['min']:.1f} - {hr_stats['max']:.1f} BPM")
            print(f"  标准差: {hr_stats['std']:.1f}")
            print(f"  测量次数: {hr_stats['count']}")
        else:
            print("  无心率数据")
            
        print("=" * 50)

# 测试代码
if __name__ == "__main__":
    print("测试统计追踪器...")
    
    # 创建统计追踪器
    tracker = StatisticsTracker()
    
    # 模拟数据
    emotions = ['happy', 'neutral', 'sad', 'happy', 'angry', 'happy', 'neutral']
    heart_rates = [72, 75, 78, 80, 77, 74, 76]
    
    print("添加模拟数据...")
    
    # 添加表情数据
    for emotion in emotions:
        tracker.update_emotion(emotion)
        tracker.update_frame_stats(faces_detected=1)
        time.sleep(0.1)  # 模拟时间间隔
        
    # 添加心率数据
    for hr in heart_rates:
        tracker.update_heart_rate(hr)
        time.sleep(0.1)
        
    # 打印统计摘要
    tracker.print_summary()
    
    # 测试其他功能
    print(f"\n表情计数: {tracker.get_emotion_counts()}")
    print(f"表情百分比: {tracker.get_emotion_percentages()}")
    print(f"最常见表情: {tracker.get_most_common_emotion()}")
    
    # 测试导出功能
    tracker.export_stats("test_stats.json")
    
    # 测试重置功能
    print(f"\n重置前表情总数: {sum(tracker.get_emotion_counts().values())}")
    tracker.reset_emotion_stats()
    print(f"重置后表情总数: {sum(tracker.get_emotion_counts().values())}")
    
    print("统计追踪器测试完成")
