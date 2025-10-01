#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算机视觉情感与心率分析系统 - 主GUI程序
"""

import sys
import cv2
import time
import os
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QGroupBox,
                             QStatusBar, QProgressBar,
                             QSplitter, QFrame, QGridLayout)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt
from PyQt5.QtGui import QPixmap, QImage

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas)
from matplotlib.figure import Figure
import matplotlib.style as mplstyle

# 导入自定义模块
from video_handler import VideoStreamHandler
from face_detector import FaceDetector
from emotion_recognizer import EmotionRecognizer
from heart_rate_estimator import HeartRateEstimator
from statistics_tracker import StatisticsTracker
from text_renderer import draw_text_cn

# 设置matplotlib样式
mplstyle.use('default')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class ProcessingThread(QThread):
    """
    视频处理工作线程
    负责视频读取、人脸检测、表情识别和心率估计
    """
    
    # 信号定义
    frame_updated = pyqtSignal(np.ndarray)
    emotion_updated = pyqtSignal(str, dict)
    hr_updated = pyqtSignal(float, list)
    status_updated = pyqtSignal(str)
    stats_updated = pyqtSignal(dict)
    face_count_updated = pyqtSignal(int)  # 新增：人脸数量更新信号
    
    def __init__(self):
        super().__init__()
        
        # 组件初始化
        self.video_handler = None
        self.face_detector = None
        self.emotion_recognizer = None
        self.hr_estimator = None
        self.stats_tracker = None
        
        # 控制标志
        self.is_running = False
        self.emotion_recognition_enabled = False
        self.heart_rate_estimation_enabled = False
        
        # 统计信息
        self.frame_count = 0
        self.start_time = None
        
    def initialize_components(self):
        """初始化所有组件"""
        try:
            self.status_updated.emit("初始化摄像头...")
            self.video_handler = VideoStreamHandler(source=0)
            
            self.status_updated.emit("初始化人脸检测器...")
            self.face_detector = FaceDetector(method='haar')
            
            self.status_updated.emit("初始化表情识别器...")
            # 自动检测可用的模型文件（支持.pth, .h5, .onnx）
            model_path = None
            models_dir = "models"
            
            # 按优先级检查模型文件
            for ext in ['.pth', '.h5', '.onnx']:
                candidate_path = os.path.join(models_dir, f"emotion_model{ext}")
                if os.path.exists(candidate_path):
                    model_path = candidate_path
                    self.status_updated.emit(f"发现{ext}格式模型...")
                    break
            
            # 如果没有找到任何模型，创建一个
            if model_path is None:
                self.status_updated.emit("首次运行，创建表情识别模型...")
                try:
                    from download_models import download_or_create_model
                    model_path = download_or_create_model()
                except Exception as e:
                    print(f"模型创建失败: {e}，使用默认模型")
                    model_path = None
            
            self.emotion_recognizer = EmotionRecognizer(
                model_path=model_path, confidence_threshold=0.2)
            
            self.status_updated.emit("初始化心率估计器...")
            self.hr_estimator = HeartRateEstimator(buffer_size=180, fps=30)
            
            self.status_updated.emit("初始化统计追踪器...")
            self.stats_tracker = StatisticsTracker()
            
            return True
            
        except Exception as e:
            self.status_updated.emit(f"初始化失败: {str(e)}")
            return False
            
    def start_processing(self):
        """开始处理"""
        if not self.initialize_components():
            return False
            
        if not self.video_handler.start():
            self.status_updated.emit("无法启动摄像头")
            return False
            
        self.is_running = True
        self.start_time = time.time()
        self.status_updated.emit("系统运行中...")
        self.start()
        return True
        
    def stop_processing(self):
        """停止处理"""
        self.is_running = False
        if self.video_handler:
            self.video_handler.stop()
        self.wait()
        self.status_updated.emit("系统已停止")
        
    def run(self):
        """主处理循环"""
        while self.is_running:
            try:
                # 读取帧
                frame = self.video_handler.read()
                if frame is None:
                    time.sleep(0.01)
                    continue
                    
                self.frame_count += 1
                display_frame = frame.copy()
                
                # 检测人脸
                faces = self.face_detector.detect(frame)
                
                # 更新统计
                self.stats_tracker.update_frame_stats(len(faces))
                
                current_emotion = "未检测"
                current_bpm = None
                
                # 处理第一个检测到的人脸
                if len(faces) > 0:
                    face_bbox = faces[0]
                    x, y, w, h = face_bbox
                    
                    # 绘制人脸框
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # 表情识别
                    if self.emotion_recognition_enabled:
                        emotion = self.emotion_recognizer.predict(frame, face_bbox)
                        current_emotion = self.emotion_recognizer.get_chinese_label(emotion)
                        
                        # 更新统计
                        self.stats_tracker.update_emotion(emotion)
                        
                        # 在人脸上显示表情（中文防乱码）
                        display_frame = draw_text_cn(
                            display_frame,
                            current_emotion,
                            (x, y - 10),
                            color=(0, 255, 0),
                            font_size=22,
                            stroke_width=1,
                            stroke_fill=(0, 0, 0),
                        )
                    
                    # 心率估计
                    if self.heart_rate_estimation_enabled:
                        success = self.hr_estimator.process_frame(frame, face_bbox)
                        
                        if success and self.frame_count % 15 == 0:  # 每15帧估计一次
                            bpm = self.hr_estimator.estimate_bpm()
                            if bpm is not None:
                                current_bpm = bpm
                                self.stats_tracker.update_heart_rate(bpm)
                                
                                # 显示心率信息
                                hr_text = f"心率: {bpm:.1f} BPM"
                                display_frame = draw_text_cn(
                                    display_frame, hr_text, (x, y + h + 25),
                                    color=(255, 0, 0), font_size=20,
                                    stroke_width=1, stroke_fill=(0, 0, 0)
                                )
                
                # 添加系统信息
                info_y = 30
                cv2.putText(display_frame, f"FPS: {self.video_handler.get_fps():.1f}", 
                          (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                info_y += 25
                
                display_frame = draw_text_cn(
                    display_frame, f"人脸: {len(faces)}", (10, info_y),
                    color=(255, 255, 255), font_size=20,
                    stroke_width=1, stroke_fill=(0, 0, 0)
                )
                
                # 发送信号更新UI
                self.frame_updated.emit(display_frame)
                self.face_count_updated.emit(len(faces))  # 新增：发送人脸数量
                
                if self.emotion_recognition_enabled:
                    emotion_stats = self.stats_tracker.get_emotion_counts()
                    self.emotion_updated.emit(current_emotion, emotion_stats)
                
                if self.heart_rate_estimation_enabled and current_bpm is not None:
                    hr_history = self.hr_estimator.get_bpm_history()
                    self.hr_updated.emit(current_bpm, hr_history)
                
                # 定期发送统计信息
                if self.frame_count % 30 == 0:
                    session_stats = self.stats_tracker.get_session_stats()
                    self.stats_updated.emit(session_stats)
                
                # 控制帧率
                time.sleep(0.033)  # 约30FPS
                
            except Exception as e:
                self.status_updated.emit(f"处理错误: {str(e)}")
                time.sleep(0.1)

class EmotionChart(FigureCanvas):
    """表情统计图表"""
    
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(4, 3.2), dpi=85)
        super().__init__(self.figure)
        self.setParent(parent)
        
        self.axes = self.figure.add_subplot(111)
        self.figure.patch.set_facecolor('white')
        self.figure.subplots_adjust(left=0.08, right=0.92, top=0.85, bottom=0.15)
        
        self.clear_chart()
        
    def clear_chart(self):
        """清空图表"""
        self.axes.clear()
        self.axes.set_title('表情统计', fontsize=10, fontweight='bold')
        self.axes.text(0.5, 0.5, '暂无数据', ha='center', va='center', 
                      transform=self.axes.transAxes, fontsize=10)
        self.draw()
        
    def update_chart(self, emotion_stats):
        """更新表情统计图"""
        self.axes.clear()
        
        if not emotion_stats:
            self.clear_chart()
            return
            
        # 获取中文标签
        from emotion_recognizer import EmotionRecognizer
        recognizer = EmotionRecognizer()
        
        emotions = list(emotion_stats.keys())
        counts = list(emotion_stats.values())
        chinese_emotions = [recognizer.get_chinese_label(emotion) for emotion in emotions]
        
        # 创建饼图
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE']
        
        wedges, texts, autotexts = self.axes.pie(counts, labels=chinese_emotions, 
                                                autopct='%1.1f%%', startangle=90,
                                                colors=colors[:len(emotions)])
        
        self.axes.set_title('表情分布统计', fontsize=11, fontweight='bold')
        
        # 美化文本
        for text in texts:
            text.set_fontsize(9)
            text.set_fontweight('bold')
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(8)
            
        self.draw()

class HeartRateChart(FigureCanvas):
    """心率趋势图表"""
    
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(4, 3.2), dpi=85)
        super().__init__(self.figure)
        self.setParent(parent)
        
        self.axes = self.figure.add_subplot(111)
        self.figure.patch.set_facecolor('white')
        self.figure.subplots_adjust(left=0.12, right=0.95, top=0.85, bottom=0.2)
        
        self.clear_chart()
        
    def clear_chart(self):
        """清空图表"""
        self.axes.clear()
        self.axes.set_title('心率趋势', fontsize=11, fontweight='bold')
        self.axes.set_xlabel('时间', fontsize=9)
        self.axes.set_ylabel('BPM', fontsize=9)
        self.axes.grid(True, alpha=0.3)
        self.axes.tick_params(axis='both', which='major', labelsize=8)
        self.axes.text(0.5, 0.5, '暂无数据', ha='center', va='center', 
                      transform=self.axes.transAxes, fontsize=11)
        self.draw()
        
    def update_chart(self, hr_history):
        """更新心率趋势图"""
        self.axes.clear()
        
        if not hr_history or len(hr_history) < 2:
            self.clear_chart()
            return
            
        # 绘制心率曲线
        x = list(range(len(hr_history)))
        self.axes.plot(x, hr_history, 'r-', linewidth=2, marker='o', markersize=4)
        
        # 添加平均线
        avg_hr = np.mean(hr_history)
        self.axes.axhline(y=avg_hr, color='blue', linestyle='--', alpha=0.7, 
                         label=f'平均: {avg_hr:.1f} BPM')
        
        self.axes.set_title('心率变化趋势', fontsize=11, fontweight='bold')
        self.axes.set_xlabel('测量次数', fontsize=9)
        self.axes.set_ylabel('心率 (BPM)', fontsize=9)
        self.axes.grid(True, alpha=0.3)
        self.axes.tick_params(axis='both', which='major', labelsize=8)
        self.axes.legend(fontsize=9)
        
        # 设置y轴范围
        if hr_history:
            min_hr = min(hr_history)
            max_hr = max(hr_history)
            margin = (max_hr - min_hr) * 0.1 + 5
            self.axes.set_ylim(min_hr - margin, max_hr + margin)
        
        self.draw()

class MainWindow(QMainWindow):
    """主窗口类"""
    
    def __init__(self):
        super().__init__()
        
        self.processing_thread = None
        self.initUI()
        
    def initUI(self):
        """初始化用户界面"""
        self.setWindowTitle('🎭 计算机视觉情感与心率分析系统 💓')
        self.setGeometry(100, 100, 1200, 800)  # 减小窗口尺寸
        self.setMinimumSize(1000, 600)  # 设置最小尺寸
        
        # 设置简化的现代化应用样式（移除不兼容属性）
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #f8f9fa, stop:1 #e9ecef);
                font-family: 'Microsoft YaHei UI', '微软雅黑', Arial, sans-serif;
            }
            
            QGroupBox {
                font-weight: bold;
                font-size: 12px;
                color: #2c3e50;
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ffffff, stop:1 #f8f9fa);
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 4px 10px;
                background: #3498db;
                color: white;
                border-radius: 6px;
                font-weight: bold;
                font-size: 11px;
            }
            
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3498db, stop:1 #2980b9);
                border: none;
                color: white;
                padding: 8px 15px;
                text-align: center;
                font-size: 12px;
                font-weight: bold;
                border-radius: 6px;
                min-height: 30px;
                max-height: 40px;
            }
            
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3cb0fd, stop:1 #3498db);
            }
            
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2980b9, stop:1 #21618c);
            }
            
            QPushButton:disabled {
                background: #bdc3c7;
                color: #7f8c8d;
            }
            
            /* 特殊按钮样式 */
            QPushButton#camera_btn {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #27ae60, stop:1 #229954);
            }
            
            QPushButton#camera_btn:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2ecc71, stop:1 #27ae60);
            }
            
            QPushButton#emotion_btn {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #e74c3c, stop:1 #c0392b);
            }
            
            QPushButton#emotion_btn:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ec7063, stop:1 #e74c3c);
            }
            
            QPushButton#heart_rate_btn {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f39c12, stop:1 #e67e22);
            }
            
            QPushButton#heart_rate_btn:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f7dc6f, stop:1 #f39c12);
            }
            
            QPushButton#reset_btn {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #8e44ad, stop:1 #7d3c98);
            }
            
            QPushButton#reset_btn:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #a569bd, stop:1 #8e44ad);
            }
            
            QLabel {
                color: #2c3e50;
                font-size: 12px;
            }
            
            QFrame {
                border-radius: 8px;
                background: rgba(255, 255, 255, 0.9);
            }
            
            QStatusBar {
                background: #34495e;
                color: white;
                font-weight: bold;
                padding: 4px;
                font-size: 11px;
            }
            
            QProgressBar {
                border: 2px solid #bdc3c7;
                border-radius: 6px;
                text-align: center;
                background: #ecf0f1;
                font-size: 11px;
            }
            
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3498db, stop:1 #2980b9);
                border-radius: 4px;
            }
        """)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # 左侧：视频显示区域
        self.create_video_panel(splitter)
        
        # 右侧：控制和信息面板
        self.create_control_panel(splitter)
        
        # 设置分割器比例
        splitter.setSizes([800, 600])
        
        # 创建状态栏
        self.create_status_bar()
        
        # 初始化定时器
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_ui)
        self.update_timer.start(100)  # 100ms更新一次
        
    def create_video_panel(self, parent):
        """创建视频显示面板"""
        video_frame = QFrame()
        video_frame.setFrameStyle(QFrame.NoFrame)
        video_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ffffff, stop:1 #f8f9fa);
                border: 2px solid #bdc3c7;
                border-radius: 15px;
                padding: 10px;
            }
        """)
        video_layout = QVBoxLayout(video_frame)
        video_layout.setContentsMargins(15, 15, 15, 15)
        
        # 添加标题
        title_label = QLabel("📹 实时视频监控")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #2c3e50;
                padding: 8px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3498db, stop:1 #2980b9);
                color: white;
                border-radius: 8px;
                margin-bottom: 10px;
            }
        """)
        title_label.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(title_label)
        
        # 视频显示标签
        self.video_label = QLabel()
        self.video_label.setMinimumSize(550, 400)
        self.video_label.setMaximumSize(700, 500)
        self.video_label.setStyleSheet("""
            QLabel {
                border: 2px solid #3498db;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #2c3e50, stop:1 #34495e);
                color: white;
                font-size: 14px;
                font-weight: bold;
                text-align: center;
                border-radius: 8px;
                padding: 15px;
            }
        """)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText('🎥 点击 "打开摄像头" 开始实时分析\n\n系统将自动检测人脸并进行情感与心率分析')
        self.video_label.setScaledContents(True)  # 允许内容缩放
        
        video_layout.addWidget(self.video_label)
        parent.addWidget(video_frame)
        
    def create_control_panel(self, parent):
        """创建控制面板"""
        control_frame = QFrame()
        control_layout = QVBoxLayout(control_frame)
        control_layout.setContentsMargins(0, 0, 0, 0) # 确保边距由子控件控制
        
        # 创建子面板
        control_group = self.create_control_buttons()
        info_group = self.create_info_display()
        charts_group = self.create_charts()
        
        # 将子面板添加到布局中
        control_layout.addWidget(control_group)
        control_layout.addWidget(info_group)
        control_layout.addWidget(charts_group)
        
        # 设置拉伸因子，让图表区域占据所有可用空间
        control_layout.setStretch(0, 0) # control_group 不拉伸
        control_layout.setStretch(1, 0) # info_group 不拉伸
        control_layout.setStretch(2, 1) # charts_group 占据所有剩余空间 (拉伸因子为1)
        
        parent.addWidget(control_frame)
        
    def create_control_buttons(self):
        """创建控制按钮组 (2x2 网格布局)"""
        control_group = QGroupBox("🎛️ 系统控制")
        # 使用 QGridLayout
        control_layout = QGridLayout(control_group)
        control_layout.setSpacing(10)
        control_layout.setContentsMargins(15, 20, 15, 15)
        
        # --- 创建按钮 ---
        # 摄像头控制
        self.camera_btn = QPushButton("📹 打开摄像头")
        self.camera_btn.setObjectName("camera_btn")
        self.camera_btn.clicked.connect(self.toggle_camera)
        
        # 表情识别控制
        self.emotion_btn = QPushButton("🎭 开始表情识别")
        self.emotion_btn.setObjectName("emotion_btn")
        self.emotion_btn.setEnabled(False)
        self.emotion_btn.clicked.connect(self.toggle_emotion_recognition)
        
        # 心率检测控制
        self.heart_rate_btn = QPushButton("💓 开始心率检测")
        self.heart_rate_btn.setObjectName("heart_rate_btn")
        self.heart_rate_btn.setEnabled(False)
        self.heart_rate_btn.clicked.connect(self.toggle_heart_rate_detection)
        
        # 重置统计按钮
        self.reset_btn = QPushButton("🔄 重置统计")
        self.reset_btn.setObjectName("reset_btn")
        self.reset_btn.setEnabled(False)
        self.reset_btn.clicked.connect(self.reset_statistics)
        
        # --- 将按钮添加到 2x2 网格中 ---
        # 第 1 行
        control_layout.addWidget(self.camera_btn, 0, 0)
        control_layout.addWidget(self.emotion_btn, 0, 1)
        
        # 第 2 行
        control_layout.addWidget(self.heart_rate_btn, 1, 0)
        control_layout.addWidget(self.reset_btn, 1, 1)

        # 我们可以为列和行设置最小宽度和高度，确保它们均匀分布
        control_layout.setColumnStretch(0, 1)
        control_layout.setColumnStretch(1, 1)
        
        return control_group
    
    def create_info_display(self):
        """创建信息显示组"""
        info_group = QGroupBox("📊 实时监测数据")
        info_layout = QVBoxLayout(info_group)
        info_layout.setContentsMargins(15, 20, 15, 15)
        info_layout.setSpacing(10)
        
        # 创建2×2网格布局的信息卡片容器
        cards_widget = QWidget()
        cards_layout = QGridLayout(cards_widget)
        cards_layout.setSpacing(8)
        cards_layout.setContentsMargins(0, 0, 0, 0)
        
        # 当前表情卡片 (第1行第1列)
        emotion_card = self.create_info_card("🎭", "当前表情", "未检测", "#e74c3c")
        self.emotion_label = emotion_card.findChild(QLabel, "value_label")
        cards_layout.addWidget(emotion_card, 0, 0)
        
        # 当前心率卡片 (第1行第2列)
        hr_card = self.create_info_card("💓", "心率监测", "-- BPM", "#e67e22")
        self.heart_rate_label = hr_card.findChild(QLabel, "value_label")
        cards_layout.addWidget(hr_card, 0, 1)
        
        # 人脸数量卡片 (第2行第1列)
        face_card = self.create_info_card("👤", "检测人脸", "0", "#3498db")
        self.face_count_label = face_card.findChild(QLabel, "value_label")
        cards_layout.addWidget(face_card, 1, 0)
        
        # 运行时间卡片 (第2行第2列)
        runtime_card = self.create_info_card("⏱️", "运行时长", "00:00:00", "#27ae60")
        self.runtime_label = runtime_card.findChild(QLabel, "value_label")
        cards_layout.addWidget(runtime_card, 1, 1)
        
        info_layout.addWidget(cards_widget)
        return info_group
    
    def create_info_card(self, icon, title, value, color):
        """创建信息卡片 (样式微调)"""
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {color}, stop:1 rgba({self.hex_to_rgb(color)}, 0.8));
                border-radius: 8px;
                padding: 5px; /* 减小内边距 */
                margin: 1px;
                border: 1px solid rgba(255, 255, 255, 0.2);
            }}
        """)
        
        layout = QHBoxLayout(card)
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setSpacing(10)
        
        # 图标标签
        icon_label = QLabel(icon)
        icon_label.setStyleSheet("""
            QLabel {
                font-size: 18px; /* 减小图标字体大小 */
                color: white;
                background: none;
                border: none;
                min-width: 25px; /* 减小图标宽度 */
                max-width: 25px;
            }
        """)
        icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon_label)
        
        # 信息布局 - 使用伸缩项实现垂直居中
        info_container = QWidget()
        info_container.setStyleSheet("background: none; border: none;") # 确保容器透明
        info_layout = QVBoxLayout(info_container)
        info_layout.setSpacing(2)
        info_layout.setContentsMargins(0, 0, 0, 0)
        
        # 添加顶部伸缩项
        info_layout.addStretch()

        # 标题标签
        title_label = QLabel(title)
        title_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 9px; /* 减小标题字体 */
                font-weight: bold;
                background: none;
                border: none;
            }
        """)
        info_layout.addWidget(title_label)
        
        # 数值标签
        value_label = QLabel(value)
        value_label.setObjectName("value_label")
        value_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 12px; /* 稍微减小数值字体 */
                font-weight: bold;
                background: none;
                border: none;
            }
        """)
        info_layout.addWidget(value_label)

        # 添加底部伸缩项
        info_layout.addStretch()
        
        layout.addWidget(info_container)
        
        return card
    
    def hex_to_rgb(self, hex_color):
        """将十六进制颜色转换为RGB"""
        hex_color = hex_color.lstrip('#')
        return ', '.join(str(int(hex_color[i:i+2], 16)) for i in (0, 2, 4))
        
    def create_charts(self):
        """创建图表组"""
        charts_group = QGroupBox("📈 数据可视化分析")
        charts_layout = QVBoxLayout(charts_group)
        charts_layout.setContentsMargins(15, 20, 15, 15)
        charts_layout.setSpacing(12)
        
        # 表情统计图
        emotion_title = QLabel("🎭 表情分布统计")
        emotion_title.setStyleSheet("""
            QLabel {
                font-size: 11px;
                font-weight: bold;
                color: white;
                padding: 6px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #e74c3c, stop:1 #c0392b);
                border-radius: 6px;
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
        """)
        emotion_title.setAlignment(Qt.AlignCenter)
        charts_layout.addWidget(emotion_title)
        
        self.emotion_chart = EmotionChart()
        charts_layout.addWidget(self.emotion_chart, 1) # 分配伸缩因子
        
        # 心率趋势图
        hr_title = QLabel("💓 心率变化趋势")
        hr_title.setStyleSheet("""
            QLabel {
                font-size: 11px;
                font-weight: bold;
                color: white;
                padding: 6px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #e67e22, stop:1 #d35400);
                border-radius: 6px;
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
        """)
        hr_title.setAlignment(Qt.AlignCenter)
        charts_layout.addWidget(hr_title)
        
        self.heart_rate_chart = HeartRateChart()
        charts_layout.addWidget(self.heart_rate_chart, 1) # 分配伸缩因子
        
        return charts_group
        
    def create_status_bar(self):
        """创建状态栏"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
        
        self.status_bar.showMessage("就绪")
        
    def toggle_camera(self):
        """切换摄像头状态"""
        if not self.processing_thread or not self.processing_thread.isRunning():
            # 启动摄像头
            self.processing_thread = ProcessingThread()
            
            # 连接信号
            self.processing_thread.frame_updated.connect(self.update_frame)
            self.processing_thread.emotion_updated.connect(self.update_emotion)
            self.processing_thread.hr_updated.connect(self.update_heart_rate)
            self.processing_thread.status_updated.connect(self.update_status)
            self.processing_thread.stats_updated.connect(self.update_stats)
            self.processing_thread.face_count_updated.connect(self.update_face_count)  # 新增：连接人脸数量信号
            
            if self.processing_thread.start_processing():
                self.camera_btn.setText("📹 关闭摄像头")
                self.emotion_btn.setEnabled(True)
                self.heart_rate_btn.setEnabled(True)
                self.reset_btn.setEnabled(True)
            else:
                self.processing_thread = None
        else:
            # 关闭摄像头
            self.processing_thread.stop_processing()
            self.processing_thread = None
            
            self.camera_btn.setText("📹 打开摄像头")
            self.emotion_btn.setEnabled(False)
            self.heart_rate_btn.setEnabled(False)
            self.reset_btn.setEnabled(False)
            self.emotion_btn.setText("🎭 开始表情识别")
            self.heart_rate_btn.setText("💓 开始心率检测")
            
            # 重置显示
            self.video_label.setText('🎥 点击 "打开摄像头" 开始实时分析\n\n系统将自动检测人脸并进行情感与心率分析')
            self.emotion_label.setText("未检测")
            self.heart_rate_label.setText("-- BPM")
            self.face_count_label.setText("0")
            
    def toggle_emotion_recognition(self):
        """切换表情识别状态"""
        if self.processing_thread:
            enabled = not self.processing_thread.emotion_recognition_enabled
            self.processing_thread.emotion_recognition_enabled = enabled
            
            if enabled:
                self.emotion_btn.setText("🎭 停止表情识别")
                self.update_status("表情识别已开启")
            else:
                self.emotion_btn.setText("🎭 开始表情识别")
                self.update_status("表情识别已停止")
                
    def toggle_heart_rate_detection(self):
        """切换心率检测状态"""
        if self.processing_thread:
            enabled = not self.processing_thread.heart_rate_estimation_enabled
            self.processing_thread.heart_rate_estimation_enabled = enabled
            
            if enabled:
                self.heart_rate_btn.setText("💓 停止心率检测")
                self.update_status("心率检测已开启")
            else:
                self.heart_rate_btn.setText("💓 开始心率检测")
                self.update_status("心率检测已停止")
                
    def reset_statistics(self):
        """重置统计数据"""
        if self.processing_thread and self.processing_thread.stats_tracker:
            self.processing_thread.stats_tracker.reset_all_stats()
            
            # 清空图表
            self.emotion_chart.clear_chart()
            self.heart_rate_chart.clear_chart()
            
            self.update_status("统计数据已重置")
            
    def update_frame(self, frame):
        """更新视频帧显示"""
        try:
            # 将BGR转换为RGB，并确保内存连续
            if frame is None:
                return
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if not rgb_frame.flags.c_contiguous:
                rgb_frame = np.ascontiguousarray(rgb_frame)

            height, width, channel = rgb_frame.shape
            bytes_per_line = channel * width

            # 通过bytes创建QImage，避免memoryview类型不兼容
            buffer = rgb_frame.tobytes()
            q_image = QImage(buffer, width, height, bytes_per_line, QImage.Format_RGB888)
            
            # 缩放以适应标签大小
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            self.video_label.setPixmap(scaled_pixmap)
            
        except Exception as e:
            print(f"更新帧显示错误: {e}")
            
    def update_emotion(self, emotion, emotion_stats):
        """更新表情信息"""
        self.emotion_label.setText(emotion)
        self.emotion_chart.update_chart(emotion_stats)
        
    def update_heart_rate(self, bpm, hr_history):
        """更新心率信息"""
        self.heart_rate_label.setText(f"{bpm:.1f} BPM")
        self.heart_rate_chart.update_chart(hr_history)
    
    def update_face_count(self, count):
        """更新人脸数量"""
        self.face_count_label.setText(str(count))
        
    def update_status(self, message):
        """更新状态栏信息"""
        self.status_bar.showMessage(message)
        
    def update_stats(self, stats):
        """更新统计信息"""
        # 更新运行时间
        if 'session_duration' in stats:
            duration = int(stats['session_duration'])
            hours = duration // 3600
            minutes = (duration % 3600) // 60
            seconds = duration % 60
            self.runtime_label.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
            
    def update_ui(self):
        """定期更新UI"""
        # 这里可以添加需要定期更新的UI元素
        pass
        
    def closeEvent(self, event):
        """窗口关闭事件"""
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop_processing()
            
        event.accept()

def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用信息
    app.setApplicationName("CV情感心率分析系统")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("Caria_Tarnished")
    
    # 创建主窗口
    window = MainWindow()
    window.show()
    
    # 运行应用
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
