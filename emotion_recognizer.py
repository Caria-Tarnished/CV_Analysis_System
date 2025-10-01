#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
表情识别模块
基于深度学习的人脸表情识别
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, Union
import os

class SimpleCNN(nn.Module):
    """
    简单的CNN表情识别模型
    适用于FER2013数据集 (48x48灰度图像，7种表情)
    """
    
    def __init__(self, num_classes=7):
        super(SimpleCNN, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # 第一组卷积
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        
        # 第二组卷积
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        
        # 第三组卷积
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout1(x)
        
        # 展平
        x = x.view(-1, 128 * 6 * 6)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x

class EmotionRecognizer:
    """
    表情识别器
    支持多种预训练模型格式
    """
    
    # FER2013数据集的表情标签
    EMOTION_LABELS = {
        0: 'angry',      # 愤怒
        1: 'disgust',    # 厌恶
        2: 'fear',       # 恐惧
        3: 'happy',      # 高兴
        4: 'sad',        # 悲伤
        5: 'surprise',   # 惊讶
        6: 'neutral'     # 中性
    }
    
    # 中文标签映射
    EMOTION_LABELS_CN = {
        'angry': '愤怒',
        'disgust': '厌恶',
        'fear': '恐惧',
        'happy': '高兴',
        'sad': '悲伤',
        'surprise': '惊讶',
        'neutral': '中性'
    }
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu', confidence_threshold: float = 0.3):
        """
        初始化表情识别器
        
        Args:
            model_path: 预训练模型路径 (.pth, .onnx, .h5)
            device: 运行设备 ('cpu' 或 'cuda')
            confidence_threshold: 置信度阈值，低于此值则返回neutral
        """
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.input_size = (48, 48)  # FER2013标准输入尺寸
        self.model_type = None
        self.confidence_threshold = confidence_threshold
        
        # 历史预测，用于时间平滑
        self.prediction_history = []
        self.history_size = 5
        
        # 初始化模型
        self._load_model()
        
        print(f"? 表情识别器初始化完成，使用设备: {self.device}")
        print(f"  置信度阈值: {confidence_threshold}")
        
    def _load_model(self):
        """加载预训练模型"""
        if self.model_path and os.path.exists(self.model_path):
            self._load_pretrained_model()
        else:
            self._create_default_model()
            
    def _load_pretrained_model(self):
        """加载预训练模型文件"""
        try:
            if self.model_path.endswith('.pth'):
                self._load_pytorch_model()
            elif self.model_path.endswith('.onnx'):
                self._load_onnx_model()
            elif self.model_path.endswith('.h5'):
                self._load_tensorflow_model()
            else:
                raise ValueError(f"不支持的模型格式: {self.model_path}")
                
        except Exception as e:
            print(f"加载预训练模型失败: {e}")
            print("使用默认模型...")
            self._create_default_model()
            
    def _load_pytorch_model(self):
        """加载PyTorch模型"""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        if isinstance(checkpoint, dict):
            # 如果是完整的checkpoint
            self.model = SimpleCNN()
            self.model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        else:
            # 如果直接是模型
            self.model = checkpoint
            
        self.model.to(self.device)
        self.model.eval()
        self.model_type = 'pytorch'
        print(f"? 成功加载PyTorch模型: {self.model_path}")
        
    def _load_onnx_model(self):
        """加载ONNX模型"""
        try:
            import onnxruntime as ort
            self.model = ort.InferenceSession(self.model_path)
            self.model_type = 'onnx'
            print(f"? 成功加载ONNX模型: {self.model_path}")
        except ImportError:
            raise ImportError("需要安装onnxruntime: pip install onnxruntime")
            
    def _load_tensorflow_model(self):
        """加载TensorFlow/Keras模型"""
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(self.model_path)
            self.model_type = 'tensorflow'
            print(f"成功加载TensorFlow模型: {self.model_path}")
        except ImportError:
            raise ImportError("需要安装tensorflow: pip install tensorflow")
            
    def _create_default_model(self):
        """创建默认模型（随机权重，仅用于演示）"""
        self.model = SimpleCNN()
        self.model.to(self.device)
        self.model.eval()
        self.model_type = 'pytorch'
        print("使用默认随机权重模型（仅用于演示）")
        print("建议下载预训练模型以获得更好的效果")
        
    def _preprocess(self, face_roi: np.ndarray) -> torch.Tensor:
        """
        预处理人脸ROI
        
        Args:
            face_roi: 人脸区域图像
            
        Returns:
            torch.Tensor: 预处理后的张量
        """
        # 转换为灰度图
        if len(face_roi.shape) == 3:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_roi.copy()
            
        # 调整尺寸
        resized = cv2.resize(gray, self.input_size)
        
        # 直方图均衡化，增强对比度
        equalized = cv2.equalizeHist(resized)
        
        # 归一化到[0, 1]
        normalized = equalized.astype(np.float32) / 255.0
        
        # 标准化 (零均值，单位方差)
        normalized = (normalized - 0.485) / 0.229  # 使用ImageNet统计值
        
        # 转换为PyTorch张量并添加批次维度
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        return tensor.to(self.device)
        
    def _predict_raw(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Tuple[str, float]:
        """
        原始预测方法，返回表情和置信度
        
        Args:
            frame: 完整帧图像
            face_bbox: 人脸边界框 (x, y, w, h)
            
        Returns:
            Tuple[str, float]: (表情标签, 置信度)
        """
        try:
            # 提取人脸ROI
            x, y, w, h = face_bbox
            
            # 确保边界框在图像范围内
            h_img, w_img = frame.shape[:2]
            x = max(0, min(x, w_img - 1))
            y = max(0, min(y, h_img - 1))
            w = max(1, min(w, w_img - x))
            h = max(1, min(h, h_img - y))
            
            face_roi = frame[y:y+h, x:x+w]
            
            if face_roi.size == 0:
                return 'neutral', 0.0
                
            # 预处理
            input_tensor = self._preprocess(face_roi)
            
            # 推理
            with torch.no_grad():
                if self.model_type == 'pytorch':
                    outputs = self.model(input_tensor)
                    probabilities = F.softmax(outputs, dim=1)
                    confidence, predicted_idx = torch.max(probabilities, dim=1)
                    confidence = confidence.item()
                    predicted_idx = predicted_idx.item()
                    
                elif self.model_type == 'onnx':
                    input_array = input_tensor.cpu().numpy()
                    outputs = self.model.run(None, {'input': input_array})[0]
                    probabilities = torch.softmax(torch.from_numpy(outputs[0]), dim=0)
                    confidence, predicted_idx = torch.max(probabilities, dim=0)
                    confidence = confidence.item()
                    predicted_idx = predicted_idx.item()
                    
                elif self.model_type == 'tensorflow':
                    input_array = input_tensor.cpu().numpy()
                    outputs = self.model.predict(input_array, verbose=0)
                    probabilities = torch.softmax(torch.from_numpy(outputs[0]), dim=0)
                    confidence, predicted_idx = torch.max(probabilities, dim=0)
                    confidence = confidence.item()
                    predicted_idx = predicted_idx.item()
                    
                else:
                    return 'neutral', 0.0
                    
            return self.EMOTION_LABELS.get(predicted_idx, 'neutral'), confidence
            
        except Exception as e:
            print(f"表情识别错误: {e}")
            return 'neutral', 0.0
    
    def predict(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int], use_smoothing: bool = True) -> str:
        """
        预测表情
        
        Args:
            frame: 完整帧图像
            face_bbox: 人脸边界框 (x, y, w, h)
            use_smoothing: 是否使用时间平滑
            
        Returns:
            str: 预测的表情标签
        """
        emotion, confidence = self._predict_raw(frame, face_bbox)
        
        # 使用置信度阈值过滤
        if confidence < self.confidence_threshold:
            emotion = 'neutral'
            
        # 时间平滑
        if use_smoothing:
            self.prediction_history.append(emotion)
            if len(self.prediction_history) > self.history_size:
                self.prediction_history.pop(0)
                
            # 返回历史中最常见的表情
            from collections import Counter
            emotion_counts = Counter(self.prediction_history)
            emotion = emotion_counts.most_common(1)[0][0]
            
        return emotion
            
    def predict_with_confidence(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Tuple[str, float]:
        """
        预测表情并返回置信度
        
        Args:
            frame: 完整帧图像
            face_bbox: 人脸边界框 (x, y, w, h)
            
        Returns:
            Tuple[str, float]: (表情标签, 置信度)
        """
        return self._predict_raw(frame, face_bbox)
            
    def get_all_emotions(self) -> Dict[int, str]:
        """获取所有支持的表情标签"""
        return self.EMOTION_LABELS.copy()
        
    def reset_history(self):
        """重置预测历史"""
        self.prediction_history.clear()
        
    def get_chinese_label(self, emotion: str) -> str:
        """获取表情的中文标签"""
        return self.EMOTION_LABELS_CN.get(emotion, emotion)
    
    def set_confidence_threshold(self, threshold: float):
        """设置置信度阈值"""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        print(f"置信度阈值已设置为: {self.confidence_threshold}")

# 模型下载和准备函数
def download_pretrained_model(model_type: str = 'simple') -> str:
    """
    下载预训练模型
    
    Args:
        model_type: 模型类型
        
    Returns:
        str: 模型文件路径
    """
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"准备下载 {model_type} 模型...")
    print("由于模型文件较大，这里提供模型URL供手动下载：")
    
    if model_type == 'simple':
        model_url = "https://github.com/example/fer2013_simple_cnn.pth"
        model_path = os.path.join(models_dir, "fer2013_simple_cnn.pth")
        
    print(f"模型URL: {model_url}")
    print(f"请下载并保存到: {model_path}")
    
    return model_path

# 测试代码
if __name__ == "__main__":
    print("测试表情识别器...")
    
    # 创建表情识别器
    recognizer = EmotionRecognizer()
    
    # 测试所有表情标签
    print("\n支持的表情:")
    for idx, emotion in recognizer.get_all_emotions().items():
        chinese = recognizer.get_chinese_label(emotion)
        print(f"{idx}: {emotion} ({chinese})")
        
    # 如果有摄像头，进行实时测试
    try:
        import sys
        sys.path.append('.')
        from face_detector import FaceDetector
        
        print("\n开始实时表情识别测试...")
        print("按 'q' 退出")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("无法打开摄像头")
        else:
            face_detector = FaceDetector('haar')
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # 检测人脸
                faces = face_detector.detect(frame)
                
                # 对每个人脸进行表情识别
                for face_bbox in faces:
                    x, y, w, h = face_bbox
                    
                    # 绘制人脸框
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    
                    # 预测表情
                    emotion, confidence = recognizer.predict_with_confidence(frame, face_bbox)
                    chinese_emotion = recognizer.get_chinese_label(emotion)
                    
                    # 显示表情标签
                    label = f"{chinese_emotion} ({confidence:.2f})"
                    cv2.putText(frame, label, (x, y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # 显示图像
                cv2.imshow('表情识别测试', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            cap.release()
            cv2.destroyAllWindows()
            
    except ImportError as e:
        print(f"依赖模块未安装: {e}")
        
    print("表情识别器测试完成")
