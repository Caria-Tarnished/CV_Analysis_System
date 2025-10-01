#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人脸检测模块
支持多种人脸检测算法：Haar级联、dlib、OpenCV DNN等
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import os

class FaceDetector:
    """
    人脸检测器
    支持多种检测方法和算法
    """
    
    def __init__(self, method: str = 'haar', model_path: Optional[str] = None):
        """
        初始化人脸检测器
        
        Args:
            method: 检测方法 ('haar', 'dlib', 'dnn', 'mtcnn')
            model_path: 自定义模型路径（可选）
        """
        self.method = method.lower()
        self.detector = None
        self.model_path = model_path
        
        # 初始化相应的检测器
        self._initialize_detector()
        
    def _initialize_detector(self):
        """初始化指定的人脸检测器"""
        
        if self.method == 'haar':
            self._init_haar_detector()
        elif self.method == 'dlib':
            self._init_dlib_detector()
        elif self.method == 'dnn':
            self._init_dnn_detector()
        elif self.method == 'mtcnn':
            self._init_mtcnn_detector()
        else:
            raise ValueError(f"不支持的检测方法: {self.method}")
            
    def _init_haar_detector(self):
        """初始化Haar级联检测器"""
        try:
            # 使用OpenCV内置的Haar级联分类器
            if self.model_path:
                classifier_path = self.model_path
            else:
                # 使用OpenCV内置的分类器
                classifier_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                
            if not os.path.exists(classifier_path):
                raise FileNotFoundError(f"Haar级联文件不存在: {classifier_path}")
                
            self.detector = cv2.CascadeClassifier(classifier_path)
            
            if self.detector.empty():
                raise ValueError("无法加载Haar级联分类器")
                
            print(f"? Haar级联检测器初始化成功: {classifier_path}")
            
        except Exception as e:
            print(f"? Haar级联检测器初始化失败: {e}")
            raise
            
    def _init_dlib_detector(self):
        """初始化dlib检测器"""
        try:
            import dlib
            self.detector = dlib.get_frontal_face_detector()
            print("? dlib检测器初始化成功")
            
        except ImportError:
            raise ImportError("dlib库未安装，请运行: pip install dlib")
        except Exception as e:
            print(f"? dlib检测器初始化失败: {e}")
            raise
            
    def _init_dnn_detector(self):
        """初始化OpenCV DNN检测器"""
        try:
            if self.model_path:
                model_path = self.model_path
                config_path = self.model_path.replace('.caffemodel', '.prototxt')
            else:
                # 这里需要下载预训练模型
                print("?? 请提供DNN模型路径或下载预训练模型")
                raise FileNotFoundError("需要DNN模型文件")
                
            self.detector = cv2.dnn.readNetFromCaffe(config_path, model_path)
            print("? OpenCV DNN检测器初始化成功")
            
        except Exception as e:
            print(f"? OpenCV DNN检测器初始化失败: {e}")
            raise
            
    def _init_mtcnn_detector(self):
        """初始化MTCNN检测器"""
        try:
            from mtcnn import MTCNN
            self.detector = MTCNN()
            print("? MTCNN检测器初始化成功")
            
        except ImportError:
            raise ImportError("MTCNN库未安装，请运行: pip install mtcnn")
        except Exception as e:
            print(f"? MTCNN检测器初始化失败: {e}")
            raise
            
    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        检测人脸
        
        Args:
            frame: 输入图像帧
            
        Returns:
            List[Tuple[int, int, int, int]]: 人脸边界框列表，格式为(x, y, w, h)
        """
        if frame is None or frame.size == 0:
            return []
            
        try:
            if self.method == 'haar':
                return self._detect_haar(frame)
            elif self.method == 'dlib':
                return self._detect_dlib(frame)
            elif self.method == 'dnn':
                return self._detect_dnn(frame)
            elif self.method == 'mtcnn':
                return self._detect_mtcnn(frame)
            else:
                return []
                
        except Exception as e:
            print(f"人脸检测过程中发生错误: {e}")
            return []
            
    def _detect_haar(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """使用Haar级联检测人脸"""
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        # 检测人脸
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,    # 图像缩放因子
            minNeighbors=5,     # 最小邻居数
            minSize=(30, 30),   # 最小人脸尺寸
            maxSize=(300, 300)  # 最大人脸尺寸
        )
        
        # 转换为列表格式
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
        
    def _detect_dlib(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """使用dlib检测人脸"""
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        # 检测人脸
        faces = self.detector(gray)
        
        # 转换为(x, y, w, h)格式
        face_boxes = []
        for face in faces:
            x = face.left()
            y = face.top()
            w = face.right() - face.left()
            h = face.bottom() - face.top()
            face_boxes.append((x, y, w, h))
            
        return face_boxes
        
    def _detect_dnn(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """使用OpenCV DNN检测人脸"""
        h, w = frame.shape[:2]
        
        # 创建blob
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123])
        
        # 设置输入
        self.detector.setInput(blob)
        
        # 前向传播
        detections = self.detector.forward()
        
        face_boxes = []
        confidence_threshold = 0.5
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > confidence_threshold:
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                
                face_boxes.append((x1, y1, x2 - x1, y2 - y1))
                
        return face_boxes
        
    def _detect_mtcnn(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """使用MTCNN检测人脸"""
        # MTCNN需要RGB格式
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if len(frame.shape) == 3 else frame
        
        # 检测人脸
        result = self.detector.detect_faces(rgb_frame)
        
        face_boxes = []
        for face in result:
            if face['confidence'] > 0.9:  # 置信度阈值
                x, y, w, h = face['box']
                face_boxes.append((x, y, w, h))
                
        return face_boxes
        
    def get_method(self) -> str:
        """获取当前使用的检测方法"""
        return self.method
        
    def set_parameters(self, **kwargs):
        """设置检测参数（针对不同方法）"""
        # 这里可以根据不同方法设置特定参数
        pass

# 测试代码
if __name__ == "__main__":
    print("测试人脸检测器...")
    
    # 测试不同的检测方法
    methods = ['haar']  # 首先测试Haar级联
    
    try:
        import dlib
        methods.append('dlib')
    except ImportError:
        print("?? dlib未安装，跳过dlib测试")
    
    for method in methods:
        print(f"\n测试 {method} 检测器:")
        
        try:
            detector = FaceDetector(method=method)
            
            # 使用摄像头测试
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("无法打开摄像头")
                continue
                
            print(f"按 'q' 退出 {method} 测试...")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # 检测人脸
                faces = detector.detect(frame)
                
                # 绘制检测结果
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, f"{method.upper()}", (x, y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # 显示信息
                cv2.putText(frame, f"Method: {method.upper()}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Faces: {len(faces)}", (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow(f'Face Detection - {method.upper()}', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            cap.release()
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"? {method} 检测器测试失败: {e}")
            
    print("人脸检测器测试完成")
