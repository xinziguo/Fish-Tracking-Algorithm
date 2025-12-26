from typing import Optional, Tuple, List, Union
import os
import glob
import cv2
import numpy as np
import csv
from datetime import timedelta
from dataclasses import dataclass
from simple_log_helper import CustomLogger
import time

@dataclass
class DetectionResult:
    """存储鱼类检测结果的数据类"""
    box: np.ndarray
    centroid: Tuple[int, int]
    rect: Tuple[Tuple[float, float], Tuple[float, float], float]
    timestamp: float

class FishDetector:
    """鱼类检测器类，用于从视频中检测和跟踪鱼类
    
    Attributes:
        display_scale (float): 显示窗口的缩放比例
        logger (CustomLogger): 日志记录器
        background (np.ndarray): 背景模型图像
        roi_mask (np.ndarray): 感兴趣区域掩码
    """
    
    def __init__(
        self, 
        display_scale: float = 0.5,
        logger: Optional[CustomLogger] = None,
        background: Optional[np.ndarray] = None,
        mask_path: Optional[str] = None
    ):
        """初始化鱼类检测器
        
        Args:
            display_scale: 显示窗口的缩放比例
            logger: 日志记录器实例
            background: 预计算的背景模型
            mask_path: ROI掩码文件路径
        """
        self.display_scale = display_scale
        self.logger = logger or CustomLogger(__name__, log_filename='Logs/fish_tracking.log')
        self.background = background
        self.current_mask = None
        
        # 运动检测参数
        self._frame_buffer = []
        self._buffer_size = 3
        self._last_detection = None
        self._motion_threshold = 1000
        self._diff_threshold = 30
        
        # 加载ROI掩码
        self.roi_mask = self._load_roi_mask(mask_path) if mask_path else None

    def _load_roi_mask(self, mask_path: str) -> Optional[np.ndarray]:
        """加载并预处理ROI掩码
        
        Args:
            mask_path: 掩码文件路径
            
        Returns:
            处理后的二值掩码图像，如果加载失败则返回None
        """
        if not os.path.exists(mask_path):
            self.logger.warning(f"Mask file not found at {mask_path}")
            return None
            
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            self.logger.warning("Failed to load mask file")
            return None
            
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return mask

    def _compute_foreground_mask(self, frame: np.ndarray) -> np.ndarray:
        """计算前景掩码
        
        使用双阈值背景差分方法计算前景掩码
        
        Args:
            frame: 输入帧
            
        Returns:
            前景掩码
        """
        # 背景差分
        diff = cv2.absdiff(frame, self.background)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # 双阈值处理
        _, fg_mask1 = cv2.threshold(gray_diff, 10, 255, cv2.THRESH_BINARY)
        _, fg_mask2 = cv2.threshold(gray_diff, 25, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.bitwise_and(fg_mask1, fg_mask2)
        
        # 应用ROI掩码
        if self.roi_mask is not None:
            fg_mask = cv2.bitwise_and(fg_mask, self.roi_mask)
        
        # 形态学处理
        kernel = np.ones((6, 6), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)
        fg_mask = cv2.erode(fg_mask, kernel, iterations=1)
        
        return fg_mask

    def _filter_contours(
        self, 
        contours: List[np.ndarray], 
        frame_shape: Tuple[int, int]
    ) -> List[np.ndarray]:
        """过滤检测到的轮廓
        
        Args:
            contours: 待过滤的轮廓列表
            frame_shape: 帧的形状 (height, width)
            
        Returns:
            过滤后的有效轮廓列表
        """
        height, width = frame_shape
        valid_contours = []
        
        for contour in contours:
            # 基本面积过滤
            if cv2.contourArea(contour) < 300:
                continue
                
            # 创建轮廓掩码
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            
            # 计算投影
            vertical_proj = np.sum(mask > 0, axis=0)
            horizontal_proj = np.sum(mask > 0, axis=1)
            
            # 检查连续性
            if (np.max(vertical_proj) > height * 0.7 or 
                np.max(horizontal_proj) > width * 0.7):
                continue
                
            # 检查形状
            rect = cv2.minAreaRect(contour)
            width_rect, length_rect = sorted(rect[1])
            if length_rect > 0 and width_rect / length_rect < 0.1:
                continue
                
            valid_contours.append(contour)
            
        return valid_contours

    def detect_fish(self, frame: np.ndarray) -> Optional[DetectionResult]:
        """检测单帧中的鱼类
        
        Args:
            frame: 输入视频帧
            
        Returns:
            DetectionResult 或 None (如果没有检测到鱼类)
        """
        if self.background is None:
            self.logger.warning("No background model found, using first frame as background")
            self.background = frame.copy()
            return None
            
        # 计算前景掩码
        fg_mask = self._compute_foreground_mask(frame)
        self.current_mask = fg_mask
        
        # 检查运动状态
        motion_pixels = np.sum(fg_mask > 0)
        frame_motion = self._detect_motion(frame)
        
        # 如果没有显著运动且存在上一帧检测结果，返回上一帧结果
        if not frame_motion and motion_pixels < self._motion_threshold and self._last_detection is not None:
            return self._last_detection
        
        # 寻找轮廓
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤轮廓
        valid_contours = self._filter_contours(contours, frame.shape[:2])
        
        if not valid_contours:
            return None
            
        # 选择最佳轮廓
        fish_contour = self._select_best_contour(valid_contours)
        if fish_contour is None:
            return None
            
        # 计算检测结果
        timestamp = time.time()
        result = self._compute_detection_result(fish_contour, timestamp)
        self._last_detection = result
        
        return result

    def _detect_motion(self, frame: np.ndarray) -> bool:
        """检测帧间运动
        
        使用帧差分法检测连续帧之间的运动
        
        Args:
            frame: 当前帧
            
        Returns:
            bool: 是否检测到显著运动
        """
        if len(self._frame_buffer) < self._buffer_size:
            self._frame_buffer.append(frame)
            return True
        
        # 计算帧差分
        self._frame_buffer.append(frame)
        gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in self._frame_buffer]
        
        # 使用numpy计算连续帧之间的差分
        diffs = np.array([
            np.mean(cv2.absdiff(gray_frames[i], gray_frames[i+1]))
            for i in range(self._buffer_size)
        ])
        
        # 移除最老的帧
        self._frame_buffer.pop(0)
        
        return np.any(diffs > self._diff_threshold)

    def _select_best_contour(
        self, 
        valid_contours: List[np.ndarray]
    ) -> Optional[np.ndarray]:
        """从有效轮廓中选择最佳轮廓
        
        基于以下标准选择最佳轮廓：
        1. 如果只有一个轮廓，直接返回
        2. 如果有上一帧检测结果，选择最接近的轮廓
        3. 否则选择最大的轮廓
        
        Args:
            valid_contours: 过滤后的有效轮廓列表
            
        Returns:
            最佳轮廓或None
        """
        if not valid_contours:
            return None
            
        if len(valid_contours) == 1:
            return valid_contours[0]
            
        if self._last_detection is not None:
            last_centroid = self._last_detection.centroid
            
            # 计算所有轮廓到上一个检测位置的距离
            distances = []
            centroids = []
            
            for contour in valid_contours:
                M = cv2.moments(contour)
                if M['m00'] == 0:
                    continue
                    
                centroid = (
                    int(M['m10'] / M['m00']), 
                    int(M['m01'] / M['m00'])
                )
                centroids.append(centroid)
                
                distance = np.sqrt(
                    (centroid[0] - last_centroid[0])**2 + 
                    (centroid[1] - last_centroid[1])**2
                )
                distances.append(distance)
            
            if distances:
                min_dist_idx = np.argmin(distances)
                if distances[min_dist_idx] < 100:  # 距离阈值
                    return valid_contours[min_dist_idx]
        
        # 如果没有合适的近邻轮廓，返回最大的轮廓
        return max(valid_contours, key=cv2.contourArea)

    def _compute_detection_result(
        self, 
        contour: np.ndarray,
        timestamp: float
    ) -> DetectionResult:
        """计算检测结果的特征
        
        Args:
            contour: 检测到的鱼类轮廓
            timestamp: 当前时间戳
            
        Returns:
            DetectionResult: 包含所有检测特征的结果对象
        """
        # 计算最小外接矩形
        rect = cv2.minAreaRect(contour)
        box = np.int0(cv2.boxPoints(rect))
        
        # 计算质心
        M = cv2.moments(contour)
        centroid = (
            int(M['m10'] / M['m00']), 
            int(M['m01'] / M['m00'])
        )
        
        return DetectionResult(
            box=box,
            centroid=centroid,
            rect=rect,
            timestamp=timestamp
        )

    def process_video(
        self,
        video_path: str,
        output_dir: str,
        frame_interval: int = 1,
        show_video: bool = False
    ) -> None:
        """处理单个视频文件
        
        Args:
            video_path: 输入视频路径
            output_dir: 输出目录
            frame_interval: 处理帧间隔
            show_video: 是否显示处理过程
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"Error opening video file: {video_path}")
            return

        # 获取视频信息
        video_info = self._get_video_info(cap)
        output_paths = self._prepare_output_files(video_path, output_dir, video_info)
        
        # 初始化跟踪数据
        tracking_data = self._initialize_tracking()
        
        try:
            self._process_video_frames(
                cap, output_paths, video_info, tracking_data,
                frame_interval, show_video
            )
        finally:
            self._cleanup_resources(cap, output_paths['video'])
            
        # 保存跟踪数据
        if tracking_data['fish_data']:
            self._save_tracking_data(tracking_data['fish_data'], output_paths['csv'])
            self.logger.info(f"Processing completed for {video_path}")
        else:
            self.logger.warning(f"No fish detected in video: {video_path}")

    def _get_video_info(self, cap: cv2.VideoCapture) -> dict:
        """获取视频基本信息"""
        return {
            'fps': int(cap.get(cv2.CAP_PROP_FPS)),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        }

    def _prepare_output_files(
        self,
        video_path: str,
        output_dir: str,
        video_info: dict
    ) -> dict:
        """准备输出文件"""
        base_name = os.path.basename(video_path).replace('.mp4', '')
        output_paths = {
            'video': os.path.join(output_dir, f'{base_name}_output.mp4'),
            'csv': os.path.join(output_dir, f'{base_name}_data.csv')
        }
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            output_paths['video'], fourcc,
            video_info['fps'],
            (video_info['width'], video_info['height'])
        )
        
        return {**output_paths, 'writer': video_writer}

    def _process_video_frames(
        self,
        cap: cv2.VideoCapture,
        output_paths: dict,
        video_info: dict,
        tracking_data: dict,
        frame_interval: int,
        show_video: bool
    ) -> None:
        """处理视频的所有帧"""
        processed_frames = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            processed_frames += 1
            if processed_frames % frame_interval != 0:
                continue
                
            # 检测鱼类
            result = self.detect_fish(frame)
            
            # 更新跟踪数据和显示
            if result is not None:
                self._update_tracking_data(
                    result, tracking_data,
                    processed_frames, video_info['fps']
                )
                frame = self._draw_detection_results(
                    frame, result, tracking_data['centroids']
                )
            
            output_paths['writer'].write(frame)
            
            if show_video:
                self._show_processing_window(frame)
            
            if processed_frames % 100 == 0:
                self.logger.info(
                    f"Processed {processed_frames}/{video_info['frame_count']} frames"
                )

    @staticmethod
    def generate_background_model(
        video_path: str,
        output_path: Optional[str] = None,
        num_frames: int = 500
    ) -> Optional[np.ndarray]:
        """生成背景模型
        
        Args:
            video_path: 输入视频路径
            output_path: 背景模型保存路径
            num_frames: 用于生成背景的帧数
            
        Returns:
            背景模型图像，失败则返回None
        """
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=num_frames,
            varThreshold=13,
            detectShadows=False
        )
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video file {video_path}")
            return None
        
        try:
            background = FishDetector._process_background_frames(
                cap, bg_subtractor, num_frames
            )
            
            if background is not None and output_path:
                cv2.imwrite(output_path, background)
                print(f"Background model saved to {output_path}")
                
            return background
            
        finally:
            cap.release()

    @staticmethod
    def _process_background_frames(
        cap: cv2.VideoCapture,
        bg_subtractor: cv2.BackgroundSubtractor,
        num_frames: int
    ) -> Optional[np.ndarray]:
        """处理视频帧以生成背景模型"""
        for frame_idx in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
                
            bg_subtractor.apply(frame)
            
            if frame_idx % 50 == 0:
                print(f"Processing frame {frame_idx}/{num_frames}")
        
        background = bg_subtractor.getBackgroundImage()
        if background is None:
            print("Error: Failed to generate background model")
            
        return background

    def _initialize_tracking(self) -> dict:
        """初始化跟踪数据结构"""
        return {
            'centroids': [],
            'fish_data': []
        }

    def _update_tracking_data(
        self,
        result: DetectionResult,
        tracking_data: dict,
        frame_count: int,
        fps: int
    ) -> None:
        """更新跟踪数据"""
        tracking_data['centroids'].append(result.centroid)
        
        time = timedelta(seconds=frame_count/fps)
        tracking_data['fish_data'].append([
            time
        ])

    def _draw_detection_results(
        self,
        frame: np.ndarray,
        result: DetectionResult,
        centroids: List[Tuple[int, int]]
    ) -> np.ndarray:
        """绘制检测结果"""
        # 绘制检测框
        cv2.drawContours(frame, [result.box], -1, (0, 255, 0), 2)
        
        # 绘制质心
        for centroid in centroids:
            cv2.circle(frame, centroid, 5, (0, 0, 255), -1)
        
        return frame

    def _show_processing_window(self, frame: np.ndarray) -> None:
        """显示处理窗口"""
        cv2.imshow('Processing', frame)
        cv2.waitKey(1)

    def _cleanup_resources(self, cap: cv2.VideoCapture, video_path: str) -> None:
        """清理资源"""
        cap.release()
        cv2.destroyAllWindows()
