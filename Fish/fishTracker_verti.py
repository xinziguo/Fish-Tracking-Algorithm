import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import timedelta
from simple_log_helper import CustomLogger
from abc import ABC, abstractmethod
import multiprocessing
# from utils.styleTransfer import style_transfer
logger = CustomLogger(__name__, log_filename='Logs/fish_tracking.log')

class AccelerationStrategy(ABC):
    @abstractmethod
    def process_frame(self, frame):
        pass

class CPUAcceleration(AccelerationStrategy):
    def process_frame(self, frame):
        return frame

class OpenCLAcceleration(AccelerationStrategy):
    def __init__(self):
        cv2.ocl.setUseOpenCL(True)

    def process_frame(self, frame):
        return cv2.UMat(frame)

class FishTracker:
    def __init__(self, acceleration_strategy, display_scale=0.5, logger=None):
        self.acceleration_strategy = acceleration_strategy
        self.display_scale = display_scale
        self.logger = logger or CustomLogger(__name__, log_filename='Logs/fish_tracking.log')

    def create_mask(self, image):
        # 缓存转换后的图像格式，避免重复转换
        if hasattr(self, '_last_image') and self._last_image is image:
            return self._last_mask
            
        if isinstance(image, cv2.UMat):
            image = image.get()
        
        # 使用更高效的颜色空间转换
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # 使用位运算优化边界检测 # 创建掩码过滤三个通道值都在180附近的像素点
        mask_180 = cv2.inRange(image, np.array([175, 175, 175]), np.array([185, 185, 185]))
        lower_bound = np.array([120, 140, 180])
        upper_bound = np.array([200, 230, 255])
        mask = cv2.inRange(image, lower_bound, upper_bound)
        mask = cv2.bitwise_and(mask, ~mask_180)

        # 优化差分计算，使用Sobel算子替代
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # 应用阈值
        mask[gradient < 10] = 0
        
        # 缓存结果
        self._last_image = image
        self._last_mask = mask
        return mask

    def define_roi(self, image):
        symmetry_x, symmetry_y = 0, 0
        if isinstance(image, cv2.UMat):
            height, width = image.get().shape[:2]
        else:
            height, width = image.shape[:2]
        roi_mask = np.zeros((height, width), np.uint8)
        roi_mask[symmetry_y:, symmetry_x:] = 255
        return cv2.UMat(roi_mask) if isinstance(image, cv2.UMat) else roi_mask

    def distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def rectangles_overlap(self, rect1, rect2):
        # 将旋转矩形转换为轴对齐的边界框
        box1 = cv2.boxPoints(rect1)
        box2 = cv2.boxPoints(rect2)
        
        # 计算边界框的最小和最大坐标
        min1 = np.min(box1, axis=0)
        max1 = np.max(box1, axis=0)
        min2 = np.min(box2, axis=0)
        max2 = np.max(box2, axis=0)
        
        # 检查边界框是否重叠
        return not (max1[0] < min2[0] or min1[0] > max2[0] or
                    max1[1] < min2[1] or min1[1] > max2[1])
    def find_fish_features(self, mask):
        # 使用更高效的轮廓检测方法
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # 使用numpy操作替代列表解析，提高性能
        areas = np.array([cv2.contourArea(c) for c in contours])
        valid_mask = (areas > 5) & (areas < mask.shape[0] * mask.shape[1] / 2)
        valid_contours = [c for i, c in enumerate(contours) if valid_mask[i]]
        
        if not valid_contours:
            return None
            
        fish_contour = max(valid_contours, key=cv2.contourArea)
        
        # 一次性计算所需的特征
        rect = cv2.minAreaRect(fish_contour)
        box = np.int0(cv2.boxPoints(rect))
        M = cv2.moments(fish_contour)
        
        if M['m00'] == 0:
            return None
            
        centroid = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
        return box, centroid, rect

    def process_video(self, video_path, output_dir, frame_interval=1, show_video=False, logger=None):
        if logger:
            self.logger = logger
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            self.logger.info(f"Error opening video file: {video_path}")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 修改缩放逻辑，直接缩放到目标分辨率
        target_width, target_height = 1000, 600
        scale_x = target_width / original_width
        scale_y = target_height / original_height
        scale = min(scale_x, scale_y)
        
        width = int(original_width * scale)
        height = int(original_height * scale)
        
        self.logger.info(f"Processing video: {video_path}")
        self.logger.info(f"Original video size: {original_width}x{original_height}")
        self.logger.info(f"Scaled video size: {width}x{height}")
        self.logger.info(f"Total frames: {frame_count}")
        self.logger.info(f"FPS: {fps}")

        output_video = os.path.join(output_dir, os.path.basename(video_path).replace('.mov', '_output.mp4'))
        output_csv = os.path.join(output_dir, os.path.basename(video_path).replace('.mov', '_data.csv'))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # Use the scaled dimensions for the output video
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        
        centroids = []
        fish_data = []
        detected_fish_count = 0
        processed_frame_count = 0
        
        while cap.isOpened():
            # Skip frames according to frame_interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, processed_frame_count * frame_interval)
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame_count += 1

            # Scale the frame to 1/3 of its original size
            frame = cv2.resize(frame, (width, height))
            frame = self.acceleration_strategy.process_frame(frame)
            mask = self.create_mask(frame)
            # gray_image = style_transfer(frame)
            #save grag image as png
            # cv2.imwrite('Logs/gray_image.png', frame)
            # _, mask = cv2.threshold(frame, 140, 180, cv2.THRESH_BINARY)
            
            result = self.find_fish_features(mask.get() if isinstance(mask, cv2.UMat) else mask)
            # print(result)
            if result is not None:
                box, centroid, rect = result
                time = timedelta(seconds=processed_frame_count/fps)
                
                detected_fish_count += 1
                centroids.append(centroid)
                
                fish_data.append([time, centroid[0], centroid[1], rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]])
                
                cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
                cv2.circle(frame, centroid, 5, (0, 0, 255), -1)
                
                recent_positions = centroids[-100:]
                for i in range(1, len(recent_positions)):
                    cv2.line(frame, recent_positions[i-1], recent_positions[i], (0, 255, 255), 1)
                cv2.putText(frame, str(time), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                out.write(frame.get() if isinstance(frame, cv2.UMat) else frame)
                if show_video:
                    display_frame = cv2.resize(frame.get() if isinstance(frame, cv2.UMat) else frame, 
                                               (int(width * self.display_scale), int(height * self.display_scale)))
                    cv2.imshow('Fish Tracking', display_frame)

            if processed_frame_count % 100 == 0:
                self.logger.info(f"Processed {processed_frame_count}/{frame_count} frames")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        self.logger.info(f"Detected fish in {detected_fish_count} frames")
        self.logger.info(f"Total processed frames: {processed_frame_count}")
        self.logger.info(f"Output video saved to: {output_video}")

        if centroids:
            self.save_to_csv(fish_data, output_csv)
            self.logger.info(f"Processing completed successfully for {video_path}")
            self.logger.info(f"CSV data saved to: {output_csv}")
        else:
            self.logger.warning(f"No fish detected in the video: {video_path}")

    def save_to_csv(self, fish_data, output_file):
        with open(output_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Time", "Centroid_X", "Centroid_Y", "Rect_Center_X", "Rect_Center_Y", "Rect_Width", "Rect_Height", "Rect_Angle"])
            writer.writerows(fish_data)

def process_video_wrapper(args):
    video_file, output_dir, frame_interval, show_video, use_gpu = args
    acceleration_strategy = OpenCLAcceleration() if use_gpu else CPUAcceleration()
    
    
    # Create a process-specific logger
    process_logger = CustomLogger(f"{__name__}.{multiprocessing.current_process().name}", 
                                  log_filename='Logs/fish_tracking.log')
    fish_tracker = FishTracker(acceleration_strategy,logger=process_logger)
    process_logger.info(f"Starting processing of video: {video_file}")
    fish_tracker.process_video(video_file, output_dir, frame_interval, show_video, process_logger)
    process_logger.info(f"Finished processing of video: {video_file}")

def process_videos(input_dir, output_dir, frame_interval, show_video=False, max_workers=4, use_gpu=False):
    os.makedirs(output_dir, exist_ok=True)
    video_files = glob.glob(os.path.join(input_dir, '*.mov'))
    
    # Debug: Check the number of video files found
    logger.info(f"Found {len(video_files)} video files to process")
    if len(video_files) == 0:
        logger.warning("No video files found in the input directory.")
        return

    with multiprocessing.Pool(processes=max_workers) as pool:
        args_list = [(video_file, output_dir, frame_interval, show_video, use_gpu) for video_file in video_files]
        pool.map(process_video_wrapper, args_list)
    
    logger.info("All videos have been processed")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Fish Tracking Tool")
    # parser.add_argument("--input_dir", default='data', help="Directory containing input video files")
    # parser.add_argument("--output_dir", default='output', help="Directory to save output files")
    # parser.add_argument("--frame_interval", type=int, default=1, help="Interval of frames to process (default: 1)")m n
    # parser.add_argument("--show_video", type=bool, default=False, help="Show video frames")
    # parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of worker threads")
    # parser.add_argument("--use_gpu", type=bool, default=False, help="Use GPU accelerati
    # .on (OpenCL)")
    # args = parser.parse_args()

    # process_videos(args.input_dir, args.output_dir, args.frame_interval, args.show_video, args.max_workers, args.use_gpu)
    # max_workers = multiprocessing.cpu_count()
    # max_workers = 1
    # # process_videos('data/hori', 'output/hori', 1, False, max_workers, False)
    # process_videos('data/fix', 'output/fix', 1, False, max_workers, False)
    tracker = FishTracker(OpenCLAcceleration(), display_scale=0.5)
    tracker.process_video('data/verti/IMG_4252.mov', 'output/fix', 1, True, CustomLogger(__name__, log_filename='Logs/fish_tracking.log'))

    # fish_tracker = FishTracker(OpenCLAcceleration(), display_scale=0.5)
    # fish_tracker.process_video('data/IMG_4219.mov', 'output', 1, True)



