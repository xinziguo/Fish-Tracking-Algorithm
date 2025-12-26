import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import timedelta
from matplotlib.colors import LinearSegmentedColormap
from simple_log_helper import CustomLogger
import os
import glob
from abc import ABC, abstractmethod
import multiprocessing
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

    def transfer_rpg_to_hsv(self, rpg_image):
        return cv2.cvtColor(rpg_image, cv2.COLOR_RGB2HSV)

    def create_mask(self, hsv_image):
        lower_bound = np.array([85, 20, 0])
        upper_bound = np.array([120, 255, 170])
        return cv2.inRange(hsv_image, lower_bound, upper_bound)

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

    def correct_fish_orientation(self, prev_heads, prev_tails, current_head, current_tail, rect, time):
        if len(prev_heads) < 2 or len(prev_tails) < 2:
            # 如果没有足够的历史数据，就使用当前的检测结果
            return current_head, current_tail

        prev_head = prev_heads[-1]
        prev_tail = prev_tails[-1]
        # 检查当前鱼的位置是否与上一帧的鱼重叠
        current_rect = cv2.minAreaRect(np.array([current_head, current_tail]))
        prev_rect = cv2.minAreaRect(np.array([prev_head, prev_tail]))
        
        if not self.rectangles_overlap(current_rect, prev_rect):
            self.logger.info(f"鱼的位置在帧 {time} 发生了跳变，视为新的鱼")
            return current_head, current_tail
        
        # 计算前一帧的运动向量
        prev_direction = np.array(prev_head) - np.array(prev_tail)
        
        # 计算当前帧的两个可能的运动向量
        current_direction1 = np.array(current_head) - np.array(current_tail)
        current_direction2 = np.array(current_tail) - np.array(current_head)
        
        # 计算与前一帧运动向量的夹角
        angle1 = self.angle_between(prev_direction, current_direction1)
        angle2 = self.angle_between(prev_direction, current_direction2)
        
        # 如果第二个方向与前一帧的方向更接近（夹角更小），则交换头尾
        if angle2 < angle1:
            self.logger.info(f"交换头尾 {time}, 基于运动方向")
            return current_tail, current_head
        
        return current_head, current_tail

    def angle_between(self, v1, v2):
        # 计算两个向量之间的夹角（以度为单位）
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) * 180 / np.pi

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
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        fish_contour = max(contours, key=cv2.contourArea)
        
        rect = cv2.minAreaRect(fish_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        moments = cv2.moments(fish_contour)
        centroid_x = int(moments['m10'] / moments['m00'])
        centroid_y = int(moments['m01'] / moments['m00'])
        centroid = (centroid_x, centroid_y)

        distances = np.sqrt(((fish_contour - centroid) ** 2).sum(axis=2))
        farthest_point = tuple(fish_contour[distances.argmax()][0])

        distances_from_tail = np.sqrt(((fish_contour - farthest_point) ** 2).sum(axis=2))
        head = tuple(fish_contour[distances_from_tail.argmax()][0])
        tail = tuple(fish_contour[distances_from_tail.argmin()][0])

        return box, head, tail, rect

    def process_video(self, video_path, output_dir, frame_interval=1, show_video=False, logger=None):
        if logger:
            self.logger = logger
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.info(f"Error opening video file: {video_path}")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Processing video: {video_path}")
        logger.info(f"Video size: {width}x{height}")
        logger.info(f"Total frames: {frame_count}")
        logger.info(f"FPS: {fps}")

        output_video = os.path.join(output_dir, os.path.basename(video_path).replace('.mov', '_output.mp4'))
        output_csv = os.path.join(output_dir, os.path.basename(video_path).replace('.mov', '_data.csv'))
        output_heatmap = os.path.join(output_dir, os.path.basename(video_path).replace('.mov', '_heatmap.png'))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        
        head_positions = []
        tail_positions = []
        fish_data = []
        detected_fish_count = 0
        processed_frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame_count += 1
            if processed_frame_count % frame_interval != 0:
                continue

            frame = self.acceleration_strategy.process_frame(frame)
            hsv_image = self.transfer_rpg_to_hsv(frame)
            mask = self.create_mask(hsv_image)
            roi_mask = self.define_roi(frame)
            mask = cv2.bitwise_and(mask, roi_mask)
            
            result = self.find_fish_features(mask.get() if isinstance(mask, cv2.UMat) else mask)

            if result is not None:
                box, head, tail, rect = result
                time = timedelta(seconds=processed_frame_count/fps)
                corrected_head, corrected_tail = self.correct_fish_orientation(head_positions, tail_positions, head, tail, rect, time)
                
                detected_fish_count += 1
                head_positions.append(corrected_head)
                tail_positions.append(corrected_tail)
                
                fish_data.append([time, corrected_head[0], corrected_head[1], rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]])
                
                cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
                cv2.circle(frame, corrected_head, 5, (0, 0, 255), -1)
                cv2.circle(frame, corrected_tail, 5, (255, 0, 0), -1)
                
                recent_positions = head_positions[-100:]
                for i in range(1, len(recent_positions)):
                    cv2.line(frame, recent_positions[i-1], recent_positions[i], (0, 255, 255), 1)
                cv2.putText(frame, str(time), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                out.write(frame.get() if isinstance(frame, cv2.UMat) else frame)
                if show_video:
                    display_frame = cv2.resize(frame.get() if isinstance(frame, cv2.UMat) else frame, 
                                               (int(width * self.display_scale), int(height * self.display_scale)))
                    cv2.imshow('Fish Tracking', display_frame)

            if processed_frame_count % 100 == 0:
                logger.info(f"Processed {processed_frame_count}/{frame_count} frames")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        logger.info(f"Detected fish in {detected_fish_count} frames")
        logger.info(f"Total processed frames: {processed_frame_count}")
        logger.info(f"Output video saved to: {output_video}")

        if head_positions:
            self.save_to_csv(fish_data, output_csv)
            self.plot_trajectory_heatmap(head_positions, (width, height), output_heatmap)
            logger.info(f"Processing completed successfully for {video_path}")
            logger.info(f"CSV data saved to: {output_csv}")
            logger.info(f"Heatmap saved to: {output_heatmap}")
        else:
            logger.warning(f"No fish detected in the video: {video_path}")

    def save_to_csv(self, fish_data, output_file):
        with open(output_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Time", "Head_X", "Head_Y", "Rect_Center_X", "Rect_Center_Y", "Rect_Width", "Rect_Height", "Rect_Angle"])
            writer.writerows(fish_data)

    def plot_trajectory_heatmap(self, head_positions, frame_shape, output_file='fish_trajectory_heatmap.png'):
        # 创建图像
        plt.figure(figsize=(10, 10))
        
        # 绘制轨迹
        x = [pos[0] for pos in head_positions]
        y = [pos[1] for pos in head_positions]
        plt.plot(x, y, color='white', linewidth=0.5, alpha=0.7)
        
        # 创建热力图数据
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=50, range=[[0, frame_shape[1]], [0, frame_shape[0]]])
        
        # 创建自定义颜色映射
        colors = ['darkblue', 'blue', 'lightblue', 'green', 'yellow', 'red']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
        
        # 绘制热力图
        plt.imshow(heatmap.T, extent=[0, frame_shape[1], frame_shape[0], 0], 
                   cmap=cmap, alpha=0.7, interpolation='gaussian')
        
        # 设置坐标轴
        plt.xlim(0, frame_shape[1])
        plt.ylim(frame_shape[0], 0)  # 反转Y轴以匹配图像坐标系
        plt.axis('off')  # 隐藏坐标轴
        
        # 添加 'S' 和 'F' 标签
        plt.text(0.05, 0.5, 'S', fontsize=20, color='white', transform=plt.gca().transAxes)
        plt.text(0.95, 0.5, 'F', fontsize=20, color='white', transform=plt.gca().transAxes)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='gray')
        plt.close()

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
    max_workers = multiprocessing.cpu_count()
    # process_videos('data/hori', 'output/hori', 1, False, max_workers, False)
    process_videos('data/fix', 'output/fix', 1, False, max_workers, False)
    # fish_tracker = FishTracker(OpenCLAcceleration(), display_scale=0.5)
    # fish_tracker.process_video('data/IMG_4219.mov', 'output', 1, True)