import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import timedelta
from matplotlib.colors import LinearSegmentedColormap
from simple_log_helper import CustomLogger
import os
import argparse
logger = CustomLogger(__name__,log_filename='Logs/fish_tracking.log')

def transfer_rpg_to_hsv(rpg_image):
    return cv2.cvtColor(rpg_image, cv2.COLOR_RGB2HSV)

def create_mask(hsv_image):
    lower_bound = np.array([90, 50, 50])
    upper_bound = np.array([100, 255, 170])
    return cv2.inRange(hsv_image, lower_bound, upper_bound)

def define_roi(image):
    symmetry_x, symmetry_y = 92, 175
    roi_mask = np.zeros(image.shape[:2], np.uint8)
    roi_mask[symmetry_y:, symmetry_x:] = 255
    return roi_mask

def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def correct_fish_orientation(prev_tails, current_head, current_tail, rect, time):
    if len(prev_tails) < 2:
        return current_head, current_tail

    # 计算矩形的中心和半径
    center_x, center_y = rect[0]
    width, height = rect[1]
    radius = np.sqrt(width**2 + height**2) / 2

    # 计算前两帧的平均尾部位置
    avg_prev_tail = np.mean(prev_tails[-2:], axis=0)

    # 检查当前头部是否在预期范围内
    if distance(avg_prev_tail, current_head) <= radius:
        # 如果尾部更接近预期的头部位置，交换头尾
        logger.info(f"交换头尾{time}")
        return current_tail, current_head
    else:
        return current_head, current_tail
    
def find_fish_features(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None, None
    
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

def process_video(video_path, output_video_path,frame_interval=1,show_video=False):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logger.info("Error opening video file")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Video size: {width}x{height}")
    logger.info(f"Total frames: {frame_count}")
    logger.info(f"FPS: {fps}")

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
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

        hsv_image = transfer_rpg_to_hsv(frame)
        mask = create_mask(hsv_image)
        roi_mask = define_roi(frame)
        mask = cv2.bitwise_and(mask, roi_mask)
        
        result = find_fish_features(mask)

        if result is not None:
            box, head, tail, rect = result
            time = timedelta(seconds=processed_frame_count/fps)
            # 使用新的函数来修正鱼的方向
            corrected_head, corrected_tail = correct_fish_orientation(tail_positions, head, tail, rect, time)
            
            detected_fish_count += 1
            head_positions.append(corrected_head)
            tail_positions.append(corrected_tail)
            
            
            fish_data.append([time, corrected_head[0], corrected_head[1], rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]])
            
            cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
            cv2.circle(frame, corrected_head, 5, (0, 0, 255), -1)
            cv2.circle(frame, corrected_tail, 5, (255, 0, 0), -1)
            
            # 只绘制最近的100个点，以提高性能
            recent_positions = head_positions[-100:]
            for i in range(1, len(recent_positions)):
                cv2.line(frame, recent_positions[i-1], recent_positions[i], (0, 255, 255), 1)
            cv2.putText(frame, str(time), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            out.write(frame)
            if show_video:
                cv2.imshow('Fish Tracking', frame)

        if processed_frame_count % 100 == 0:
            logger.info(f"Processed {processed_frame_count}/{frame_count} frames")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    logger.info(f"Detected fish in {detected_fish_count} frames")
    logger.info(f"Total processed frames: {processed_frame_count}")
    logger.info(f"Output video saved to: {output_video_path}")

    return head_positions, fish_data, (width, height)

def save_to_csv(fish_data, output_file):
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Time", "Head_X", "Head_Y", "Rect_Center_X", "Rect_Center_Y", "Rect_Width", "Rect_Height", "Rect_Angle"])
        writer.writerows(fish_data)

def plot_trajectory(head_positions, frame_shape):
    x = [pos[0] for pos in head_positions]
    y = [pos[1] for pos in head_positions]
    
    plt.figure(figsize=(10, 10))
    plt.plot(x, y)
    plt.title('Fish Head Trajectory')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.xlim(0, frame_shape[0])
    plt.ylim(frame_shape[1], 0)  # Invert Y axis to match image coordinates
    plt.savefig('fish_trajectory.png')
    plt.close()

def plot_heatmap(head_positions, frame_shape):
    x = [pos[0] for pos in head_positions]
    y = [pos[1] for pos in head_positions]
    
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=50, range=[[0, frame_shape[0]], [0, frame_shape[1]]])
    
    plt.figure(figsize=(10, 10))
    plt.imshow(heatmap.T, origin='lower', extent=[0, frame_shape[0], 0, frame_shape[1]], cmap='hot')
    plt.colorbar(label='Frequency')
    plt.title('Fish Movement Heatmap')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.gca().invert_yaxis()  # Invert Y axis to match image coordinates
    plt.savefig('fish_heatmap.png')
    plt.close()

def plot_trajectory_heatmap(head_positions, frame_shape, output_file='fish_trajectory_heatmap.png'):
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

def main(input_video, output_dir, frame_interval,show_video=False):
    os.makedirs(output_dir, exist_ok=True)
    output_video = os.path.join(output_dir, 'fish_tracking_output.mp4')
    output_csv = os.path.join(output_dir, 'fish_data.csv')
    output_heatmap = os.path.join(output_dir, 'fish_trajectory_heatmap.png')

    head_positions, fish_data, frame_shape = process_video(input_video, output_video, frame_interval,show_video)
    
    if head_positions:
        save_to_csv(fish_data, output_csv)
        plot_trajectory_heatmap(head_positions, frame_shape, output_heatmap)
        logger.info("Processing completed successfully")
    else:
        logger.warning("No fish detected in the video")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fish Tracking Tool")
    parser.add_argument("--input_video", default='raw_data.mp4', help="Path to the input video file")
    parser.add_argument("--output_dir", default='output', help="Directory to save output files")
    parser.add_argument("--frame_interval", type=int, default=1, help="Interval of frames to process (default: 1)")
    parser.add_argument("--show_video",type=bool, default=False, help="Show video frames")
    args = parser.parse_args()

    main(args.input_video, args.output_dir, args.frame_interval,args.show_video)