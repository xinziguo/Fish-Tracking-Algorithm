import cv2
import os

def sample_video(input_path, output_dir, sample_rate=1000):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(input_path)

    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_count = 0
    sample_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % sample_rate == 0:
            # Save the frame as an image
            output_path = os.path.join(output_dir, f"frame_{sample_count:04d}.jpg")
            cv2.imwrite(output_path, frame)
            sample_count += 1

        frame_count += 1

    cap.release()
    print(f"Sampled {sample_count} frames from {total_frames} total frames.")

# Usage
input_video = "data/fix/IMG_4252.mp4"
output_directory = "data/test"
sample_video(input_video, output_directory)