#resize video as 1/4
import cv2

def resize_video(input_path, output_path, scale=4):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / scale)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / scale)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(cv2.resize(frame, (width, height)))

    cap.release()
    out.release()

if __name__ == "__main__":
    import os
    folder_path = "./data/verti"
    output_folder = "./output/verti"
    for file in os.listdir(folder_path):
        if file.endswith(".mov"):
            input_path = os.path.join(folder_path, file)
            output_path = os.path.join(output_folder, f"resize_{file}")
            resize_video(input_path, output_path)
            print(f"Resized {file} to {output_path}")

