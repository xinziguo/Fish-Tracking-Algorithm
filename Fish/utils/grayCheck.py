#transfer the image to gray image
import cv2
import numpy as np

def transfer_to_gray(image_path,save_path):
    # Read the image from the specified path
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Save the grayscale image
    cv2.imwrite(save_path, gray_image)

    print(f"Grayscale image saved to {save_path}")

if __name__ == "__main__":
    import os
    files = os.listdir("data/test")
    for file in files:
        image_path = os.path.join("data/test", file)
        save_path = os.path.join("data/gray", file)
        transfer_to_gray(image_path,save_path)






