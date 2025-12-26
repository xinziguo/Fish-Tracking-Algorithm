#use matplotlib to show image
import matplotlib.pyplot as plt
import cv2

def show_image(image):
    plt.imshow(image)
    plt.show()

image_path = 'data/test/frame_0000.jpg'
image = cv2.imread(image_path)
show_image(image)
