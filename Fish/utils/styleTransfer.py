import cv2
import numpy as np
import matplotlib.pyplot as plt

def adjust_tone(image, alpha=1.1, beta=20):
    """
    调整图像的色调，使其更加暖或冷。
    alpha: 对比度控制（>1 增加对比度）
    beta: 亮度控制（可以正负）
    """
    # 调整对比度和亮度
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

def shift_hue(image, hue_shift):
    """
    改变图像的色相，使鱼体的颜色更突出。
    hue_shift: 色相偏移量（范围为-180到180）
    """
    # Convert UMat to numpy array if necessary
    if isinstance(image, cv2.UMat):
        image = image.get()
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180  # 调整色相通道
    shifted_image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return shifted_image

def style_transfer(image, alpha=1.1, beta=30, hue_shift=10):
    # Convert UMat to numpy array if necessary
    if isinstance(image, cv2.UMat):
        image = image.get()
    
    adjusted_image = adjust_tone(image, alpha, beta)
    hue_shifted_image = shift_hue(adjusted_image, hue_shift)
    gray_image = 255 - cv2.cvtColor(hue_shifted_image, cv2.COLOR_BGR2GRAY)
    return gray_image


if __name__ == '__main__':
    # 读取图像
    image1 = cv2.imread('data/test/frame_0000.jpg')
    image2 = cv2.imread('data/test/frame_0001.jpg')
    # 调整两张图片的色调
    adjusted_image1 = adjust_tone(image1, alpha=1.1, beta=30)
    adjusted_image2 = adjust_tone(image2, alpha=1.1, beta=30)

    # 改变色相，让鱼体更突出
    hue_shifted_image1 = shift_hue(adjusted_image1, hue_shift=10)
    hue_shifted_image2 = shift_hue(adjusted_image2, hue_shift=10)

    # 显示色调处理后的图像
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.title('Tone Adjusted Image 1')
    plt.imshow(cv2.cvtColor(adjusted_image1, cv2.COLOR_BGR2RGB))

    plt.subplot(2, 2, 2)
    plt.title('Hue Shifted Image 1')
    plt.imshow(cv2.cvtColor(hue_shifted_image1, cv2.COLOR_BGR2RGB))

    plt.subplot(2, 2, 3)
    plt.title('Tone Adjusted Image 2')
    plt.imshow(cv2.cvtColor(adjusted_image2, cv2.COLOR_BGR2RGB))

    plt.subplot(2, 2, 4)
    plt.title('Hue Shifted Image 2')
    plt.imshow(cv2.cvtColor(hue_shifted_image2, cv2.COLOR_BGR2RGB))

    plt.tight_layout()
    plt.show()

    # transfer the image to gray image( and reverse the pixe)
    gray_image1 = 255 - cv2.cvtColor(hue_shifted_image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = 255 - cv2.cvtColor(hue_shifted_image2, cv2.COLOR_BGR2GRAY)

    # save the gray image
    cv2.imwrite('data/style/frame_0000_gray.jpg', gray_image1)
    cv2.imwrite('data/style/frame_0001_gray.jpg', gray_image2)



