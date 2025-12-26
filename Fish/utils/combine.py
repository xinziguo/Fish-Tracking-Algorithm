import cv2
import numpy as np

def combine_masks(mask_paths, output_path):
    """
    合并多个掩码，任何一个掩码中的黑色区域（0）都会在最终掩码中变为黑色
    
    Args:
        mask_paths: 掩码文件路径列表
        output_path: 输出文件路径
    """
    combined_mask = None
    
    for mask_path in mask_paths:
        # 读取掩码
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Error: Cannot load mask image from {mask_path}")
            continue
            
        if combined_mask is None:
            # 第一个掩码作为基础
            combined_mask = mask
        else:
            # 确保掩码尺寸一致
            if mask.shape != combined_mask.shape:
                print(f"Warning: Mask {mask_path} has different dimensions. Resizing...")
                mask = cv2.resize(mask, (combined_mask.shape[1], combined_mask.shape[0]))
            
            # 使用按位与操作合并掩码
            combined_mask = cv2.bitwise_and(combined_mask, mask)
    
    if combined_mask is not None:
        # 保存结果
        cv2.imwrite(output_path, combined_mask)
        print(f"Combined mask saved to {output_path}")
        return True
    else:
        print("Error: No valid masks to combine")
        return False


def combine_backgrounds(background_paths, output_path):
    """
    合并多个背景，取均值
    """
    combined_background = None
    count = 0
    
    for background_path in background_paths:
        background = cv2.imread(background_path)
        if background is None:
            print(f"Error: Cannot load background image from {background_path}")
            continue
            
        # Convert to float32 for calculations
        background = background.astype(np.float32)
        
        if combined_background is None:
            combined_background = background
        else:
            combined_background += background
        count += 1

    if combined_background is not None:
        # Calculate mean and convert back to uint8
        combined_background = (combined_background / count).astype(np.uint8)
        cv2.imwrite(output_path, combined_background)
        print(f"Combined background saved to {output_path}")
        return True
    else:
        print("Error: No valid backgrounds to combine")
        return False
    
def extract_side_mask(image_path, output_path):
    """
    从图片中提取两边的mask
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Cannot load image from {image_path}")
        return None
    #detect if black side thick(height)
    top_side_height = 0
    bottom_side_height = 0

    for i in range(image.shape[0]):
        if np.sum(image[i, :]) < image.shape[1] * 1:
            top_side_height += 1
        else:
            break
    for i in range(image.shape[0]-1, -1, -1):
        if np.sum(image[i, :]) < image.shape[1] * 1:
            bottom_side_height += 1
        else:
            break
    print(f"Top side height: {top_side_height}, Bottom side height: {bottom_side_height}")
    mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255
    mask[:top_side_height+15, :] = 0
    mask[-bottom_side_height-15:, :] = 0
    #save mask
    cv2.imwrite(output_path, mask)
    return mask
    

# 使用示例
if __name__ == "__main__":

    mask_paths = [
        './output/tank_mask.jpg',
        './output/tank_mask1.jpg',
        './output/tank_mask2.jpg',
        './output/side_mask_0.jpg',
    ]
    
    combine_masks(
        mask_paths=mask_paths,
        output_path='./output/tank_mask_combined1.jpg'
    )
    # background_paths = [
    #     './output/fix/background_IMG_4245.jpg',
    #     './output/fix/background_IMG_4249.jpg'
    # ]
    # # combine_backgrounds(background_paths, './output/background_model_combined.jpg')
    # extract_side_mask(background_paths[0], './output/side_mask_0.jpg')
