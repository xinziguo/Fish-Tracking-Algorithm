import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap

# def add_line(empty_frame, x0, y0, x1, y1, value=1):
#     # Initialize the DP table
#     dx = abs(x1 - x0)
#     dy = abs(y1 - y0)
#     dp = [[float('inf')] * (dx + 1) for _ in range(dy + 1)]
#     dp[0][0] = 0

#     # Fill the DP table
#     for i in range(dy + 1):
#         for j in range(dx + 1):
#             if i > 0:
#                 dp[i][j] = min(dp[i][j], dp[i-1][j] + 1)
#             if j > 0:
#                 dp[i][j] = min(dp[i][j], dp[i][j-1] + 1)

#     # Trace back the path
#     i, j = dy, dx
#     while i > 0 or j > 0:
#         empty_frame[y0 + i * (1 if y1 > y0 else -1), x0 + j * (1 if x1 > x0 else -1)] += value
#         if i > 0 and dp[i][j] == dp[i-1][j] + 1:
#             i -= 1
#         else:
#             j -= 1
#     empty_frame[y0, x0] += value
#     return empty_frame

def add_line(empty_frame, x0, y0, x1, y1, value=1):
    # Bresenham's line algorithm
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        empty_frame[y0, x0] += value
        if x0 == x1 and y0 == y1:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return empty_frame

def plot_tracker(points,frame_shape):
    # Create an empty frame with uint16 to prevent overflow
    empty_frame = np.zeros((frame_shape[1], frame_shape[0]), dtype=np.float32)
    
    for i in range(len(points)-1):
        empty_frame = add_line(empty_frame, points[i][0], points[i][1], points[i+1][0], points[i+1][1], value=1)
        empty_frame = add_line(empty_frame, points[i][0]-1, points[i][1], points[i+1][0]-1, points[i+1][1], value=1)
        empty_frame = add_line(empty_frame, points[i][0]+1, points[i][1], points[i+1][0]+1, points[i+1][1], value=1)
    
    # Normalize the empty_frame to 0-255(max,min)
    max = np.max(empty_frame)
    min = np.min(empty_frame)

    empty_frame = (empty_frame - min) / (max - min) * 255
    #向上取整
    empty_frame = np.ceil(empty_frame)
    empty_frame = empty_frame.astype(np.uint8)
    print(empty_frame.shape)
    print(np.max(empty_frame))
    print(np.min(empty_frame))
    colors = ['gray'] + plt.cm.jet(np.linspace(0, 1, 256)).tolist()
    n_bins = 257  # 256 for non-zero values + 1 for zero
    cmap = ListedColormap(colors)
    norm = plt.Normalize(vmin=0, vmax=255)

    #use plt to show the empty_frame in hot colormap
    plt.imshow(empty_frame, cmap=cmap, norm=norm)
    plt.colorbar()
    plt.show()



if __name__ == "__main__":
    import pandas as pd
    data = pd.read_csv('./output/fish_data.csv')
    print(data.shape)
    frame_shape = (544,960)
    #check x,y if valid, include none,nan,inf
    # for i in range(len(data)):
    #     if data['Head_X'][i] == None or data['Head_X'][i] == np.nan or data['Head_X'][i] == np.inf:
    #         print(data['Time'][i])
    #     if data['Head_Y'][i] == None or data['Head_Y'][i] == np.nan or data['Head_Y'][i] == np.inf:
    #         print(data['Time'][i])
    x = np.array(data['Head_X'].values)
    y = np.array(data['Head_Y'].values)

    postion = np.array([x,y])
    postion = postion.T
    print(postion.shape)
    plot_tracker(postion, frame_shape)