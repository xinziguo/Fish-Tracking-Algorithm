import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.colors import ListedColormap
import pandas as pd

# 读取数据
data = pd.read_csv('./output/fish_data.csv')
print(data.shape)

frame_shape = (544, 960)

# 数据清洗，去除无效值
data = data.replace([None, np.nan, np.inf, -np.inf], np.nan).dropna(subset=['Head_X', 'Head_Y', 'Time'])

# 提取有效的x、y坐标和时间戳
x = np.array(data['Head_X'].values)
y = np.array(data['Head_Y'].values)
time = pd.to_timedelta(data['Time'].values)

# 计算每个点的停留时间
time_diff = pd.Series(time).diff().fillna(pd.Timedelta(seconds=0)).dt.total_seconds().values

# 创建2D直方图，使用停留时间作为权重
heatmap, xedges, yedges = np.histogram2d(x, y, bins=[frame_shape[1], frame_shape[0]], range=[[0, frame_shape[1]], [0, frame_shape[0]]], weights=time_diff)

# 应用高斯模糊平滑数据
heatmap = gaussian_filter(heatmap, sigma=2)

# 创建颜色映射
colors = ['gray'] + plt.cm.jet(np.linspace(0, 1, 256)).tolist()
cmap = ListedColormap(colors)
norm = plt.Normalize(vmin=0, vmax=np.max(heatmap))

# 创建图像
plt.figure(figsize=(10, 6))
plt.imshow(heatmap, cmap=cmap, norm=norm, extent=[0, frame_shape[0], 0, frame_shape[1]], interpolation='bilinear')
plt.colorbar(label='Density')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Heatmap of Fish Data')
plt.savefig('./output/fish_trajectory_heatmap.png')
plt.show()