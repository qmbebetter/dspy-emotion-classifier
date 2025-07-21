import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 设置字体为支持中文的字体
font_path = 'C:/Windows/Fonts/simsun.ttc'  # 宋体字体
font_prop = font_manager.FontProperties(fname=font_path)

# 更新matplotlib配置
plt.rcParams['font.family'] = font_prop.get_name()

# 更新后的数据范围
originA = np.array([0.1, 0.1, 0.1])
axesA = np.eye(3)  # 设备A的坐标轴方向
scaleA = 0.1

originB = np.array([0.2, 0.2, 0.2])
axesB = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])  # 设备B的坐标轴方向
scaleB = 0.2

def transform_coordinates(coords, origin, axes, scale):
    """
    将坐标转换到规范化坐标系。
    """
    coords_transformed = (coords - origin) / scale
    coords_transformed = np.dot(coords_transformed, np.linalg.inv(axes))  # 使用逆矩阵进行坐标转换
    return coords_transformed

def calculate_statistics(coords):
    """
    计算坐标的均值和方差。
    """
    mean = np.mean(coords, axis=0)
    variance = np.var(coords, axis=0)
    return mean, variance

def calculate_distance(coords1, coords2):
    """
    计算两个坐标点之间的欧氏距离。
    """
    return np.linalg.norm(coords1 - coords2, axis=1)

def calculate_rmse(coords1, coords2):
    """
    计算两个坐标点之间的均方根误差（RMSE）。
    """
    return np.sqrt(np.mean((coords1 - coords2) ** 2, axis=1))

# 更新后的示例坐标点
coords = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3]])

# 计算未规范化坐标的均值和方差
mean_unscaled, var_unscaled = calculate_statistics(coords)
print('未规范化坐标的均值:', mean_unscaled)
print('未规范化坐标的方差:', var_unscaled)

# 转换坐标到规范化坐标系
norm_coords_A = transform_coordinates(coords, originA, axesA, scaleA)
norm_coords_B = transform_coordinates(coords, originB, axesB, scaleB)

# 计算规范化坐标A的均值和方差
mean_scaled_A, var_scaled_A = calculate_statistics(norm_coords_A)
print('规范化坐标A的均值:', mean_scaled_A)
print('规范化坐标A的方差:', var_scaled_A)

# 计算规范化坐标B的均值和方差
mean_scaled_B, var_scaled_B = calculate_statistics(norm_coords_B)
print('规范化坐标B的均值:', mean_scaled_B)
print('规范化坐标B的方差:', var_scaled_B)

# 计算未规范化坐标与规范化坐标A、B之间的RMSE
rmse_unscaled_to_A = calculate_rmse(coords, norm_coords_A)
rmse_unscaled_to_B = calculate_rmse(coords, norm_coords_B)

print('未规范化坐标到规范化坐标A的RMSE:')
print(rmse_unscaled_to_A)
print('未规范化坐标到规范化坐标B的RMSE:')
print(rmse_unscaled_to_B)

# 计算平均RMSE
mean_rmse_to_A = np.mean(rmse_unscaled_to_A)
mean_rmse_to_B = np.mean(rmse_unscaled_to_B)

print('未规范化坐标到规范化坐标A的平均RMSE:', mean_rmse_to_A)
print('未规范化坐标到规范化坐标B的平均RMSE:', mean_rmse_to_B)

# 可视化结果
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制未规范化坐标
ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], color='blue', label='未规范化坐标')

# 绘制规范化坐标
ax.scatter(norm_coords_A[:, 0], norm_coords_A[:, 1], norm_coords_A[:, 2], color='red', label='规范化坐标A')
ax.scatter(norm_coords_B[:, 0], norm_coords_B[:, 1], norm_coords_B[:, 2], color='green', label='规范化坐标B')

ax.set_xlabel('X轴', fontproperties=font_prop)
ax.set_ylabel('Y轴', fontproperties=font_prop)
ax.set_zlabel('Z轴', fontproperties=font_prop)
ax.set_title('坐标对比', fontproperties=font_prop)
ax.legend()

plt.show()
