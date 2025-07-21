import numpy as np

scores = [1096, 1396, 1398, 1019, 1054, 1064, 938, 1134, 1096, 1476, 1161, 983, 1031, 858, 661, 914, 1263, 1388, 1142, 948, 909, 1044, 1071, 940, 1776, 1096, 1476, 1161, 983, 1031, 1348, 1033, 996, 921, 826, 1556, 787, 1747, 922, 935, 702, 1096, 1476, 1161, 983, 1031, 1360, 1144, 917, 1407]

# 计算基本统计量
mean_score = np.mean(scores)
std_score = np.std(scores)
min_score = np.min(scores)
max_score = np.max(scores)
median_score = np.median(scores)

print(f"Mean: {mean_score}")
print(f"Standard Deviation: {std_score}")
print(f"Min: {min_score}")
print(f"Max: {max_score}")
print(f"Median: {median_score}")
