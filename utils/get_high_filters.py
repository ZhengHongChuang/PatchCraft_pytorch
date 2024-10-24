import numpy as np
import scipy.ndimage

# 定义初始的高通滤波器核
kernels = {
    'a': np.array([[0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0],
                   [0, 0, -1, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]),
    
    'b': np.array([[0, 0, -1, 0, 0],
                   [0, 0, 3, 0, 0],
                   [0, 0, -3, 0, 0],
                   [0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0]]),
    
    'c': np.array([[0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0],
                   [0, 0, -2, 0, 0],
                   [0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0]]),
    
    'd': np.array([[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]),
    
    'e': np.array([[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]),
    
    'f': np.array([[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]),
    
    'g': np.array([[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]])
}


# 定义旋转函数
def generate_rotations(kernel, angles):
    rotated_kernels = [scipy.ndimage.rotate(kernel, angle, reshape=False) for angle in angles]
    return rotated_kernels

# 旋转角度
angles_8_directions = [45, 90, 135, 180, 225, 270, 315, 360]  # 对应 ↗, →, ↘, ↓, ↙, ←, ↖, ↑
angles_4_directions = [270, 0, 225, 315]  # 对应 →, ↓, ↗, ↘
angles_4_directions_main = [90, 180, 270, 360]  # 对应 →, ↓, ←, ↑

# 创建字典用于保存不同类别的滤波器
filter_a = {}
filter_b = {}
filter_c = {}
filter_d = {}
filter_e = {}
filter_f = {}

# 对于8方向旋转的滤波器 a 和 b
filter_a = {f'rotation_{i+1}': rot for i, rot in enumerate(generate_rotations(kernels['a'], angles_8_directions))}
filter_b = {f'rotation_{i+1}': rot for i, rot in enumerate(generate_rotations(kernels['b'], angles_8_directions))}

# 对于4方向旋转的滤波器 c
filter_c = {f'rotation_{i+1}': rot for i, rot in enumerate(generate_rotations(kernels['c'], angles_4_directions))}

# 对于4方向旋转的滤波器 d 和 e
filter_d = {f'rotation_{i+1}': rot for i, rot in enumerate(generate_rotations(kernels['d'], angles_4_directions_main))}
filter_e = {f'rotation_{i+1}': rot for i, rot in enumerate(generate_rotations(kernels['e'], angles_4_directions_main))}

# 输出各类滤波器数量
# print(filter_a)
# print(filter_b)
print(filter_c)
# print(filter_d)
# print(filter_e)

