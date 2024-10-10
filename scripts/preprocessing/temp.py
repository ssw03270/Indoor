import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation

# 예제 그리드 맵 (0: 자유 공간, 1: 장애물)
grid_map = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0]
])

# 사람의 두께(그리드 셀 단위)
thickness = 1  # 반지름

# 원형 구조 요소 생성
structuring_element = np.array([
    [0,1,0],
    [1,1,1],
    [0,1,0]
])

# 맵 팽창
inflated_map = binary_dilation(grid_map, structure=structuring_element, iterations=thickness).astype(int)

# 시각화
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(grid_map, cmap='Greys')
plt.title('원본 맵')
plt.subplot(1,2,2)
plt.imshow(inflated_map, cmap='Greys')
plt.title('팽창된 맵 (두께 고려)')
plt.show()