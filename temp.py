import numpy as np

lb = np.zeros((0, 5), dtype=np.float32)

print(lb.shape)  # 输出: (0, 5)
print(len(lb))   # 输出: 0 (代表没有目标)
print(lb[:, 0:1].shape) # 输出: (0, 1)，仍然可以切片！

print(lb[:, 0:1])
print(lb[:, 1:])