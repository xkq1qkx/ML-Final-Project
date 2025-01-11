import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
'''
# 假设 B 矩阵是一个 2D numpy 数组
B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 创建网格
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

# 定义函数 Z = f(X, Y) 这里假设是一个简单的二次函数
Z = B[0, 0] * X**2 + B[0, 1] * X * Y + B[0, 2] * Y**2

# 创建图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制曲面
ax.plot_surface(X, Y, Z, cmap='viridis')

# 添加标签
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
'''
with open('n3_Adam.json', 'r') as f:
    B_matrices_json = json.load(f)
    B = np.array(B_matrices_json[-1])
    B = B / np.linalg.norm(B, axis=1, keepdims=True)
    print("要计算的B:",B)

#B_gt =np.array([[1, 0], [0.5, np.sqrt(3)/2]])
#B_gt =np.array([[1, 0,0], [0.5, np.sqrt(3)/2,0],[0.5, np.sqrt(3)/6, np.sqrt(6)/3]])

batch_size = 10000  # Define a batch size to process vectors in chunks
dr2 = 1e-5
steps = int(5 // dr2)
print("计算的精度:",dr2,steps)
pic = np.zeros(steps)
pic_gt = np.zeros(steps)
n=B.shape[1]

vectors_all = np.array(np.meshgrid(*[np.arange(-5,5)] * B.shape[1])).T.reshape(-1, B.shape[1])


for i in tqdm(range(0, 10**n, batch_size), desc="Processing batches"):
    vectors = vectors_all[i:i + batch_size]
    dots = [vector.dot(B) for vector in vectors]
    #dots_gt = [vector.dot(B_gt) for vector in vectors]
    lengths = [np.dot(dot, dot) for dot in dots]
    #lengths_gt = [np.dot(dot, dot) for dot in dots_gt]

    lengths_sorted = np.sort(lengths)
    #lengths_gt_sorted = np.sort(lengths_gt)
    index = 0
    index_gt = 0
    for j in range(steps):
        while index < len(lengths_sorted) and lengths_sorted[index] <= j * dr2:
            pic[j] += 1
            index += 1
        #while index_gt < len(lengths_gt_sorted) and lengths_gt_sorted[index_gt] <= j * dr2:
        #    pic_gt[j] += 1
        #    index_gt += 1

for i in range(1, steps):
    pic[i] = pic[i] + pic[i - 1]
    pic_gt[i] = pic_gt[i] + pic_gt[i - 1]

plt.plot(np.arange(steps) * dr2, pic)
#plt.plot(np.arange(steps) * dr2, pic_gt, linestyle='--')
plt.xlabel('R^2')
plt.ylabel('Count')
plt.title('Histogram of Lengths')
plt.show()