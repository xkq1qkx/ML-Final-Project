import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import json

# 定义B矩阵的列表
with open('./B_n2_best.json', 'r') as f:
    B_matrices_json = json.load(f)
    B_matrices = [np.array(matrix) for matrix in B_matrices_json]

# 定义第二个矩阵C
C = np.array([
    [1, 0],
    [0.5, np.sqrt(3)/2]
])

# 定义颜色
colors_B = ['b']
colors_C = ['orange']
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制矩阵C的向量
for vector in C:
    ax.quiver(0, 0, 0, vector[0], vector[1], 0, color=colors_C, length=1.5, linestyle='dashed', normalize=True, arrow_length_ratio=0.1)

# 设置坐标轴范围
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 定义更新函数
def update_vectors(num, B_matrices, quivers1):
    if num >= len(B_matrices):
        return
    B = B_matrices[num]
    for i in range(len(B)):
        quivers1[i].remove()
        quivers1[i] = ax.quiver(0, 0, 0, B[i, 0], B[i, 1], 0, color=colors_B, length=1.0, normalize=True, arrow_length_ratio=0.1)

# 初始化quivers
if B_matrices and len(B_matrices[0]) > 0:
    quivers1 = [ax.quiver(0, 0, 0, B_matrices[0][i, 0], B_matrices[0][i, 1], 0, color=colors_B, linestyle='dashed', length=1.0, normalize=True, arrow_length_ratio=0.1) for i in range(len(B_matrices[0]))]
else:
    quivers1 = []

# 创建动画
ani = animation.FuncAnimation(fig, update_vectors, frames=len(B_matrices), fargs=(B_matrices, quivers1), interval=100, blit=False, repeat=False)
#ani = animation.FuncAnimation(fig, update_vectors, frames=len(B_matrices), fargs=(B_matrices, quivers1), interval=100, blit=False, repeat=False)
# 停止动画
def stop_animation(event):
    ani.event_source.stop()

# 在5秒后停止动画
timer = fig.canvas.new_timer(interval=5000)
timer.add_callback(stop_animation, None)
timer.start()
# 保存动画
#ani.save('./vector_animation.gif', writer='pillow')
plt.show()