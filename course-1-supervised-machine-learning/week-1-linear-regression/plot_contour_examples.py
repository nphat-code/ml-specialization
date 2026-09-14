import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

# Training data (synthetic housing style): (1, 3), (2, 5), (3, 7) -> True line: y = 2x + 1
x_data = np.array([1.0, 2.0, 3.0])
y_data = np.array([3.0, 5.0, 7.0])
m = len(x_data)

# Grid for Contour
w_vals = np.linspace(0.0, 3.8, 120)
b_vals = np.linspace(-1.0, 6.0, 120)
W, B = np.meshgrid(w_vals, b_vals)

J = np.zeros_like(W)
for i in range(len(b_vals)):
    for j in range(len(w_vals)):
        w = W[i, j]
        b = B[i, j]
        J[i, j] = np.sum((w * x_data + b - y_data)**2) / (2 * m)

# 3 example points (w, b)
# Point 1: Bad fit (outer contour)
p1 = {'w': 0.5, 'b': 5.0, 'color': '#ef4444', 'name': '1. Khớp kém (Vòng ngoài)'}
# Point 2: Medium fit (closer)
p2 = {'w': 1.2, 'b': 2.8, 'color': '#f59e0b', 'name': '2. Khớp trung bình'}
# Point 3: Best fit (center minimum)
p3 = {'w': 2.0, 'b': 1.0, 'color': '#10b981', 'name': '3. Khớp tối ưu (Tâm elip - Cực tiểu)'}
points = [p1, p2, p3]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=150)

# ----------------- LEFT PLOT: Data and Lines -----------------
ax1.scatter(x_data, y_data, color='blue', marker='x', s=130, linewidth=3, label='Dữ liệu thực tế (x, y)', zorder=6)
x_line = np.linspace(0.5, 3.5, 100)

for p in points:
    cost = np.sum((p['w'] * x_data + p['b'] - y_data)**2) / (2 * m)
    label = f"{p['name']}: f(x)={p['w']}x + {p['b']} (J={cost:.2f})"
    ax1.plot(x_line, p['w'] * x_line + p['b'], color=p['color'], linewidth=2.5, label=label, zorder=5)

ax1.set_title('1. Đồ thị Dữ liệu & Các đường thẳng f(x) = wx + b', fontsize=13, fontweight='bold', pad=12)
ax1.set_xlabel('x (Diện tích nhà)', fontsize=11, labelpad=8)
ax1.set_ylabel('y (Giá nhà)', fontsize=11, labelpad=8)
ax1.set_xlim(0.3, 3.7)
ax1.set_ylim(1.0, 9.0)
ax1.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.95, fontsize=9.5)
ax1.grid(True, linestyle='--', alpha=0.5)

# ----------------- RIGHT PLOT: Contour Map -----------------
levels = np.array([0.05, 0.3, 0.8, 2.0, 4.5, 8.0, 14.0])
contours = ax2.contour(W, B, J, levels=levels, cmap='viridis_r', linewidths=1.8)
ax2.clabel(contours, inline=True, fontsize=9, fmt='J=%.2f')

for p in points:
    cost = np.sum((p['w'] * x_data + p['b'] - y_data)**2) / (2 * m)
    ax2.scatter(p['w'], p['b'], color=p['color'], s=120, edgecolors='black', linewidth=1.5, zorder=7)
    offset_x = 0.12 if p['w'] < 2.0 else -0.55
    offset_y = 0.25 if p['b'] < 4.0 else -0.5
    ax2.annotate(f"{p['name'][:1]}\n(w={p['w']}, b={p['b']})\nJ={cost:.2f}",
                 xy=(p['w'], p['b']), xytext=(p['w'] + offset_x, p['b'] + offset_y),
                 fontsize=9.5, fontweight='bold', color=p['color'],
                 arrowprops=dict(arrowstyle='->', color=p['color'], lw=1.5))

ax2.set_title('2. Vị trí tương ứng trên Bản đồ Contour Plot của J(w, b)', fontsize=13, fontweight='bold', pad=12)
ax2.set_xlabel('Tham số w (Độ dốc)', fontsize=11, labelpad=8)
ax2.set_ylabel('Tham số b (Điểm cắt trục tung)', fontsize=11, labelpad=8)
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()

# Save image
out_repo_path = os.path.join(os.getcwd(), 'course-1-supervised-machine-learning', 'week-1-linear-regression', 'contour_examples.png')
plt.savefig(out_repo_path, bbox_inches='tight')
print(f"Saved image to: {out_repo_path}")

artifact_dir = r"C:\Users\MY MSI\.gemini\antigravity-ide\brain\e29858b5-935f-4a1f-b451-1cc2e8c5608d"
if os.path.exists(artifact_dir):
    dest = os.path.join(artifact_dir, 'contour_examples.png')
    shutil.copyfile(out_repo_path, dest)
    print(f"Copied to artifact dir: {dest}")
