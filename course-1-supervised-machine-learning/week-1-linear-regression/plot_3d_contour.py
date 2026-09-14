import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import shutil

# Data points (Portland housing style sample):
# x: size (in 1000 sq ft), y: price (in $100k)
# E.g.: (1.0, 3.0), (2.0, 5.0), (3.0, 7.0) -> perfect line y = 2x + 1 (w=2, b=1)
x_data = np.array([1.0, 2.0, 3.0])
y_data = np.array([3.0, 5.0, 7.0])
m = len(x_data)

# Grid of w and b values
w_vals = np.linspace(0.5, 3.5, 100)
b_vals = np.linspace(-1.0, 3.0, 100)
W, B = np.meshgrid(w_vals, b_vals)

# Compute J(w, b) = 1/(2m) * sum((w*x + b - y)^2)
J = np.zeros_like(W)
for i in range(len(b_vals)):
    for j in range(len(w_vals)):
        w = W[i, j]
        b = B[i, j]
        J[i, j] = np.sum((w * x_data + b - y_data)**2) / (2 * m)

# Create side-by-side figure
fig = plt.figure(figsize=(15, 6), dpi=150)

# ----------------- 1. 3D SURFACE PLOT -----------------
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf = ax1.plot_surface(W, B, J, cmap='viridis', alpha=0.85, edgecolor='none')
ax1.scatter([2.0], [1.0], [0.0], color='red', s=120, zorder=10, label='Đáy bát: Cực tiểu (w=2, b=1, J=0)')
ax1.set_title('1. Đồ thị 3D Mặt cong: J(w, b)\n(Hình chiếc bát ngửa / Soup bowl)', fontsize=13, fontweight='bold', pad=12)
ax1.set_xlabel('Tham số w (Độ dốc)', fontsize=11, labelpad=8)
ax1.set_ylabel('Tham số b (Điểm cắt trục tung)', fontsize=11, labelpad=8)
ax1.set_zlabel('Hàm chi phí J(w, b)', fontsize=11, labelpad=8)
ax1.view_init(elev=28, azim=-55)
ax1.legend(loc='upper left', fontsize=10)

# ----------------- 2. CONTOUR PLOT (ĐƯỜNG ĐỒNG MỨC) -----------------
ax2 = fig.add_subplot(1, 2, 2)
# Log-spaced levels for nice concentric ellipses
levels = np.array([0.05, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 14.0])
contours = ax2.contour(W, B, J, levels=levels, cmap='viridis', linewidths=1.8)
ax2.clabel(contours, inline=True, fontsize=9, fmt='J=%.2f')

# Mark minimum
ax2.scatter([2.0], [1.0], color='red', marker='x', s=140, linewidth=3, zorder=10, label='Tâm đường đồng mức: Đáy cực tiểu\n(w=2.0, b=1.0, J=0)')
ax2.annotate('Đáy cực tiểu!\n(Tâm các vòng elip)', xy=(2.0, 1.0), xytext=(2.3, 1.6),
             arrowprops=dict(arrowstyle='->', color='red', lw=2),
             fontsize=11, fontweight='bold', color='#b91c1c')

# Highlight multiple points on the SAME contour line to demonstrate "Same Height"
# Point A & Point B on contour J ≈ 1.0
ax2.set_title('2. Bản đồ Đường đồng mức: Contour Plot\n(Các lát cắt ngang nhìn từ trên máy bay xuống)', fontsize=13, fontweight='bold', pad=12)
ax2.set_xlabel('Tham số w', fontsize=11, labelpad=8)
ax2.set_ylabel('Tham số b', fontsize=11, labelpad=8)
ax2.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()

# Save image to repo
out_repo_path = os.path.join(os.getcwd(), 'course-1-supervised-machine-learning', 'week-1-linear-regression', 'cost_3d_contour.png')
plt.savefig(out_repo_path, bbox_inches='tight')
print(f"Saved image to: {out_repo_path}")

# Copy to artifact directory for rendering in IDE
artifact_dir = r"C:\Users\MY MSI\.gemini\antigravity-ide\brain\e29858b5-935f-4a1f-b451-1cc2e8c5608d"
if os.path.exists(artifact_dir):
    dest = os.path.join(artifact_dir, 'cost_3d_contour.png')
    shutil.copyfile(out_repo_path, dest)
    print(f"Copied to artifact dir: {dest}")
