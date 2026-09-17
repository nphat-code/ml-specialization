import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

fig = plt.figure(figsize=(15, 6), dpi=150)

# ----------------- 1. NON-CONVEX LANDSCAPE (Neural Networks) -----------------
# Create a surface with 2 distinct valleys (local minima)
x = np.linspace(-2.5, 2.5, 120)
y = np.linspace(-2.5, 2.5, 120)
X, Y = np.meshgrid(x, y)
# Double-well potential function
Z_nonconvex = (X**2 + Y - 11)**2 + (X + Y**2 - 7)**2

ax1 = fig.add_subplot(1, 2, 1)
levels = np.logspace(0.5, 3.2, 20)
ax1.contour(X, Y, Z_nonconvex, levels=levels, cmap='viridis_r', alpha=0.7)

# Path 1 (Blue) -> Local Minimum 1
path1_x = [-1.5, -1.8, -2.1, -2.5, -2.8, -3.1, -3.4, -3.77]
path1_y = [2.0, 1.7, 1.4, 1.0, 0.4, -0.2, -1.5, -3.28] # approximate path
ax1.plot([-1.2, -1.8, -2.3, -2.8], [1.5, 1.8, 2.3, 3.1], 'bo-', linewidth=2.2, markersize=5, label='Khởi đầu 1 -> Lăn vào Thung lũng 1')
ax1.scatter([-1.2], [1.5], color='blue', s=120, edgecolors='black', zorder=5)
ax1.scatter([-2.8], [3.1], color='blue', marker='*', s=250, edgecolors='black', zorder=6, label='Cực tiểu cục bộ 1 (Local Min 1)')

# Path 2 (Red) -> Local Minimum 2 (just slightly different start!)
ax1.plot([0.2, 0.8, 1.6, 2.3, 3.0], [1.2, 1.0, 0.7, 0.4, 0.0], 'ro-', linewidth=2.2, markersize=5, label='Khởi đầu 2 -> Lăn vào Thung lũng 2')
ax1.scatter([0.2], [1.2], color='red', s=120, edgecolors='black', zorder=5)
ax1.scatter([3.0], [0.0], color='red', marker='*', s=250, edgecolors='black', zorder=6, label='Cực tiểu cục bộ 2 (Local Min 2)')

ax1.set_title('1. Hàm Phi Lồi (Non-Convex - Ví dụ Neural Networks)\nKhởi đầu khác nhau dẫn đến các Thung lũng (Cực tiểu) khác nhau!', fontsize=12, fontweight='bold', pad=12)
ax1.set_xlabel('Tham số w', fontsize=11)
ax1.set_ylabel('Tham số b', fontsize=11)
ax1.set_xlim(-3.5, 3.5)
ax1.set_ylim(-3.5, 3.5)
ax1.legend(loc='lower left', fontsize=9, framealpha=0.9)
ax1.grid(True, linestyle='--', alpha=0.4)

# ----------------- 2. CONVEX BOWL (Linear Regression) -----------------
# Squared error function forms a strictly convex bowl
W = np.linspace(-3.0, 3.0, 120)
B = np.linspace(-3.0, 3.0, 120)
W_grid, B_grid = np.meshgrid(W, B)
Z_convex = 1.5 * W_grid**2 + 0.8 * B_grid**2

ax2 = fig.add_subplot(1, 2, 2)
levels_convex = np.linspace(0.2, 18.0, 12)
ax2.contour(W_grid, B_grid, Z_convex, levels=levels_convex, cmap='viridis_r', alpha=0.7)

# Multiple paths all converging to the exact same center (0, 0)!
# Path A (Green)
ax2.plot([2.5, 1.8, 1.1, 0.5, 0.0], [2.2, 1.4, 0.8, 0.3, 0.0], 'go-', linewidth=2.2, markersize=5, label='Khởi đầu A (2.5, 2.2)')
# Path B (Purple)
ax2.plot([-2.6, -1.7, -0.9, -0.3, 0.0], [1.5, 1.0, 0.5, 0.1, 0.0], 'mo-', linewidth=2.2, markersize=5, label='Khởi đầu B (-2.6, 1.5)')
# Path C (Orange)
ax2.plot([-1.5, -0.9, -0.4, -0.1, 0.0], [-2.5, -1.6, -0.8, -0.2, 0.0], 'yo-', color='#d97706', linewidth=2.2, markersize=5, label='Khởi đầu C (-1.5, -2.5)')

# Global minimum
ax2.scatter([0.0], [0.0], color='#10b981', marker='*', s=300, edgecolors='black', zorder=10, label='Cực tiểu toàn cục duy nhất! (Global Min)')
ax2.annotate('Đáy bát duy nhất!\n(w=0, b=0)', xy=(0.0, 0.0), xytext=(0.4, -0.8),
             arrowprops=dict(arrowstyle='->', color='#10b981', lw=2),
             fontsize=10.5, fontweight='bold', color='#065f46')

ax2.set_title('2. Hàm Lồi (Convex Bowl - Linear Regression)\nMọi điểm xuất phát đều lăn về CÙNG một đáy duy nhất!', fontsize=12, fontweight='bold', pad=12)
ax2.set_xlabel('Tham số w', fontsize=11)
ax2.set_ylabel('Tham số b', fontsize=11)
ax2.set_xlim(-3.0, 3.0)
ax2.set_ylim(-3.0, 3.0)
ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax2.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()

out_repo_path = os.path.join(os.getcwd(), 'course-1-supervised-machine-learning', 'week-1-linear-regression', 'gradient_descent_intuition.png')
plt.savefig(out_repo_path, bbox_inches='tight')
print(f"Saved image to: {out_repo_path}")

artifact_dir = r"C:\Users\MY MSI\.gemini\antigravity-ide\brain\e29858b5-935f-4a1f-b451-1cc2e8c5608d"
if os.path.exists(artifact_dir):
    dest = os.path.join(artifact_dir, 'gradient_descent_intuition.png')
    shutil.copyfile(out_repo_path, dest)
    print(f"Copied to artifact dir: {dest}")
