import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

# Set clean aesthetic style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

# Data points: (1,1), (2,2), (3,3)
x_data = np.array([1.0, 2.0, 3.0])
y_data = np.array([1.0, 2.0, 3.0])
m = len(x_data)

# ----------------- LEFT PLOT: Model f(x) = w*x -----------------
ax1.scatter(x_data, y_data, color='red', marker='x', s=100, linewidth=2.5, label='Dữ liệu thực tế: (x, y)', zorder=5)

x_line = np.linspace(0, 3.5, 100)
colors = {'w=1.0': '#10b981', 'w=0.5': '#3b82f6', 'w=0.0': '#f59e0b', 'w=-0.5': '#ef4444'}

# Draw lines for different w
for w_val, col in [(-0.5, colors['w=-0.5']), (0.0, colors['w=0.0']), (0.5, colors['w=0.5']), (1.0, colors['w=1.0'])]:
    label = f'f(x) = {w_val}x (w={w_val})' + (' [Khớp tốt nhất!]' if w_val == 1.0 else '')
    ax1.plot(x_line, w_val * x_line, label=label, color=col, linewidth=2.5 if w_val == 1.0 else 1.8)

# Highlight error bars for w=0.5
w_test = 0.5
for xi, yi in zip(x_data, y_data):
    y_pred = w_test * xi
    ax1.plot([xi, xi], [yi, y_pred], color='#3b82f6', linestyle='--', linewidth=1.8, zorder=4)

ax1.set_title('1. Đồ thị Mô hình: f(x) = w * x (với b = 0)', fontsize=14, fontweight='bold', pad=12)
ax1.set_xlabel('x (Biến đầu vào)', fontsize=12, labelpad=8)
ax1.set_ylabel('y (Biến mục tiêu)', fontsize=12, labelpad=8)
ax1.set_xlim(-0.2, 3.5)
ax1.set_ylim(-1.8, 3.5)
ax1.axhline(0, color='gray', linewidth=0.8, alpha=0.6)
ax1.axvline(0, color='gray', linewidth=0.8, alpha=0.6)
ax1.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.5)

# ----------------- RIGHT PLOT: Cost Function J(w) -----------------
# Compute J(w) analytically: J(w) = 1/(2m) * sum((w*x - y)^2)
w_range = np.linspace(-0.8, 2.8, 200)
J_vals = [np.sum((w * x_data - y_data)**2) / (2 * m) for w in w_range]

ax2.plot(w_range, J_vals, color='#8b5cf6', linewidth=2.8, label='Đường cong chi phí J(w) (Parabol)')

# Highlight calculated points
test_w = [-0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
for tw in test_w:
    cost = np.sum((tw * x_data - y_data)**2) / (2 * m)
    col = colors.get(f'w={tw}', '#8b5cf6') if f'w={tw}' in colors else '#6366f1'
    if tw == 1.0:
        ax2.scatter(tw, cost, color='#10b981', s=140, zorder=6, label='Cực tiểu: w=1.0 -> J=0.0')
        ax2.annotate(f'Đáy Parabol!\n(w=1.0, J=0.0)', xy=(tw, cost), xytext=(tw+0.15, cost+0.8),
                     arrowprops=dict(arrowstyle='->', color='#10b981', lw=2),
                     fontsize=11, fontweight='bold', color='#065f46')
    else:
        ax2.scatter(tw, cost, color=col, s=80, zorder=5)
        ax2.annotate(f'w={tw}\nJ={cost:.2f}', xy=(tw, cost), xytext=(tw-0.28 if tw <= 0.5 else tw+0.08, cost+0.4),
                     fontsize=9, color='#374151')

ax2.set_title('2. Đồ thị Hàm Chi Phí: J(w) theo tham số w', fontsize=14, fontweight='bold', pad=12)
ax2.set_xlabel('w (Tham số độ dốc)', fontsize=12, labelpad=8)
ax2.set_ylabel('J(w) (Sai số bình phương trung bình)', fontsize=12, labelpad=8)
ax2.set_xlim(-0.9, 2.9)
ax2.set_ylim(-0.3, 6.0)
ax2.axhline(0, color='gray', linewidth=0.8, alpha=0.6)
ax2.axvline(0, color='gray', linewidth=0.8, alpha=0.6)
ax2.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()

# Save image to repo
out_repo_path = os.path.join(os.getcwd(), 'course-1-supervised-machine-learning', 'week-1-linear-regression', 'cost_function_intuition.png')
plt.savefig(out_repo_path, bbox_inches='tight')
print(f"Saved image to: {out_repo_path}")

# Copy to artifact directory for rendering in IDE
artifact_dir = r"C:\Users\MY MSI\.gemini\antigravity-ide\brain\e29858b5-935f-4a1f-b451-1cc2e8c5608d"
if os.path.exists(artifact_dir):
    dest = os.path.join(artifact_dir, 'cost_function_intuition.png')
    shutil.copyfile(out_repo_path, dest)
    print(f"Copied to artifact dir: {dest}")
