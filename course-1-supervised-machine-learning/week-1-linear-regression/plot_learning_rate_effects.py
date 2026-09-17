import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

# Cost function: J(w) = w^2 (minimum at w = 0)
w = np.linspace(-4.5, 4.5, 200)
J = w**2

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5), dpi=150)

# ----------------- 1. ALPHA TOO SMALL -----------------
ax1.plot(w, J, color='#8b5cf6', linewidth=2)
# Tiny steps
w_curr = 4.0
path1 = [w_curr]
for _ in range(6):
    w_curr = w_curr - 0.05 * (2 * w_curr) # alpha = 0.05 (too small)
    path1.append(w_curr)
ax1.plot(path1, [x**2 for x in path1], 'ro-', linewidth=1.5, markersize=5)
ax1.set_title('1. Learning rate alpha quá nhỏ\n(Bước đi rùa bò - Quá chậm)', fontsize=12, fontweight='bold', pad=10)
ax1.set_xlabel('Tham số w', fontsize=10)
ax1.set_ylabel('J(w)', fontsize=10)
ax1.annotate('Các bước nhảy tí hon,\ncực kỳ lâu mới tới đáy!', xy=(path1[3], path1[3]**2), xytext=(0.5, 12),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5), fontsize=9.5, color='red')
ax1.grid(True, linestyle='--', alpha=0.4)

# ----------------- 2. ALPHA JUST RIGHT -----------------
ax2.plot(w, J, color='#8b5cf6', linewidth=2)
# Good steps
w_curr = 4.0
path2 = [w_curr]
for _ in range(5):
    w_curr = w_curr - 0.35 * (2 * w_curr) # alpha = 0.35 (just right)
    path2.append(w_curr)
ax2.plot(path2, [x**2 for x in path2], 'go-', linewidth=1.8, markersize=6)
ax2.scatter([0.0], [0.0], color='#10b981', marker='*', s=220, zorder=6, label='Cực tiểu (Đáy)')
ax2.set_title('2. Learning rate alpha chuẩn xác\n(Hội tụ nhanh & mượt mà)', fontsize=12, fontweight='bold', pad=10)
ax2.set_xlabel('Tham số w', fontsize=10)
ax2.annotate('Hội tụ êm đềm\nvề đúng đáy thung lũng!', xy=(0.0, 0.0), xytext=(-3.5, 8),
             arrowprops=dict(arrowstyle='->', color='#10b981', lw=1.8), fontsize=9.5, fontweight='bold', color='#065f46')
ax2.grid(True, linestyle='--', alpha=0.4)

# ----------------- 3. ALPHA TOO LARGE -----------------
ax3.plot(w, J, color='#8b5cf6', linewidth=2)
# Overshooting / Diverging steps
path3_w = [1.0, -1.8, 3.2, -4.8]
path3_J = [x**2 for x in path3_w]
ax3.plot(path3_w, path3_J, 'mo--', color='#ef4444', linewidth=1.8, markersize=6)
ax3.set_title('3. Learning rate alpha quá lớn\n(Vượt qua đáy - Phân kỳ Divergence)', fontsize=12, fontweight='bold', pad=10)
ax3.set_xlabel('Tham số w', fontsize=10)
ax3.annotate('Nhảy vượt qua đáy,\nvăng ra xa (Diverge)!', xy=(3.2, 3.2**2), xytext=(-3.8, 14),
             arrowprops=dict(arrowstyle='->', color='#ef4444', lw=1.8), fontsize=9.5, fontweight='bold', color='#b91c1c')
ax3.set_ylim(-1, 25)
ax3.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()

out_repo_path = os.path.join(os.getcwd(), 'course-1-supervised-machine-learning', 'week-1-linear-regression', 'learning_rate_effects.png')
plt.savefig(out_repo_path, bbox_inches='tight')
print(f"Saved image to: {out_repo_path}")

artifact_dir = r"C:\Users\MY MSI\.gemini\antigravity-ide\brain\e29858b5-935f-4a1f-b451-1cc2e8c5608d"
if os.path.exists(artifact_dir):
    dest = os.path.join(artifact_dir, 'learning_rate_effects.png')
    shutil.copyfile(out_repo_path, dest)
    print(f"Copied to artifact dir: {dest}")
