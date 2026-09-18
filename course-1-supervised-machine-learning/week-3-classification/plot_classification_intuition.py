import numpy as np
import matplotlib.pyplot as plt

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

# --- Subplot 1: Why Linear Regression Fails on Classification ---
x_normal = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y_normal = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# Outlier point
x_outlier = 20
y_outlier = 1

x_all = np.append(x_normal, x_outlier)
y_all = np.append(y_normal, y_outlier)

# Fit line without outlier
w_no_outlier, b_no_outlier = np.polyfit(x_normal, y_normal, 1)
# Fit line with outlier
w_outlier, b_outlier = np.polyfit(x_all, y_all, 1)

x_line = np.linspace(0, 22, 200)

ax1.scatter(x_normal[y_normal == 0], y_normal[y_normal == 0], color='#0096ff', marker='o', s=80, label='Lành tính (y=0)')
ax1.scatter(x_normal[y_normal == 1], y_normal[y_normal == 1], color='#e74c3c', marker='x', s=90, linewidths=2.5, label='Ác tính (y=1)')
ax1.scatter(x_outlier, y_outlier, color='#c0392b', marker='x', s=130, linewidths=3, label='Khối u cực lớn (Outlier)')

ax1.plot(x_line, w_no_outlier * x_line + b_no_outlier, '--', color='#2ecc71', label='Hồi quy không có outlier (Ngưỡng ~ 4.5)')
ax1.plot(x_line, w_outlier * x_line + b_outlier, '-', color='#e67e22', linewidth=2, label='Hồi quy bị outlier kéo lệch (Ngưỡng nhảy > 7.5)')
ax1.axhline(0.5, color='gray', linestyle=':', label='Ngưỡng quyết định 0.5')

ax1.set_ylim(-0.2, 1.2)
ax1.set_xlim(0, 22)
ax1.set_title('Vì sao Linear Regression thất bại với Phân loại?', fontsize=13, fontweight='bold', pad=12)
ax1.set_xlabel('Kích thước khối u (Tumor Size)', fontsize=11)
ax1.set_ylabel('Mục tiêu $y$', fontsize=11)
ax1.legend(loc='lower right', fontsize=9)
ax1.grid(True, alpha=0.3)

# --- Subplot 2: Logistic Regression (Sigmoid Function) ---
z = np.linspace(-10, 10, 200)
g_z = 1 / (1 + np.exp(-z))

ax2.plot(z, g_z, color='#9b59b6', linewidth=2.5, label=r'Hàm Sigmoid: $g(z) = \frac{1}{1 + e^{-z}}$')
ax2.axhline(0.5, color='red', linestyle='--', alpha=0.7, label=r'Ngưỡng phân loại $g(z) = 0.5 \Rightarrow z = 0$')
ax2.axvline(0, color='gray', linestyle=':', alpha=0.7)
ax2.axhline(1.0, color='black', linestyle=':', alpha=0.3)
ax2.axhline(0.0, color='black', linestyle=':', alpha=0.3)

# Shading decision regions
ax2.fill_between(z, 0.5, 1.0, where=(z >= 0), color='#e74c3c', alpha=0.15, label=r'Dự đoán $\hat{y} = 1$ ($z \geq 0$)')
ax2.fill_between(z, 0.0, 0.5, where=(z < 0), color='#3498db', alpha=0.15, label=r'Dự đoán $\hat{y} = 0$ ($z < 0$)')

# Key annotations
ax2.scatter([0], [0.5], color='black', s=60, zorder=5)
ax2.annotate(r'$g(0) = 0.5$', xy=(0, 0.5), xytext=(1.2, 0.42),
             arrowprops=dict(facecolor='black', arrowstyle='->', shrinkA=3))

ax2.set_ylim(-0.05, 1.05)
ax2.set_xlim(-10, 10)
ax2.set_title(r'Hàm Sigmoid ép đầu ra về xác suất $[0, 1]$', fontsize=13, fontweight='bold', pad=12)
ax2.set_xlabel(r'$z = \mathbf{w} \cdot \mathbf{x} + b$', fontsize=11)
ax2.set_ylabel(r'$g(z) = P(y=1|\mathbf{x})$', fontsize=11)
ax2.legend(loc='lower right', fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('course-1-supervised-machine-learning/week-3-classification/classification_intuition.png', dpi=150)
print('Saved classification_intuition.png successfully!')
