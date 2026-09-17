import numpy as np
import matplotlib.pyplot as plt

def plot_feature_scaling():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    
    # 1. Unscaled Contour: w1 (size in sqft, range [300, 2000]) vs w2 (bedrooms [1, 5])
    # Very skewed scales: J is very sensitive to w1, insensitive to w2 -> very tall, skinny ellipse
    w1_unscaled = np.linspace(-0.2, 0.2, 200)
    w2_unscaled = np.linspace(-20, 20, 200)
    W1_u, W2_u = np.meshgrid(w1_unscaled, w2_unscaled)
    J_unscaled = (W1_u * 50)**2 + (W2_u * 0.8)**2

    ax1.contour(W1_u, W2_u, J_unscaled, levels=[5, 20, 50, 100, 200, 400], colors='#34495e', linewidths=1.2)
    ax1.plot(0, 0, 'r*', markersize=14, label='Đáy cực tiểu (Minimum)')
    
    # Zigzag trajectory for unscaled
    path_w1 = [-0.18, 0.15, -0.12, 0.09, -0.06, 0.04, -0.02, 0.0]
    path_w2 = [18.0, 15.0, 12.0, 9.0, 6.0, 4.0, 2.0, 0.0]
    ax1.plot(path_w1, path_w2, 'r-o', linewidth=1.5, markersize=5, label='Đường đi Gradient Descent (Dao động zic-zac)')
    
    ax1.set_title("CHƯA CHUẨN HÓA (Unscaled Features)\nĐường đồng mức bị dẹt dài ngoặc -> GD nhảy zic-zac rất chậm", 
                  fontsize=11, fontweight='bold', color='#c0392b')
    ax1.set_xlabel("Trọng số w1 (diện tích ~ 1000 sqft)", fontsize=10, fontweight='bold')
    ax1.set_ylabel("Trọng số w2 (số phòng ngủ ~ 1-5)", fontsize=10, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right', fontsize=9)

    # 2. Scaled Contour: Normalized features with similar variance -> circular symmetric contours
    w1_scaled = np.linspace(-10, 10, 200)
    w2_scaled = np.linspace(-10, 10, 200)
    W1_s, W2_s = np.meshgrid(w1_scaled, w2_scaled)
    J_scaled = W1_s**2 + W2_s**2

    ax2.contour(W1_s, W2_s, J_scaled, levels=[5, 20, 40, 70, 100, 150], colors='#34495e', linewidths=1.2)
    ax2.plot(0, 0, 'r*', markersize=14, label='Đáy cực tiểu (Minimum)')
    
    # Direct straight trajectory for scaled
    path_s1 = np.linspace(-8, 0, 7)
    path_s2 = np.linspace(8, 0, 7)
    ax2.plot(path_s1, path_s2, 'g-o', linewidth=2.0, markersize=6, label='Đường đi Gradient Descent (Thẳng tiến về đích)')
    
    ax2.set_title("ĐÃ CHUẨN HÓA (Scaled Features - Z-score)\nĐường đồng mức tròn đều -> GD lao thẳng đến cực tiểu cực nhanh", 
                  fontsize=11, fontweight='bold', color='#27ae60')
    ax2.set_xlabel("Trọng số w1 (đã chuẩn hóa ~ [-1, 1])", fontsize=10, fontweight='bold')
    ax2.set_ylabel("Trọng số w2 (đã chuẩn hóa ~ [-1, 1])", fontsize=10, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right', fontsize=9)

    plt.suptitle("Tác Động Của Feature Scaling Lên Thuật Toán Gradient Descent", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig("feature_scaling_effect.png")
    print("Successfully generated feature_scaling_effect.png")

if __name__ == '__main__':
    plot_feature_scaling()
