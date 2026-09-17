import numpy as np
import time
import matplotlib.pyplot as plt

def benchmark_vectorization():
    n = 1_000_000 # 1 million features / elements
    a = np.random.rand(n)
    b = np.random.rand(n)

    # 1. Non-vectorized (for loop)
    tic = time.time()
    c_loop = 0.0
    for i in range(n):
        c_loop += a[i] * b[i]
    toc = time.time()
    time_loop = (toc - tic) * 1000 # convert to ms

    # 2. Vectorized (np.dot)
    tic = time.time()
    c_dot = np.dot(a, b)
    toc = time.time()
    time_dot = (toc - tic) * 1000 # convert to ms

    speedup = time_loop / time_dot if time_dot > 0 else 0

    print(f"For loop time: {time_loop:.2f} ms")
    print(f"NumPy dot time: {time_dot:.2f} ms")
    print(f"Speedup: {speedup:.1f}x faster!")

    # Plotting comparison
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)

    bars = ax.bar(['Vòng lặp for\n(Non-vectorized)', 'NumPy np.dot()\n(Vectorized - SIMD)'], 
                   [time_loop, time_dot], 
                   color=['#e74c3c', '#2ecc71'], 
                   width=0.45, edgecolor='black', linewidth=1.2)

    ax.set_ylabel('Thời gian thực thi (mili-giây, ms)', fontsize=12, fontweight='bold')
    ax.set_title(f'So sánh Tốc độ: Vòng lặp For vs NumPy Vectorization (N = 1.000.000 phần tử)\nNhanh hơn gấp ~{speedup:.0f} lần nhờ tính toán song song phần cứng (SIMD)', 
                 fontsize=13, fontweight='bold', pad=15)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Annotations on bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + (time_loop * 0.02), 
                f'{yval:.2f} ms', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig('vectorization_benchmark.png')
    print("Plot saved to vectorization_benchmark.png")

if __name__ == '__main__':
    benchmark_vectorization()
