# Course 1: Supervised Machine Learning — Week 2
# Ghi chú Lý thuyết: Multiple Linear Regression & Practical Tricks

Tài liệu ghi chép chi tiết, trực quan theo phương pháp Fast-Track cho tuần học thứ 2.

---

## 1. Multiple Features (Hồi quy Tuyến tính Đa biến / Nhiều Đặc trưng)

### 🎯 Đặt vấn đề:
Ở Tuần 1, ta chỉ dự đoán giá nhà dựa trên một đặc trưng duy nhất là diện tích ($x$). Nhưng trong thực tế, giá nhà phụ thuộc vào rất nhiều yếu tố khác nhau:
- Diện tích sàn ($x_1$ - sqft)
- Số phòng ngủ ($x_2$)
- Số tầng ($x_3$)
- Tuổi thọ của ngôi nhà ($x_4$ - năm)

### 📐 Hệ thống ký hiệu toán học chuẩn mực (Mathematical Notation):
- $n$: Số lượng đặc trưng (Number of features). Ở ví dụ trên, $n = 4$.
- $m$: Số lượng mẫu dữ liệu huấn luyện (Number of training examples).
- $\mathbf{x}^{(i)}$: Vector chứa **toàn bộ các đặc trưng** của mẫu dữ liệu thứ $i$.
  - Đây là một vector hàng (hoặc vector 1D kích thước $n$):
  $$\mathbf{x}^{(i)} = \begin{bmatrix} x_1^{(i)} & x_2^{(i)} & \dots & x_n^{(i)} \end{bmatrix}$$
  - Ví dụ: $\mathbf{x}^{(2)} = [1416, 3, 2, 40]$ nghĩa là ngôi nhà thứ 2 rộng 1416 sqft, 3 phòng ngủ, 2 tầng, 40 năm tuổi.
- $x_j^{(i)}$: Giá trị của **đặc trưng thứ $j$** nằm trong **mẫu dữ liệu thứ $i$**.
  - Ví dụ: $x_3^{(2)} = 2$ (đặc trưng thứ 3 - số tầng của ngôi nhà thứ 2).

### 🏷️ Mô hình Hồi quy tuyến tính Đa biến (Model Representation):
- **Dạng đại số thông thường:**
  $$f(x) = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b$$
- **Dạng Vector (Vectorized Form):**
  - Đặt vector trọng số: $\mathbf{w} = \begin{bmatrix} w_1 & w_2 & \dots & w_n \end{bmatrix}$
  - Đặt vector đầu vào: $\mathbf{x} = \begin{bmatrix} x_1 & x_2 & \dots & x_n \end{bmatrix}$
  - $b$: Giá trị vô hướng (scalar - bias / intercept).
  - Mô hình được viết gọn gàng thành **Tích vô hướng (Dot Product)**:
  $$f_{\mathbf{w}, b}(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b = \left(\sum_{j=1}^n w_j x_j\right) + b$$

---

## 2. Vectorization (Kỹ thuật Vector hóa với NumPy)

### ⚡ Tại sao Vectorization là "vũ khí tối thượng" trong Machine Learning?
Trong Machine Learning và Deep Learning, ta thường phải tính toán trên hàng trăm ngàn đặc trưng và hàng triệu mẫu dữ liệu.

![So sánh Tốc độ For loop vs Vectorization](vectorization_benchmark.png)

#### So sánh 2 cách code:
1. **Không vector hóa (Dùng vòng lặp `for` — Rất chậm):**
   ```python
   f = 0
   for j in range(n):
       f = f + w[j] * x[j]
   f = f + b
   ```
2. **Vector hóa với NumPy (`np.dot` — Cực nhanh & code ngắn 1 dòng):**
   ```python
   f = np.dot(w, x) + b
   ```

### 🖥️ Bản chất phần cứng: Tại sao `np.dot` nhanh gấp hàng chục đến hàng trăm lần?
- **Vòng lặp `for` (Tuần tự):** CPU chỉ dùng 1 luồng tính toán, phải nhân $w_0 \times x_0$, lưu kết quả tạm, rồi sang bước lặp kế tiếp nhân $w_1 \times x_1$... lặp lại $n$ lần.
- **NumPy `np.dot` (Song song phần cứng - Hardware Parallelism):**
  - Tận dụng các lệnh tối ưu hóa cao cấp trong CPU và GPU (tập lệnh **SIMD: Single Instruction, Multiple Data** như AVX/SSE trên Intel/AMD hay CUDA trên NVIDIA GPU).
  - Máy tính nạp nhiều cặp $(w_j, x_j)$ vào các thanh ghi vector chuyên dụng cùng lúc và **thực hiện phép nhân đồng loạt trong một chu kỳ xung nhịp**, sau đó cộng dồn với tốc độ ánh sáng!

---

## 3. Gradient Descent for Multiple Linear Regression

### 🥣 Hàm chi phí (Cost Function):
$$J(\mathbf{w}, b) = \frac{1}{2m} \sum_{i=1}^m \left( f_{\mathbf{w}, b}(\mathbf{x}^{(i)}) - y^{(i)} \right)^2$$
Trong đó $f_{\mathbf{w}, b}(\mathbf{x}^{(i)}) = \mathbf{w} \cdot \mathbf{x}^{(i)} + b$.

### 🔄 Thuật toán cập nhật Gradient Descent:
Lặp lại các bước sau cho đến khi hội tụ (Convergence):
$$\text{repeat until convergence: } \{$$
$$w_j := w_j - \alpha \frac{\partial J(\mathbf{w}, b)}{\partial w_j} \quad (\text{cho mọi } j = 1, 2, \dots, n)$$
$$b := b - \alpha \frac{\partial J(\mathbf{w}, b)}{\partial b}$$
$$\}$$

### 📐 Công thức đạo hàm riêng cụ thể:
1. **Đạo hàm theo từng tham số $w_j$:**
   $$\frac{\partial J(\mathbf{w}, b)}{\partial w_j} = \frac{1}{m} \sum_{i=1}^m \left( f_{\mathbf{w}, b}(\mathbf{x}^{(i)}) - y^{(i)} \right) x_j^{(i)}$$
2. **Đạo hàm theo hệ số tự do $b$:**
   $$\frac{\partial J(\mathbf{w}, b)}{\partial b} = \frac{1}{m} \sum_{i=1}^m \left( f_{\mathbf{w}, b}(\mathbf{x}^{(i)}) - y^{(i)} \right)$$

### ⚠️ Lưu ý sống còn:
- **Simultaneous Update:** Tất cả $n$ giá trị $w_1, w_2, \dots, w_n$ và $b$ phải được tính toán đạo hàm xong xuôi rồi mới cập nhật cùng lúc!
- **Đạo hàm theo $w_j$ có nhân với $x_j^{(i)}$:** Ở tuần 1 chỉ có 1 biến $x$ nên ta nhân với $x^{(i)}$. Ở tuần 2 có $n$ biến, muốn đạo hàm theo $w_j$ nào thì nhân với đúng đặc trưng $x_j^{(i)}$ tương ứng của biến đó!
