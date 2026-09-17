# Course 1: Supervised Machine Learning — Week 2
## Multiple Linear Regression & Practical Tricks

Tài liệu theo dõi tiến độ chi tiết 1:1 theo giáo trình Coursera.

---

### 📌 1. Multiple Linear Regression
- [x] [Multiple features](https://www.coursera.org/learn/machine-learning/lecture/gFuSx/multiple-features) *(9 min)* — Ký hiệu toán học ma trận: $n$ features, vector hàng $\mathbf{x}^{(i)}$, mô hình vector $f_{\mathbf{w},b}(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b$
- [x] [Vectorization part 1](https://www.coursera.org/learn/machine-learning/lecture/ismjc/vectorization-part-1) *(6 min)* — Vì sao cần Vectorization? So sánh vòng lặp `for` vs `np.dot(w, x) + b`
- [x] [Vectorization part 2](https://www.coursera.org/learn/machine-learning/lecture/p2Nqv/vectorization-part-2) *(6 min)* — Tận dụng phần cứng song song (SIMD CPU/GPU) & Phép tính ma trận - vector
- [x] [Optional lab: Python, NumPy and vectorization](https://www.coursera.org/learn/machine-learning/ungradedLab/zadmO/optional-lab-python-numpy-and-vectorization) *(Lab thực hành)*
- [x] [Gradient descent for multiple linear regression](https://www.coursera.org/learn/machine-learning/lecture/ltMMp/gradient-descent-for-multiple-linear-regression) *(7 min)* — Cập nhật đồng thời $n$ tham số $w_j$ và $b$
- [x] [Optional Lab: Multiple linear regression](https://www.coursera.org/learn/machine-learning/ungradedLab/7GEJh/optional-lab-multiple-linear-regression) *(Lab thực hành)*
- [x] **Practice quiz: Multiple linear regression** *(15 min)*

---

### 📌 2. Gradient Descent in Practice
- [x] [Feature scaling part 1](https://www.coursera.org/learn/machine-learning/lecture/KMDV3/feature-scaling-part-1) *(6 min)* — Vấn đề thang đo lệch (Contour quả trứng dài ngoằng làm GD dao động zic-zac)
- [x] [Feature scaling part 2](https://www.coursera.org/learn/machine-learning/lecture/akapu/feature-scaling-part-2) *(7 min)* — Các phương pháp: Max scaling, Mean normalization, Z-score normalization
- [x] [Checking gradient descent for convergence](https://www.coursera.org/learn/machine-learning/lecture/rOTkB/checking-gradient-descent-for-convergence) *(5 min)* — Đồ thị Learning Curve $J$ theo số vòng lặp & $\epsilon$-test (Automatic convergence test)
- [x] [Choosing the learning rate](https://www.coursera.org/learn/machine-learning/lecture/10ZVv/choosing-the-learning-rate) *(6 min)* — Chiến thuật thử nghiệm $\alpha$ theo cấp số nhân: $..., 0.001, 0.003, 0.01, 0.03, 0.1, ...$
- [ ] [Optional Lab: Feature scaling and learning rate](https://www.coursera.org/learn/machine-learning/ungradedLab/kIf25/optional-lab-feature-scaling-and-learning-rate) *(Lab thực hành)*
- [x] [Feature engineering](https://www.coursera.org/learn/machine-learning/lecture/dgZYR/feature-engineering) *(3 min)* — Tự thiết kế đặc trưng mới (ví dụ: Diện tích = Chiều rộng $\times$ Chiều sâu)
- [x] [Polynomial regression](https://www.coursera.org/learn/machine-learning/lecture/OnGhN/polynomial-regression) *(5 min)* — Hồi quy đa thức: Uốn cong hàm dự đoán với $x^2, x^3, \sqrt{x}$
- [ ] [Optional lab: Feature engineering and Polynomial regression](https://www.coursera.org/learn/machine-learning/ungradedLab/Xat0X/optional-lab-feature-engineering-and-polynomial-regression) *(Lab thực hành)*
- [ ] [Optional lab: Linear regression with scikit-learn](https://www.coursera.org/learn/machine-learning/ungradedLab/uaIsm/optional-lab-linear-regression-with-scikit-learn) *(Lab thực hành Scikit-learn)*
- [ ] **Practice quiz: Gradient descent in practice** *(30 min)*

---

### 📌 3. Week 2 Practice Lab (Graded Programming Assignment)
- [ ] **[Week 2 practice lab: Linear regression](https://www.coursera.org/learn/machine-learning/programming/jsE7w/week-2-practice-lab-linear-regression)** *(Bài tập lập trình tính điểm chính thức - 3h)*
