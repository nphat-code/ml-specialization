# Course 1: Supervised Machine Learning — Week 3
## Classification & Logistic Regression

Tài liệu theo dõi tiến độ chi tiết 1:1 theo giáo trình Coursera.

---

### 📌 1. Classification with Logistic Regression
- [x] [Motivations](https://www.coursera.org/learn/machine-learning/lecture/Z95y4/motivations) *(7 min)* — Vì sao Linear Regression thất bại khi phân loại nhị phân ($y \in \{0, 1\}$)?
- [x] [Logistic regression](https://www.coursera.org/learn/machine-learning/lecture/0m3X4/logistic-regression) *(8 min)* — Hàm Sigmoid/Logistic: $g(z) = \frac{1}{1 + e^{-z}}$, mô hình xác suất $f_{\mathbf{w},b}(\mathbf{x}) = g(\mathbf{w} \cdot \mathbf{x} + b) = P(y=1|\mathbf{x})$
- [x] [Decision boundary](https://www.coursera.org/learn/machine-learning/lecture/WqEeh/decision-boundary) *(8 min)* — Ranh giới quyết định tuyến tính & phi tuyến: $\mathbf{w} \cdot \mathbf{x} + b = 0$
- [x] [Optional lab: Classification](https://www.coursera.org/learn/machine-learning/ungradedLab/2hQ2p/optional-lab-classification) *(Lab thực hành)*
- [x] [Optional lab: Sigmoid function and logistic regression](https://www.coursera.org/learn/machine-learning/ungradedLab/5QjM0/optional-lab-sigmoid-function-and-logistic-regression) *(Lab thực hành)*
- [x] [Optional lab: Decision boundary](https://www.coursera.org/learn/machine-learning/ungradedLab/kI0x7/optional-lab-decision-boundary) *(Lab thực hành)*

---

### 📌 2. Cost Function for Logistic Regression
- [ ] [Cost function for logistic regression](https://www.coursera.org/learn/machine-learning/lecture/G5Y8A/cost-function-for-logistic-regression) *(9 min)* — Vấn đề Non-convex của Squared Error & Sự ra đời của hàm Logistic Loss lồi
- [ ] [Simplified cost function for logistic regression](https://www.coursera.org/learn/machine-learning/lecture/3b14r/simplified-cost-function-for-logistic-regression) *(7 min)* — Biểu thức hàm chi phí tổng hợp nhị phân (Binary Cross-Entropy Loss)
- [ ] [Gradient descent for logistic regression](https://www.coursera.org/learn/machine-learning/lecture/q1W9n/gradient-descent-for-logistic-regression) *(6 min)* — Cập nhật gradient descent: công thức đạo hàm tương tự nhưng $f$ là hàm Sigmoid
- [ ] [Optional lab: Logistic loss](https://www.coursera.org/learn/machine-learning/ungradedLab/rN2x0/optional-lab-logistic-loss) *(Lab thực hành)*
- [ ] [Optional lab: Cost function for logistic regression](https://www.coursera.org/learn/machine-learning/ungradedLab/WJ2o5/optional-lab-cost-function-for-logistic-regression) *(Lab thực hành)*
- [ ] [Optional lab: Gradient descent for logistic regression](https://www.coursera.org/learn/machine-learning/ungradedLab/lPz2N/optional-lab-gradient-descent-for-logistic-regression) *(Lab thực hành)*
- [ ] [Optional lab: Logistic regression with scikit-learn](https://www.coursera.org/learn/machine-learning/ungradedLab/1O3w3/optional-lab-logistic-regression-with-scikit-learn) *(Lab thực hành Scikit-learn)*
- [ ] **Practice quiz: Classification with logistic regression** *(30 min)*

---

### 📌 3. The Problem of Overfitting & Regularization
- [ ] [The problem of overfitting](https://www.coursera.org/learn/machine-learning/lecture/pI8yF/the-problem-of-overfitting) *(9 min)* — Underfitting (High bias) vs Generalization vs Overfitting (High variance)
- [ ] [Addressing overfitting](https://www.coursera.org/learn/machine-learning/lecture/vQ3n9/addressing-overfitting) *(3 min)* — 3 chiến lược: Thêm dữ liệu, Chọn lọc thuộc tính, Kỹ thuật Regularization
- [ ] [Cost function with regularization](https://www.coursera.org/learn/machine-learning/lecture/vK2N5/cost-function-with-regularization) *(7 min)* — Thêm số hạng phạt trọng số: $\frac{\lambda}{2m} \sum_{j=1}^n w_j^2$
- [ ] [Regularized linear regression](https://www.coursera.org/learn/machine-learning/lecture/7jR1p/regularized-linear-regression) *(7 min)* — Hiệu ứng Shrinkage: $w_j(1 - \alpha \frac{\lambda}{m})$
- [ ] [Regularized logistic regression](https://www.coursera.org/learn/machine-learning/lecture/s1Lw0/regularized-logistic-regression) *(6 min)* — Áp dụng Regularization cho hàm mất mát phân loại nhị phân
- [ ] [Optional lab: Overfitting](https://www.coursera.org/learn/machine-learning/ungradedLab/UeKzP/optional-lab-overfitting) *(Lab thực hành)*
- [ ] [Optional lab: Regularization](https://www.coursera.org/learn/machine-learning/ungradedLab/71lF8/optional-lab-regularization) *(Lab thực hành)*
- [ ] **Practice quiz: Overfitting and regularization** *(30 min)*

---

### 📌 4. Week 3 Practice Lab (Graded Programming Assignment)
- [ ] **[Week 3 practice lab: Logistic regression](https://www.coursera.org/learn/machine-learning/programming/dK2i0/week-3-practice-lab-logistic-regression)** *(Bài tập lập trình tính điểm kết thúc Course 1 - 3h)*
