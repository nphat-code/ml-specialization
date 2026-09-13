# Course 1: Supervised Machine Learning — Week 1
## Introduction to Machine Learning & Linear Regression

Tài liệu theo dõi tiến độ chi tiết 1:1 theo giáo trình Coursera.

---

### 📌 1. Overview of Machine Learning
- [x] [Welcome to machine learning!](https://www.coursera.org/learn/machine-learning/lecture/iYR2y/welcome-to-machine-learning) *(2 min)*
- [x] [Applications of machine learning](https://www.coursera.org/learn/machine-learning/lecture/IjrpM/applications-of-machine-learning) *(4 min)*
- [x] [Intake Survey](https://www.coursera.org/learn/machine-learning/ungradedLti/9TxpS/intake-survey) *(1 min)*
- [x] [DeepLearning.AI Forum](https://www.coursera.org/learn/machine-learning/supplement/nviCw/join-the-deeplearning-ai-forum-to-ask-questions-get-support-or-share-amazing) *(2 min)*

---

### 📌 2. Supervised vs. Unsupervised Machine Learning
- [x] [What is machine learning?](https://www.coursera.org/learn/machine-learning/lecture/PNeuX/what-is-machine-learning) *(5 min)* — Định nghĩa Arthur Samuel (1959) & Câu chuyện cờ đam (Checkers)
- [x] [Supervised learning part 1](https://www.coursera.org/learn/machine-learning/lecture/s91wX/supervised-learning-part-1) *(6 min)* — Bài toán Hồi quy (Regression: dự đoán số liên tục)
- [x] [Supervised learning part 2](https://www.coursera.org/learn/machine-learning/lecture/Q8Vvp/supervised-learning-part-2) *(7 min)* — Bài toán Phân loại (Classification: dự đoán nhãn rời rạc)
- [x] [Unsupervised learning part 1](https://www.coursera.org/learn/machine-learning/lecture/TxO6F/unsupervised-learning-part-1) *(8 min)* — Phân cụm (Clustering: Google News, DNA microarray, Phân khúc khách hàng)
- [ ] [Unsupervised learning part 2](https://www.coursera.org/learn/machine-learning/lecture/jKBHE/unsupervised-learning-part-2) *(3 min)* — Phát hiện bất thường & Giảm chiều (Anomaly Detection & Dimensionality Reduction)
- [ ] [Jupyter Notebooks](https://www.coursera.org/learn/machine-learning/lecture/lwqzq/jupyter-notebooks) *(4 min)*
- [ ] [Python and Jupyter Notebooks Lab](https://www.coursera.org/learn/machine-learning/ungradedLab/rNe84/python-and-jupyter-notebooks) *(Lab thực hành)*
- [ ] **Practice quiz: Supervised vs unsupervised learning** *(15 min)*

---

### 📌 3. Regression Model
- [ ] [Linear regression model part 1](https://www.coursera.org/learn/machine-learning/lecture/1ACA2/linear-regression-model-part-1) *(10 min)* — Ký hiệu toán học: $(x, y)$, $m$, $f_{w,b}(x) = wx + b$
- [ ] [Linear regression model part 2](https://www.coursera.org/learn/machine-learning/lecture/nucNi/linear-regression-model-part-2) *(6 min)* — Mô hình dự đoán giá nhà theo diện tích
- [ ] [Optional lab: Model representation](https://www.coursera.org/learn/machine-learning/ungradedLab/PhN1X/optional-lab-model-representation) *(Lab)*
- [ ] [Cost function formula](https://www.coursera.org/learn/machine-learning/lecture/1Z0TT/cost-function-formula) *(9 min)* — Hàm mất mát bình phương trung bình: $J(w, b) = \frac{1}{2m} \sum_{i=1}^m (f_{w,b}(x^{(i)}) - y^{(i)})^2$
- [ ] [Cost function intuition](https://www.coursera.org/learn/machine-learning/lecture/FthLz/cost-function-intuition) *(15 min)* — Trực giác khi $b=0$, đồ thị $J(w)$ parabol
- [ ] [Visualizing the cost function](https://www.coursera.org/learn/machine-learning/lecture/QI1h6/visualizing-the-cost-function) *(8 min)* — Đồ thị 3D và Đường đồng mức (Contour plots) của $J(w, b)$
- [ ] [Visualization examples](https://www.coursera.org/learn/machine-learning/lecture/Ov8Zt/visualization-examples) *(6 min)*
- [ ] [Optional lab: Cost function](https://www.coursera.org/learn/machine-learning/ungradedLab/udPHh/optional-lab-cost-function) *(Lab)*
- [ ] **Practice quiz: Regression** *(10 min)*

---

### 📌 4. Train the model with gradient descent
- [ ] [Gradient descent](https://www.coursera.org/learn/machine-learning/lecture/2f2PA/gradient-descent) *(8 min)* — Ý tưởng lăn bóng xuống thung lũng tìm cực tiểu
- [ ] [Implementing gradient descent](https://www.coursera.org/learn/machine-learning/lecture/TXDBu/implementing-gradient-descent) *(9 min)* — Cập nhật đồng thời (Simultaneous update): $w := w - \alpha \frac{\partial J}{\partial w}$, $b := b - \alpha \frac{\partial J}{\partial b}$
- [ ] [Gradient descent intuition](https://www.coursera.org/learn/machine-learning/lecture/2EoN6/gradient-descent-intuition) *(7 min)* — Bản chất tiếp tuyến và chiều di chuyển
- [ ] [Learning rate](https://www.coursera.org/learn/machine-learning/lecture/OoP3Y/learning-rate) *(9 min)* — Tác động của $\alpha$ quá nhỏ vs $\alpha$ quá lớn (divergence)
- [ ] [Gradient descent for linear regression](https://www.coursera.org/learn/machine-learning/lecture/lgSMj/gradient-descent-for-linear-regression) *(6 min)* — Công thức đạo hàm cụ thể của hàm bình phương
- [ ] [Running gradient descent](https://www.coursera.org/learn/machine-learning/lecture/349Ay/running-gradient-descent) *(5 min)* — Batch Gradient Descent
- [ ] [Optional lab: Gradient descent](https://www.coursera.org/learn/machine-learning/ungradedLab/lE1al/optional-lab-gradient-descent) *(Lab)*
- [ ] **Practice quiz: Train the model with gradient descent** *(10 min)*
