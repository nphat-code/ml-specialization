# Machine Learning Specialization — Andrew Ng (Coursera & DeepLearning.AI)

Repository này ghi lại toàn bộ lộ trình học tập, bài giải lab, ghi chú toán học và code thực hành của khóa học **Machine Learning Specialization**.

---

## 🎯 Vai trò & Nguyên tắc Hướng dẫn của AI Assistant (Pair Programmer & 1-on-1 Tutor)

Khi học tập và trao đổi trong repository này, AI Assistant tuân theo các nguyên tắc cốt lõi sau:
1. **Trực giác trước, Công thức sau (Intuition First):** Luôn giải thích trực quan bản chất hình học, vật lý hoặc logic thực tế trước khi đi vào các phương trình toán học phức tạp.
2. **Tư duy Vector hóa (NumPy Vectorization):** Luôn khuyến khích và hướng dẫn cách biểu diễn phép tính dưới dạng ma trận/vector thay vì dùng vòng lặp `for` lồng nhau.
3. **Phương pháp gợi mở (Socratic Method):** Khi gặp bài lab hoặc câu hỏi quiz, trợ lý sẽ chỉ ra vị trí lỗi, đặt câu hỏi gợi mở và phân tích nguyên lý để người học tự làm chủ kiến thức, không spoil thẳng đáp án.
4. **Clean Code & Thực hành chuẩn:** Code được viết rõ ràng, có chú thích kích thước ma trận `(m, n)` hoặc `(m, 1)` để tránh lỗi `shape mismatch`.
5. **Quy trình học 3 bước (3-Step Learning Loop):**
   - **Bước 1 (Trước khi xem):** AI cung cấp **Preview nhanh (30s - 1 phút)** nêu rõ vấn đề cần giải quyết và các khái niệm/từ khóa then chốt cần chú ý.
   - **Bước 2 (Trong lúc xem):** Người học xem video bài giảng của Thầy Andrew Ng để nạp trực giác qua hình vẽ và lời giảng.
   - **Bước 3 (Sau khi xem):** Người học gửi transcript $\rightarrow$ AI tổng hợp **Notes chi tiết** (công thức LaTeX, bảng so sánh), cập nhật checklist và tự động commit/push GitHub.


---

## 🗺️ Lộ trình & Bảng Theo Dõi Tiến Độ (Progress Tracker)

### Course 1: Supervised Machine Learning: Regression and Classification
- [ ] **Week 1: Introduction to Machine Learning & Linear Regression**
  - [x] Khái niệm Supervised vs Unsupervised Learning
  - [x] Model representation: $f_{w, b}(x) = wx + b$
  - [ ] Cost function: Squared error cost function $J(w, b)$
  - [ ] Gradient descent algorithm & Learning rate $\alpha$
  - [ ] Lab: Linear Regression with one variable
- [ ] **Week 2: Multiple Linear Regression & Practical Tricks**
  - [ ] Vectorized multiple linear regression: $f_{\mathbf{w}, b}(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b$
  - [ ] Feature scaling & Mean normalization (Z-score normalization)
  - [ ] Checking gradient descent for convergence & Learning rate selection
  - [ ] Feature engineering & Polynomial regression
  - [ ] Lab: Multiple Linear Regression
- [ ] **Week 3: Classification & Logistic Regression**
  - [ ] Motivation for Logistic Regression & Sigmoid function $g(z)$
  - [ ] Decision boundary
  - [ ] Logistic loss function & Cost function
  - [ ] Gradient descent for Logistic Regression
  - [ ] Overfitting & Regularization (L2 Regularization / Cost function penalty $\lambda$)
  - [ ] Lab: Logistic Regression & Regularization

---

### Course 2: Advanced Learning Algorithms
- [ ] **Week 1: Neural Networks Intuition & Architecture**
  - [ ] Neurons and the brain analogy vs mathematical reality
  - [ ] Layer notation, activations, forward propagation
  - [ ] Vectorized implementation of forward propagation
  - [ ] Lab: Neural Networks for multi-class classification
- [ ] **Week 2: Neural Network Training**
  - [ ] Activation functions (ReLU, Sigmoid, Linear, Softmax)
  - [ ] Multi-class classification & Categorical cross-entropy
  - [ ] Backpropagation intuition & Gradient computation
  - [ ] Lab: Backpropagation & Neural Network training with TensorFlow/Keras
- [ ] **Week 3: Advice for Applying Machine Learning**
  - [ ] Evaluating a model: Train / Validation / Test sets
  - [ ] Bias vs. Variance diagnosis (High bias = underfitting, High variance = overfitting)
  - [ ] Learning curves & Regularization parameter $\lambda$ tuning
  - [ ] Error analysis & Data-centric approach
- [ ] **Week 4: Decision Trees**
  - [ ] Decision tree model & Entropy / Information Gain
  - [ ] Continuous features & Multi-class splits
  - [ ] Tree ensembles: Random Forests & Boosted Trees (XGBoost)
  - [ ] Lab: Decision Trees & XGBoost

---

### Course 3: Unsupervised Learning, Recommenders, Reinforcement Learning
- [ ] **Week 1: Unsupervised Learning (Clustering & Anomaly Detection)**
  - [ ] K-Means clustering algorithm & Choosing number of clusters (Elbow method)
  - [ ] Anomaly detection with Gaussian distribution
  - [ ] Anomaly detection vs Supervised learning
  - [ ] Lab: K-Means & Anomaly Detection
- [ ] **Week 2: Recommender Systems**
  - [ ] Collaborative filtering algorithm
  - [ ] Content-based filtering & Deep learning for recommendations
  - [ ] Principal Component Analysis (PCA) intuition & Dimensionality reduction
  - [ ] Lab: Collaborative Filtering & Movie Recommender
- [ ] **Week 3: Reinforcement Learning**
  - [ ] What is Reinforcement Learning: State, Action, Reward, Discount factor $\gamma$
  - [ ] Markov Decision Process (MDP) & Bellman Equation
  - [ ] State-action value function (Q-function)
  - [ ] Deep Q-Learning (DQN) & Lunar Lander project
  - [ ] Lab: Deep Q-Learning (Lunar Lander)

---

## 📁 Cấu trúc Thư mục Đề xuất

```text
ml-specialization/
├── README.md                                          # Mục tiêu, vai trò và tiến độ học tập
├── notes/                                             # Ghi chú tổng hợp lý thuyết, công thức toán
├── course-1-supervised-machine-learning/
│   ├── week-1-linear-regression/
│   ├── week-2-multiple-linear-regression/
│   └── week-3-classification/
├── course-2-advanced-learning-algorithms/
│   ├── week-1-neural-networks/
│   ├── week-2-neural-network-training/
│   ├── week-3-advice-for-applying-ml/
│   └── week-4-decision-trees/
└── course-3-unsupervised-learning-recommenders-reinforcement-learning/
    ├── week-1-unsupervised-learning/
    ├── week-2-recommender-systems/
    └── week-3-reinforcement-learning/
```
